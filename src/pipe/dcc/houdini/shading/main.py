"""SKD MatLib: build RenderMan and USD Preview materials from published textures.

The flow is one straight line:

    published texture files   (textures.published_materials)
    -> MaterialSpec per texture set
    -> MaterialGraphBuilder creates only the nodes each spec needs
       (and `update` keeps whatever an artist edited; see update.py)
    -> the HDA's `errors` LOP checks every texture path still relocates

`matlib_*` and `MatlibErrorChecker` are the SKD_MatLib HDA's entry points; see
the note on `MatlibErrorChecker` before renaming any of them.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import hou
from env_sg import DB_Config

from pipe.core.shotgrid import Asset, ShotGrid

from . import textures, update, variants
from .update import MaterialPlan
from .update import node_name as _node_name

log = logging.getLogger(__name__)

_MATLIB_TYPE = "materiallibrary"
_MATLIB_NAME = "Material_Library"
_NO_TEXTURES = "NO_EXPORTED_TEXTURES"

_GENERATED_KEY = "skd_matlib_generated"
_GENERATED_VALUE = "1"

# RenderMan's material builder ships exactly one output node of type `collect`.
_BUILDER_OUTPUT = "output_collect"
# PxrLayerMixer mixes a base layer plus this many stacked layers.
_MIXER_SLOTS = 4

_MATERIAL_Y_STEP = 3.5
_LAYER_Y_STEP = 8.0
# One node's height in the network editor. Texture readers stack in one column
# and each one's processing node sits level with it in the next.
_NODE_ROW = 1.38
_READER_X = -5.0
_PROCESS_X = -2.0
# The row spacing assumes these nodes show only their wired connectors; the
# rest stay fully expanded.
_CONNECTOR_DETAIL = {
    "pxrtexture::3.0": hou.nodeFlag.InOutDetailLow,
    "pxrnormalmap::3.0": hou.nodeFlag.InOutDetailLow,
    "pxrremap::3.0": hou.nodeFlag.InOutDetailLow,
    "pxrinvert::3.0": hou.nodeFlag.InOutDetailLow,
    "usduvtexture::2.0": hou.nodeFlag.InOutDetailLow,
    "pxrmetallicworkflow::3.0": hou.nodeFlag.InOutDetailMedium,
}
_UV_PRIMVAR = "st"
# RenderMan colour-config aliases. Published `.tex` colour maps are already
# ACEScg, so they are tagged "rendering" rather than converted on read.
_COLOR_SPACE = "rendering"
_DATA_SPACE = "data"


@dataclass(frozen=True)
class _PreviewInput:
    """How one published preview map drives UsdPreviewSurface.

    A usduvtexture output feeds one UsdPreviewSurface input.
    `source_colorspace` is a USD Preview Material spec token, not an OCIO name.
    """

    map_name: str
    surface_input: str
    texture_output: str
    source_colorspace: str
    row: int
    scale: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    bias: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


_PREVIEW_INPUTS = (
    _PreviewInput("BaseColor", "diffuseColor", "rgb", "sRGB", 0),
    _PreviewInput("Metallic", "metallic", "r", "raw", 1),
    _PreviewInput("SpecularRoughness", "roughness", "r", "raw", 2),
    # Normal maps store 0..1; UsdPreviewSurface wants -1..1 tangent space.
    _PreviewInput(
        "Normal",
        "normal",
        "rgb",
        "raw",
        3,
        scale=(2.0, 2.0, 2.0, 1.0),
        bias=(-1.0, -1.0, -1.0, 0.0),
    ),
)

# Shaded in renders only. Houdini's VK viewport ignores a texture's bias, so
# the preview can't remap the mask into a usable opacity.
_RENDER_ONLY_MAPS = frozenset({"Transparency"})


if {preview.map_name for preview in _PREVIEW_INPUTS} != (
    set(textures.MAPS) - _RENDER_ONLY_MAPS
):
    # Drift here is silent: a discovered map with no entry is simply never wired.
    raise ImportError(
        "_PREVIEW_INPUTS and textures.MAPS disagree, so a published preview "
        "map would be discovered and then dropped"
    )


def _parm(node: hou.Node, name: str) -> hou.Parm:
    """A parameter the node type guarantees.

    Missing means the DCC changed under us, so fail loudly with the node path
    rather than silently building a material with the value left at default.
    """
    parm = node.parm(name)
    if parm is None:
        raise hou.OperationFailed(
            f"{node.path()} ({node.type().name()}) has no '{name}' parameter"
        )
    return parm


def _parm_tuple(node: hou.Node, name: str) -> hou.ParmTuple:
    parm_tuple = node.parmTuple(name)
    if parm_tuple is None:
        raise hou.OperationFailed(
            f"{node.path()} ({node.type().name()}) has no '{name}' parameter"
        )
    return parm_tuple


class MaterialGraphBuilder:
    """Creates the generated material graphs inside a `materiallibrary` LOP.

    Every node it creates is tagged. Materials an artist authored by hand are
    left alone, and so are generated ones an artist has since edited.
    """

    def __init__(self, matlib: hou.Node) -> None:
        self._matlib = matlib

    def update(
        self,
        materials: Sequence[textures.MaterialSpec],
        *,
        build_preview: bool,
        own_roots: Sequence[Path],
        read_roots: Sequence[Path],
    ) -> MaterialPlan:
        plan = self.plan(
            materials,
            build_preview=build_preview,
            own_roots=own_roots,
            read_roots=read_roots,
        )
        for parm, path in plan.retarget:
            parm.set(path)
        for builder in plan.remove:
            builder.destroy()
        for builder, material in plan.rebuild:
            position = builder.position()
            builder.destroy()
            self._build_material(material, position, build_preview=build_preview)
        for material in plan.build:
            self._build_material(
                material, (0.0, self._next_free_y()), build_preview=build_preview
            )
        return plan

    def plan(
        self,
        materials: Sequence[textures.MaterialSpec],
        *,
        build_preview: bool,
        own_roots: Sequence[Path],
        read_roots: Sequence[Path],
    ) -> MaterialPlan:
        builders = {
            child.name(): child
            for child in self._matlib.children()
            if child.userData(_GENERATED_KEY) == _GENERATED_VALUE
        }
        return update.plan(
            self._matlib,
            builders,
            materials,
            build_preview=build_preview,
            own_roots=own_roots,
            read_roots=read_roots,
        )

    def _next_free_y(self) -> float:
        lowest = min(
            (child.position().y() for child in self._matlib.children()), default=None
        )
        return 0.0 if lowest is None else lowest - _MATERIAL_Y_STEP

    def _build_material(
        self,
        material: textures.MaterialSpec,
        position: hou.Vector2 | tuple[float, float],
        *,
        build_preview: bool,
    ) -> None:
        builder = self._create(
            self._matlib,
            "pxrmaterialbuilder",
            update.material_node_name(material.texture_set),
            (position[0], position[1]),
        )
        # The HDA's Material Library collects material-flagged children (matflag1).
        builder.setMaterialFlag(True)  # ty:ignore[unresolved-attribute]
        builder.setUserData("wirestyle", "rounded")
        output = self._material_output(builder)

        surface = self._build_renderman_shader(builder, material)
        preview_row = -(len(material.layers) * _LAYER_Y_STEP + 6.0)
        preview_surface = (
            self._build_preview_shader(builder, material, preview_row)
            if build_preview
            else None
        )

        # RenderMan ships the output left of where our surface lands.
        output.setPosition(hou.Vector2(14.0, 0.0))
        output.setInput(0, surface, 0)
        if preview_surface is not None:
            output.setInput(1, preview_surface, 0)
        update.stamp(builder, material, build_preview=build_preview)

    def _build_renderman_shader(
        self, builder: hou.Node, material: textures.MaterialSpec
    ) -> hou.Node:
        suffix = _node_name(material.texture_set)
        mixer = self._create(
            builder, "pxrlayermixer::3.0", f"{suffix}_LayerMixer", (8.0, 0.0)
        )
        surface = self._create(
            builder, "pxrlayersurface::3.0", f"{suffix}_PxrLayerSurface", (11.0, 0.0)
        )
        surface.setInput(0, mixer, 0)

        layer_specs = material.layers
        if len(layer_specs) > _MIXER_SLOTS + 1:
            log.warning(
                "Texture set '%s' publishes %d layers but PxrLayerMixer mixes %d; "
                "the remaining layers are not shaded.",
                material.texture_set,
                len(layer_specs),
                _MIXER_SLOTS + 1,
            )
            layer_specs = layer_specs[: _MIXER_SLOTS + 1]

        for index, layer_spec in enumerate(layer_specs):
            layer = self._build_layer(
                builder,
                layer_spec,
                f"{suffix}_{_node_name(layer_spec.name)}",
                y=-index * _LAYER_Y_STEP,
            )
            slot = "baselayer" if index == 0 else f"layer{index}"
            mixer.setNamedInput(slot, layer, "pxrMaterialOut")

        # Stated for every slot so an unconnected one can never stay enabled.
        for slot in range(1, _MIXER_SLOTS + 1):
            _parm(mixer, f"layer{slot}Enabled").set(slot < len(layer_specs))
        return surface

    def _build_layer(
        self,
        builder: hou.Node,
        layer_spec: textures.LayerSpec,
        suffix: str,
        *,
        y: float,
    ) -> hou.Node:
        """A PxrLayer and the texture chain feeding it.

        A map that was never published gets no node at all: an empty texture node
        renders black and reads as a broken filepath to the shipping check.
        """
        maps = layer_spec.render_maps
        layer = self._create(builder, "pxrlayer::3.0", f"Layer_{suffix}", (1.0, y))
        _parm(layer, "enableSpecular").set(True)
        _parm(layer, "specularGain").set(1.0)
        # RenderMan creates every input group collapsed; open the ones we wire.
        layer.destroyUserData("__inputgroup_Primary Specular")

        base_color = self._texture(
            builder,
            "pxrtexture::3.0",
            f"BaseColor_{suffix}",
            maps.get("BaseColor"),
            (_READER_X, y),
            colorspace=_COLOR_SPACE,
        )
        metallic = self._texture(
            builder,
            "pxrtexture::3.0",
            f"Metallic_{suffix}",
            maps.get("Metallic"),
            (_READER_X, y - _NODE_ROW),
            colorspace=_DATA_SPACE,
        )
        roughness = self._texture(
            builder,
            "pxrtexture::3.0",
            f"Roughness_{suffix}",
            maps.get("SpecularRoughness"),
            (_READER_X, y - 3 * _NODE_ROW),
            colorspace=_DATA_SPACE,
        )
        normal = self._texture(
            builder,
            "pxrnormalmap::3.0",
            f"Normal_{suffix}",
            maps.get("Normal"),
            (_PROCESS_X, y - 4 * _NODE_ROW),
            colorspace=_DATA_SPACE,
        )
        transparency = self._texture(
            builder,
            "pxrtexture::3.0",
            f"Transparency_{suffix}",
            maps.get("Transparency"),
            (_READER_X, y - 2 * _NODE_ROW),
            colorspace=_DATA_SPACE,
        )

        if base_color is not None or metallic is not None:
            self._insert_metallic_workflow(
                builder, layer, suffix, y, base_color, metallic
            )
        if roughness is not None:
            self._insert_roughness_remap(builder, layer, suffix, y, roughness)
        if normal is not None:
            layer.setNamedInput("bumpNormal", normal, "resultN")
        if transparency is not None:
            self._insert_glass(builder, layer, suffix, y, transparency)
        return layer

    def _insert_glass(
        self,
        builder: hou.Node,
        layer: hou.Node,
        suffix: str,
        y: float,
        transparency: hou.Node,
    ) -> hou.Node:
        """Glass where the mask is white, and no diffuse there.

        The mask drives both halves of the glass lobe: refraction alone loses
        the Fresnel reflection that makes it read as glass. PxrSurface does not
        trade diffuse against refraction, so clear areas would stay milky
        without the inverted gain. `thinGlass` stays off because the show
        models glass with thickness.
        """
        _parm(layer, "enableRR").set(True)
        layer.destroyUserData("__inputgroup_Glass")
        layer.setNamedInput("rrRefractionGain", transparency, "resultR")
        layer.setNamedInput("rrReflectionGain", transparency, "resultR")
        invert = self._create(
            builder,
            "pxrinvert::3.0",
            f"DiffuseMask_{suffix}",
            # Just under MetallicWorkflow, which shows more connectors.
            (_PROCESS_X, y - 1.68),
        )
        invert.setNamedInput("inputRGB", transparency, "resultRGB")
        layer.setNamedInput("diffuseGain", invert, "resultR")
        return invert

    def _insert_metallic_workflow(
        self,
        builder: hou.Node,
        layer: hou.Node,
        suffix: str,
        y: float,
        base_color: hou.Node | None,
        metallic: hou.Node | None,
    ) -> hou.Node:
        """Split base colour and metallic into PxrLayer's diffuse/specular inputs."""
        workflow = self._create(
            builder,
            "pxrmetallicworkflow::3.0",
            f"MetallicWorkflow_{suffix}",
            (_PROCESS_X, y),
        )
        if base_color is not None:
            workflow.setNamedInput("baseColor", base_color, "resultRGB")
        else:
            # PxrMetallicWorkflow.baseColor defaults to RenderMan's placeholder
            # blue, and we are about to drive the layer's diffuse from it. Hand
            # it the layer's own default so an unpublished base colour changes
            # nothing except the specular response.
            _parm_tuple(workflow, "baseColor").set(
                _parm_tuple(layer, "diffuseColor").evalAsFloats()
            )
        if metallic is not None:
            workflow.setNamedInput("metallic", metallic, "resultR")
        layer.setNamedInput("diffuseColor", workflow, "resultDiffuseRGB")
        layer.setNamedInput("specularFaceColor", workflow, "resultSpecularFaceRGB")
        layer.setNamedInput("specularEdgeColor", workflow, "resultSpecularEdgeRGB")
        return workflow

    def _insert_roughness_remap(
        self,
        builder: hou.Node,
        layer: hou.Node,
        suffix: str,
        y: float,
        roughness: hou.Node,
    ) -> hou.Node:
        remap = self._create(
            builder,
            "pxrremap::3.0",
            f"RoughnessRemap_{suffix}",
            (_PROCESS_X, y - 3 * _NODE_ROW),
        )
        remap.setNamedInput("inputRGB", roughness, "resultRGB")
        layer.setNamedInput("specularRoughness", remap, "resultR")
        return remap

    def _build_preview_shader(
        self, builder: hou.Node, material: textures.MaterialSpec, row_y: float
    ) -> hou.Node | None:
        if not material.preview_maps:
            return None

        suffix = _node_name(material.texture_set)
        surface = self._create(
            builder, "usdpreviewsurface", f"{suffix}_UsdPreviewSurface", (11.0, row_y)
        )
        uv_reader = self._create(
            builder,
            "usdprimvarreader",
            f"{suffix}_PreviewUv",
            (5.0, row_y - 0.83),
        )
        _parm(uv_reader, "signature").set("float2")
        _parm(uv_reader, "varname").set(_UV_PRIMVAR)

        for preview in _PREVIEW_INPUTS:
            path = material.preview_maps.get(preview.map_name)
            if path is None:
                continue
            texture = self._create(
                builder,
                "usduvtexture::2.0",
                f"{preview.map_name}_{suffix}_PreviewTex",
                (8.0, row_y + 1.23 - preview.row * _NODE_ROW),
            )
            _parm(texture, "file").set(path)
            _parm(texture, "sourceColorSpace").set(preview.source_colorspace)
            _parm_tuple(texture, "scale").set(preview.scale)
            _parm_tuple(texture, "bias").set(preview.bias)
            texture.setNamedInput("st", uv_reader, "result")
            surface.setNamedInput(
                preview.surface_input, texture, preview.texture_output
            )
        return surface

    def _texture(
        self,
        builder: hou.Node,
        node_type: str,
        name: str,
        path: str | None,
        position: tuple[float, float],
        *,
        colorspace: str,
    ) -> hou.Node | None:
        """None when the map was never published, so the caller wires nothing."""
        if path is None:
            return None
        node = self._create(builder, node_type, name, position)
        _parm(node, "filename").set(path)
        _parm(node, "filename_colorspace").set(colorspace)
        return node

    @staticmethod
    def _material_output(builder: hou.Node) -> hou.Node:
        """The output RenderMan's builder ships with.

        Substituting our own `collect` would build a material that looks right
        and carries an unknown output contract, so refuse before anything is
        wired instead.
        """
        output = builder.node(_BUILDER_OUTPUT)
        if output is None:
            raise hou.OperationFailed(
                f"{builder.path()} ({builder.type().name()}) has no "
                f"'{_BUILDER_OUTPUT}' child; this RenderMan build is unsupported"
            )
        return output

    def _create(
        self,
        parent: hou.Node,
        node_type: str,
        name: str,
        position: tuple[float, float],
    ) -> hou.Node:
        node = parent.createNode(node_type)
        node.setName(name, unique_name=True)
        node.setUserData(_GENERATED_KEY, _GENERATED_VALUE)
        node.setPosition(hou.Vector2(*position))
        detail = _CONNECTOR_DETAIL.get(node_type)
        if detail is not None:
            node.setGenericFlag(detail, True)
        return node


class PathFault(Enum):
    """Why a texture path will not survive the asset being moved or shared."""

    NOT_PORTABLE = auto()
    OUTSIDE_ASSET = auto()


@dataclass(frozen=True)
class TexturePathProblem:
    """One offending texture parm, as facts. Rendering lives in the message."""

    node_path: str
    parm_name: str
    authored: str
    resolved: str
    fault: PathFault


_ASSET_ROOT_VARIABLES = ("$HIP", "$JOB")
_PORTABLE_PREFIXES = ("$HIP/", "$JOB/", "${HIP}/", "${JOB}/")
_MAX_REPORTED = 10
_FAULT_TEXT = {
    PathFault.NOT_PORTABLE: "must start with $HIP or $JOB",
    PathFault.OUTSIDE_ASSET: "resolves outside this asset",
}


def texture_path_problems(matlib: hou.Node) -> list[TexturePathProblem]:
    """Texture parms under `matlib` that will not resolve if the asset moves."""
    roots = _asset_roots()
    problems: list[TexturePathProblem] = []
    for node in (matlib, *matlib.allSubChildren()):
        for parm in _file_reference_parms(node):
            authored = parm.unexpandedString().strip()
            resolved = str(parm.eval()).strip()
            fault = _path_fault(authored, resolved, roots)
            if fault is not None:
                problems.append(
                    TexturePathProblem(
                        node_path=matlib.relativePathTo(node),
                        parm_name=parm.name(),
                        authored=authored,
                        resolved=resolved,
                        fault=fault,
                    )
                )
    return problems


def matlib_path_error(matlib: hou.Node) -> str:
    """Artist-facing message for the HDA's error LOP; empty when every path is fine."""
    problems = texture_path_problems(matlib)
    if not problems:
        return ""

    lines = [
        f"{len(problems)} texture path(s) will break when this asset is moved or "
        "opened by someone else. Every texture must be read through $HIP or $JOB.",
        "",
    ]
    for problem in problems[:_MAX_REPORTED]:
        detail = problem.authored
        if problem.resolved and problem.resolved != problem.authored:
            detail = f"{problem.authored}  ->  {problem.resolved}"
        lines.append(
            f"    {problem.node_path} ({problem.parm_name}) "
            f"{_FAULT_TEXT[problem.fault]}:"
        )
        lines.append(f"        {detail}")
    if len(problems) > _MAX_REPORTED:
        lines.append(f"    ...and {len(problems) - _MAX_REPORTED} more.")
    lines.append("")
    lines.append(
        "Republish these textures into this asset's publish/tex folder, "
        "then press Rebuild Materials."
    )
    return "\n".join(lines)


def _path_fault(
    authored: str, resolved: str, roots: Sequence[Path]
) -> PathFault | None:
    """Portability and containment are separate faults; the authored one wins."""
    if not authored:
        # No texture assigned. That map was never published, which is a lookdev
        # gap rather than a path that breaks when the asset moves.
        return None
    if not authored.startswith(_PORTABLE_PREFIXES):
        return PathFault.NOT_PORTABLE
    if resolved and roots and not any(_is_within(resolved, root) for root in roots):
        return PathFault.OUTSIDE_ASSET
    return None


def _is_within(path: str, root: Path) -> bool:
    return Path(path).resolve().is_relative_to(root)


def _asset_roots() -> list[Path]:
    """Resolved $HIP and $JOB: the folders a published texture may live under."""
    expanded = (hou.hscriptStringExpression(name) for name in _ASSET_ROOT_VARIABLES)
    return [Path(value).resolve() for value in expanded if value.strip()]


def _file_reference_parms(node: hou.Node) -> list[hou.Parm]:
    return [parm for parm in update.file_parms(node) if not parm.isAtDefault()]


class MatlibErrorChecker:
    """Entry point for the HDA's `errors` LOP. Do not rename."""

    @staticmethod
    def CheckFilepathsRelative(matlib: hou.Node) -> int:  # noqa: N802
        return 1 if texture_path_problems(matlib) else 0


class MatlibManager:
    """Backs the SKD_MatLib HDA's parameters and its Rebuild Materials button."""

    def __init__(self, node: hou.LopNode) -> None:
        self._node = node

    def geo_variants(self) -> list[str]:
        asset = self._asset()
        return _variant_names(asset.geometry_variants if asset else None, "main")

    def mat_variants(self) -> list[str]:
        asset = self._asset()
        return _variant_names(asset.material_variants if asset else None, _NO_TEXTURES)

    def initialize_defaults(self) -> None:
        _set_string_parm(self._node, "geo_var", self.geo_variants()[0])
        _set_string_parm(self._node, "mat_var", self.mat_variants()[0])

    def rebuild(self) -> list[str]:
        """Bring the materials up to date; what it could not do, for the artist."""
        matlib = self._material_library()
        if matlib is None:
            return [self._no_matlib()]

        geo_variant, mat_variant = self.variants()
        # Component Material expects /ASSET/mtl/g_<geo>/v_<mat>/MAT_<texset>.
        _parm(matlib, "matpathprefix").set(
            variants.material_scope_path(mat_variant, geo_variant=geo_variant)
        )
        plan = MaterialGraphBuilder(matlib).update(
            self._materials(geo_variant, mat_variant),
            build_preview=self._build_preview(),
            **self._roots(geo_variant, mat_variant),
        )
        return [*plan.kept, *self._missing(plan)]

    def problems(self) -> list[str]:
        """Why the materials no longer match the published textures."""
        matlib = self._material_library()
        if matlib is None:
            return [self._no_matlib()]
        geo_variant, mat_variant = self.variants()
        plan = MaterialGraphBuilder(matlib).plan(
            self._materials(geo_variant, mat_variant),
            build_preview=self._build_preview(),
            **self._roots(geo_variant, mat_variant),
        )
        name = self._node.name()
        where = f"geo='{geo_variant}' mat='{mat_variant}'"
        unbuilt = ", ".join(material.texture_set for material in plan.build)
        changed = ", ".join(builder.name() for builder, _ in plan.rebuild)
        stale = ", ".join(builder.name() for builder in plan.remove)
        problems: list[str] = []
        if unbuilt and self._node.isGenericFlagSet(hou.nodeFlag.Bypass):
            problems.append(
                f"'{name}' is bypassed, so it publishes no materials, but {where} now "
                f"has textures for {unbuilt}. Un-bypass it and its Component "
                "Material, press Rebuild Materials on it, then publish again."
            )
        elif unbuilt:
            problems.append(
                f"'{name}' has no material for {unbuilt}, which the geometry "
                f"binds and {where} has textures for. Press Rebuild Materials on it, "
                "then publish again."
            )
        if changed:
            problems.append(
                f"'{name}' builds {changed} from textures that have changed since. "
                "Press Rebuild Materials on it, then publish again."
            )
        if stale:
            problems.append(
                f"'{name}' still builds {stale}, whose textures are no longer "
                f"published for {where} or which the geometry no longer binds. Press "
                "Rebuild Materials on it to drop them, then publish again."
            )
        if plan.retarget:
            problems.append(
                f"'{name}' has {len(plan.retarget)} texture path(s) that no longer "
                "point at the published textures. Press Rebuild Materials on it to "
                "repoint them, then publish again."
            )
        return [*problems, *plan.kept, *self._missing(plan)]

    def has_textures(self) -> bool:
        geo_variant, mat_variant = self.variants()
        hip_root = Path(hou.hscriptStringExpression("$HIP"))
        return mat_variant in variants.published_material_variants(
            hip_root, geo_variant
        )

    def has_hand_edits(self) -> bool:
        matlib = self._material_library()
        return matlib is not None and any(
            not update.is_untouched(child)
            for child in matlib.children()
            if child.userData(_GENERATED_KEY) == _GENERATED_VALUE
        )

    def _missing(self, plan: MaterialPlan) -> list[str]:
        if not plan.missing:
            return []
        shown = "\n    ".join(plan.missing[:_MAX_REPORTED])
        more = len(plan.missing) - _MAX_REPORTED
        return [
            f"'{self._node.name()}' reads {len(plan.missing)} texture(s) that are not "
            f"on disk:\n    {shown}"
            + (f"\n    ...and {more} more." if more > 0 else "")
            + "\nRepublish the textures from Substance Painter, or ask a TD."
        ]

    def _no_matlib(self) -> str:
        return (
            f"'{self._node.name()}' has no {_MATLIB_TYPE} node, so it builds nothing."
        )

    def _build_preview(self) -> bool:
        return _toggle(self._node, "build_usd_preview", default=True)

    def _roots(self, geo_variant: str, mat_variant: str) -> dict[str, list[Path]]:
        """Where this variant's textures live, and where any of the asset's do."""
        hip_root = Path(hou.hscriptStringExpression("$HIP"))
        source = variants.texture_source_dir(hip_root, geo_variant)
        own = hip_root / variants.TEX_SOURCE_DIR
        return {
            "own_roots": [source / mat_variant, own / geo_variant / mat_variant],
            "read_roots": [source, own],
        }

    def variants(self) -> tuple[str, str]:
        return (
            _string_parm(self._node, "geo_var", "main"),
            _string_parm(self._node, "mat_var", _NO_TEXTURES),
        )

    def _materials(
        self, geo_variant: str, mat_variant: str
    ) -> Sequence[textures.MaterialSpec]:
        """The materials this MatLib builds: published textures the geometry binds."""
        hip_root = Path(hou.hscriptStringExpression("$HIP"))
        tex_root = variants.texture_source_dir(hip_root, geo_variant) / mat_variant
        materials = textures.published_materials(tex_root, hip_root=hip_root)
        if not materials:
            log.warning("No materials to build from %s", tex_root)
        source = hip_root / variants.GEO_SOURCE_DIR / f"{geo_variant}.usd"
        return _bound_materials(materials, source)

    def _asset(self) -> Asset | None:
        try:
            connection = ShotGrid.connect(DB_Config)
            return connection.get_asset(name=str(hou.contextOption("ASSET")))
        except Exception:
            # Variant menus and defaults must still work when ShotGrid is
            # unreachable; the artist can type a variant name by hand.
            log.exception(
                "Could not resolve the ASSET context option for %s", self._node.path()
            )
            return None

    def _material_library(self) -> hou.Node | None:
        by_name = self._node.node(_MATLIB_NAME)
        if by_name is not None and by_name.type().name() == _MATLIB_TYPE:
            return by_name
        return next(
            (
                child
                for child in self._node.children()
                if child.type().name() == _MATLIB_TYPE
            ),
            None,
        )


def _bound_materials(
    materials: Sequence[textures.MaterialSpec], source: Path
) -> Sequence[textures.MaterialSpec]:
    """Only the texture sets `source` binds."""
    if not source.is_file():
        log.warning(
            "No geometry at %s yet, so every published material is built", source
        )
        return materials
    bound = textures.bound_texture_sets(source)
    skipped = [m.texture_set for m in materials if m.texture_set not in bound]
    if skipped:
        log.info("Not building %s: nothing in %s binds them", skipped, source)
    return [m for m in materials if m.texture_set in bound]


def _variant_names(declared: Iterable[str] | None, fallback: str) -> list[str]:
    named = sorted((name for name in (declared or ()) if name), key=str.casefold)
    return named or [fallback]


def _menu_entries(values: list[str]) -> list[str]:
    """Houdini menus want alternating token and label entries."""
    return [entry for value in values for entry in (value, value)]


def _string_parm(node: hou.LopNode, name: str, fallback: str) -> str:
    parm = node.parm(name)
    if parm is None:
        return fallback
    return parm.evalAsString().strip() or fallback


def _set_string_parm(node: hou.LopNode, name: str, value: str) -> None:
    parm = node.parm(name)
    if parm is not None:
        parm.set(value)


def _toggle(node: hou.LopNode, name: str, *, default: bool) -> bool:
    parm = node.parm(name)
    return default if parm is None else bool(parm.evalAsInt())


def matlib_on_created(node: hou.LopNode) -> None:
    MatlibManager(node).initialize_defaults()


def matlib_geo_variant_menu(node: hou.LopNode) -> list[str]:
    return _menu_entries(MatlibManager(node).geo_variants())


def matlib_mat_variant_menu(node: hou.LopNode) -> list[str]:
    return _menu_entries(MatlibManager(node).mat_variants())


def matlib_on_variant_changed(node: hou.LopNode) -> None:
    if _toggle(node, "auto_rebuild", default=False):
        matlib_rebuild(node)


def matlib_rebuild(node: hou.LopNode) -> None:
    """The Rebuild Materials button: says what it had to leave alone."""
    notes = matlib_update(node)
    if not notes:
        return
    message = "\n\n".join(notes)
    if hou.isUIAvailable():
        hou.ui.displayMessage(
            message, severity=hou.severityType.Warning, title="Rebuild Materials"
        )
    else:
        log.warning("%s: %s", node.path(), message)


def matlib_update(node: hou.LopNode) -> list[str]:
    return MatlibManager(node).rebuild()


def matlib_problems(node: hou.LopNode) -> list[str]:
    return MatlibManager(node).problems()


def matlib_has_textures(node: hou.LopNode) -> bool:
    return MatlibManager(node).has_textures()


def matlib_has_hand_edits(node: hou.LopNode) -> bool:
    """Whether any generated material inside was changed since it was built."""
    return MatlibManager(node).has_hand_edits()


def matlib_variants(node: hou.LopNode) -> tuple[str, str]:
    """The (geometry, material) variant a MatLib builds."""
    return MatlibManager(node).variants()
