"""Bring a MatLib's generated materials up to date without losing hand edits.

Each generated `MAT_*` builder remembers a fingerprint of the network it was
built as. One that still matches is nobody's work, so it is rebuilt whenever
its textures change. One an artist changed (a shader added inside, a parm
tweaked, a wire moved) is never replaced: its texture paths are repointed and
anything else it cannot take is reported. Builders made before fingerprints
existed count as edited.

Reading only; `MaterialGraphBuilder.update` in `main` applies the plan.
"""

from __future__ import annotations

import glob
import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import hou

from . import textures

FINGERPRINT_KEY = "skd_matlib_fingerprint"
SPEC_KEY = "skd_matlib_spec"

_NODE_UNSAFE_RE = re.compile(r"[^A-Za-z0-9_]+")
_UDIM_GLOB = "1[0-9][0-9][0-9]"


@dataclass
class MaterialPlan:
    """What updating a MatLib does, worked out before anything changes."""

    build: list[textures.MaterialSpec] = field(default_factory=list)
    rebuild: list[tuple[hou.Node, textures.MaterialSpec]] = field(default_factory=list)
    remove: list[hou.Node] = field(default_factory=list)
    retarget: list[tuple[hou.Parm, str]] = field(default_factory=list)
    # Why an edited material keeps a graph that no longer matches its textures.
    kept: list[str] = field(default_factory=list)
    # Texture parms reading published textures that are not on disk.
    missing: list[str] = field(default_factory=list)


def node_name(value: str) -> str:
    """Node-safe form of a texture-set or layer name, preserving case."""
    cleaned = re.sub(r"_+", "_", _NODE_UNSAFE_RE.sub("_", value.strip())).strip("_")
    if not cleaned:
        return "material"
    return f"m_{cleaned}" if cleaned[0].isdigit() else cleaned


def material_node_name(texture_set: str) -> str:
    return f"MAT_{node_name(texture_set)}"


def plan(
    matlib: hou.Node,
    builders: Mapping[str, hou.Node],
    materials: Sequence[textures.MaterialSpec],
    *,
    build_preview: bool,
    own_roots: Sequence[Path],
    read_roots: Sequence[Path],
) -> MaterialPlan:
    """How to bring `builders` (the generated ones, by name) in line with `materials`.

    `own_roots` hold this MatLib's own variant's textures; `read_roots` hold
    every texture this asset publishes, so a path under one may be repointed.
    """
    result = MaterialPlan()
    wanted = textures.wanted_paths(materials)
    leftover = dict(builders)
    for material in materials:
        builder = leftover.pop(material_node_name(material.texture_set), None)
        if builder is None:
            result.build.append(material)
        elif not is_untouched(builder):
            result.kept.extend(
                _unwired(
                    builder, material, wanted, build_preview, own_roots, read_roots
                )
            )
        elif builder.userData(SPEC_KEY) != spec_signature(material, build_preview):
            result.rebuild.append((builder, material))
    for builder in leftover.values():
        if is_untouched(builder):
            result.remove.append(builder)
        else:
            result.kept.append(
                f"'{builder.name()}' was edited by hand, so it stays although its "
                "texture set is no longer published or bound. Delete it if it is "
                "no longer needed."
            )

    replaced = [b for b, _ in result.rebuild] + result.remove
    for parm in _texture_parms(matlib, skip=replaced, roots=read_roots):
        path = wanted.get(textures.texture_key(parm.unexpandedString()))
        if path is not None and path != parm.unexpandedString():
            result.retarget.append((parm, path))
        if not _finds_file(hou.text.expandString(path or parm.unexpandedString())):
            result.missing.append(f"{parm.path()} = {parm.unexpandedString()}")
    return result


def stamp(
    builder: hou.Node, material: textures.MaterialSpec, *, build_preview: bool
) -> None:
    """Remember what the builder was built as and from."""
    builder.setUserData(FINGERPRINT_KEY, fingerprint(builder))
    builder.setUserData(SPEC_KEY, spec_signature(material, build_preview))


def is_untouched(builder: hou.Node) -> bool:
    return builder.userData(FINGERPRINT_KEY) == fingerprint(builder)


def fingerprint(builder: hou.Node) -> str:
    """The network inside `builder`, minus layout and texture paths.

    Paths are left out so repointing them never makes a material look edited.
    """
    digest = hashlib.sha1()
    for node in sorted(builder.allSubChildren(), key=lambda n: n.path()):
        digest.update(repr(_describe(builder, node)).encode())
    return digest.hexdigest()


def spec_signature(material: textures.MaterialSpec, build_preview: bool) -> str:
    return hashlib.sha1(repr((material, build_preview)).encode()).hexdigest()


def file_parms(node: hou.Node) -> list[hou.Parm]:
    """Every parm Houdini types as a file reference."""
    return [
        parm
        for parm in node.parms()
        if isinstance(template := parm.parmTemplate(), hou.StringParmTemplate)
        and template.stringType() == hou.stringParmType.FileReference
    ]


def _describe(builder: hou.Node, node: hou.Node) -> tuple:
    inputs = [
        (c.inputIndex(), builder.relativePathTo(c.inputNode()), c.outputIndex())
        for c in node.inputConnections()
    ]
    paths = set(file_parms(node))
    parms = sorted(
        (parm.name(), repr(parm.eval()))
        for parm in node.parms()
        if parm not in paths and not parm.isAtDefault()
    )
    bypassed = isinstance(node, hou.VopNode) and node.isBypassed()
    return builder.relativePathTo(node), node.type().name(), bypassed, inputs, parms


def _unwired(
    builder: hou.Node,
    material: textures.MaterialSpec,
    wanted: Mapping[textures.TextureKey, str],
    build_preview: bool,
    own_roots: Sequence[Path],
    read_roots: Sequence[Path],
) -> list[str]:
    """What an edited material reads that is no longer published, and the reverse.

    A path elsewhere in the asset counts only when it is about to be repointed:
    loose folders beside the variants (`glass maps`) hold files named like maps.
    """
    keys = {
        key
        for key in wanted
        if key.tex_set == material.texture_set
        and (build_preview or key.layer is not None)
    }
    wired = {
        key
        for parm in _texture_parms(builder, skip=(), roots=read_roots)
        if (key := textures.texture_key(parm.unexpandedString())) is not None
        and key.tex_set == material.texture_set
        and (key in wanted or _reads_under(parm, own_roots))
    }
    name = builder.name()
    notes = []
    if unread := keys - wired:
        notes.append(
            f"'{name}' was edited by hand, so it was not rebuilt and does not read "
            f"{_describe_keys(unread)}. Wire them in by hand, or delete '{name}' and "
            "press Rebuild Materials to rebuild it from the textures, losing the edits."
        )
    if gone := wired - keys:
        notes.append(
            f"'{name}' was edited by hand, so it was not rebuilt and still reads "
            f"{_describe_keys(gone)}, which are no longer published. Remove them by "
            f"hand, or delete '{name}' and press Rebuild Materials, losing the edits."
        )
    return notes


def _describe_keys(keys: Iterable[textures.TextureKey]) -> str:
    return ", ".join(
        sorted(
            f"preview {key.map_name}"
            if key.layer is None
            else f"{key.map_name} of layer '{key.layer}'"
            for key in keys
        )
    )


def _texture_parms(
    root: hou.Node, *, skip: Sequence[hou.Node], roots: Sequence[Path]
) -> list[hou.Parm]:
    """Plain-text file parms under `root` reading a file under one of `roots`."""
    skipped = tuple(f"{node.path()}/" for node in skip)
    parms = []
    for node in root.allSubChildren():
        if f"{node.path()}/".startswith(skipped):
            continue
        for parm in file_parms(node):
            try:
                value = parm.unexpandedString()
            except hou.OperationFailed:  # an expression or keyframes, not a path
                continue
            if value and _reads_under(parm, roots):
                parms.append(parm)
    return parms


def _reads_under(parm: hou.Parm, roots: Sequence[Path]) -> bool:
    resolved = Path(str(parm.eval())).resolve()
    return any(resolved.is_relative_to(root.resolve()) for root in roots)


def _finds_file(path: str) -> bool:
    return bool(glob.glob(glob.escape(path).replace("<UDIM>", _UDIM_GLOB)))
