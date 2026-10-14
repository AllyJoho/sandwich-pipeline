"""RenderMan's `denoise_batch`: the product it reads and its config.

The beauty product, which carries the denoise passes, renders into `tmp/`. Each
frame's command denoises it and writes the finished frame into the product's folder.

The config names layers as RenderMan writes them, not as the vars are named:
with the product's asrgba on, the alpha var `a` is written as `A`, and the
beauty `Ci` as `R, G, B`.

The denoiser filters each pass with a filter fitted to that pass's input, so an
LPE is denoised in the pass its first bounce matches: diffuse or specular. An
LPE of both, such as SKD Render Layer's per-light AOVs, is rendered as two
halves that are denoised apart and summed back into its own layer.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pxr import Sdf, Usd, UsdRender

from pipe.core.render import DENOISE_CONFIG
from pipe.dcc.houdini.tractor import SendRefused, paths

ASRGBA = "driver:parameters:openexr:asrgba"
SOURCE_NAME = "sourceName"
SOURCE_TYPE = "sourceType"
LPE = "lpe"

# RenderMan's symmetric denoisers as <topology, parameters>, paired as in
# $RMANTREE/lib/python3.11/site-packages/denoise/denoiser/core.py
SINGLE_FRAME = (
    "${RMANTREE}/lib/denoise/full_w1_5s_sym_gen2.topo",
    "${RMANTREE}/lib/denoise/20973-renderman.param",
)
CROSS_FRAME = (
    "${RMANTREE}/lib/denoise/full_w7_4sv2_sym_gen2.topo",
    "${RMANTREE}/lib/denoise/20970-renderman.param",
)

# The vars Denoise adds that its config reads. The beauty is read as R, G, B,
# so it isn't among them.
PASSES = frozenset(
    {"a", "mse", "sampleCount", "albedo", "albedo_mse", "normal", "normal_mse"}
    | {"diffuse", "diffuse_mse", "specular", "specular_mse"}
)

DIFFUSE = "diffuse"
SPECULAR = "specular"
# A combined LPE's halves are named after it with these suffixes, and both are
# written back into its layer.
HALVES = {DIFFUSE: "_dnDiff", SPECULAR: "_dnSpec"}

# The first bounce after the camera: D or S, alone or as an event such as
# <.D'group'> or <RS>, optionally opening a group.
_FIRST_DIFFUSE = re.compile(r"\(?(D|<[.RT]D)")
_FIRST_SPECULAR = re.compile(r"\(?(S|<[.RT]S)")

# The combined LPEs SKD Render Layer writes, each as <diffuse half, specular
# half>. Per light: everything from the light group's lights.
_PER_LIGHT = re.compile(r"C\.\*(<L\.'[^']+'>)")
# Per geo: everything whose first bounce is on the group, or its emission.
_PER_GEO = re.compile(r"C\(<\.\.('[^']+')>\.\*\[LO\]\)\|<O\.\1>")


def _lpe(var: Usd.Prim) -> str | None:
    """The var's light path expression, without its flags, if it is an LPE."""
    if var.GetAttribute(SOURCE_TYPE).Get() != LPE:
        return None
    source = var.GetAttribute(SOURCE_NAME).Get() or ""
    return source.removeprefix("lpe:").rpartition(";")[2]


def lpe_kind(var: Usd.Prim) -> str | None:
    """The pass an LPE var is denoised in, by its first bounce, if one fits."""
    expression = _lpe(var)
    if not expression or not expression.startswith("C"):
        return None
    if _FIRST_DIFFUSE.match(expression, 1):
        return DIFFUSE
    if _FIRST_SPECULAR.match(expression, 1):
        return SPECULAR
    return None


def halves(var: Usd.Prim) -> tuple[str, str] | None:
    """A combined LPE var's diffuse and specular halves, if it is one Send can split."""
    expression = _lpe(var)
    if not expression:
        return None
    if found := _PER_LIGHT.fullmatch(expression):
        light = found[1]
        # As Denoise's own diffuse, the lights seen directly are in the diffuse half.
        return f"C(D.*{light})|{light}", f"CS.*{light}"
    if found := _PER_GEO.fullmatch(expression):
        group = found[1]
        return f"C(<.D{group}>.*[LO])|<O.{group}>", f"C<.S{group}>.*[LO]"
    return None


def config(vars: list[tuple[str, str | None]], frames: int) -> dict:
    """denoise_batch's config for a product with these vars, as <name, LPE kind>.

    The finished beauty is denoised diffuse plus denoised specular, which
    denoise_batch sums into R, G, B. The halves of a split LPE are summed the
    same way into the LPE's own layer.

    If more than one frame, uses cross-frame denoise. Otherwise, uses
    single-frame denoise.
    """

    topology, parameters = CROSS_FRAME if frames > 1 else SINGLE_FRAME

    outputs: dict[str, list[dict]] = {
        "albedo": [],
        "alpha": [_output("A", "A")],
        DIFFUSE: [_output("diffuse", "RGB")],
        SPECULAR: [_output("specular", "RGB")],
        "copy": [],
    }
    for var, lpe in vars:
        # The denoiser's own inputs, the alpha and the beauty are written above.
        if var.endswith(("mse", "_var")) or var in ("a", "Ci", "sampleCount"):
            continue
        kind = lpe or var.partition("_")[0]
        if lpe and var.endswith(HALVES[lpe]):
            outputs[lpe].append(_output(var, var.removesuffix(HALVES[lpe])))
        elif var.startswith(("backward", "forward")) or var in ("__depth", "zfiltered"):
            outputs["alpha"].append(_output(var, var))
        elif kind in ("albedo", DIFFUSE, SPECULAR):
            outputs[kind].append(_output(var, var))
        else:
            outputs["copy"].append(_output(var, var))

    passes = [
        _pass("albedo", "albedo", "albedo_mse", outputs["albedo"]),
        _pass("alpha", "A", "mse", outputs["alpha"]),
        _pass(DIFFUSE, "diffuse", "diffuse_mse", outputs[DIFFUSE]),
        _pass(SPECULAR, "specular", "specular_mse", outputs[SPECULAR]),
    ]
    if outputs["copy"]:
        # No layer is named Ci, so denoise_batch skips this pass's filter and
        # copies its outputs unchanged.
        passes.append(_pass("copy", "Ci", "mse", outputs["copy"]))

    return {
        "settings": {
            "albedo": _read("albedo"),
            "albedo_variance": _read("albedo_mse"),
            "normal": _read("normal"),
            "normal_variance": _read("normal_mse"),
            "sample_count": _read("sampleCount"),
            "frame-include": "${FrameInclude}",
            "frame-exclude": "${FrameExclude}",
            "topology": topology,
            "parameters": parameters,
            "asymmetry": 0.0,
            "overwrite": "OverwriteChannels",
            "progress": True,
            "tiles": [1, 1],
        },
        "passes": passes,
    }


def _read(layer: str) -> dict:
    return {"layer": layer, "filename": "${InputFile}"}


def _output(read: str, write: str) -> dict:
    return {"read": _read(read), "write": {"layer": write, "filename": "${OutputFile}"}}


def _pass(name: str, input: str, variance: str, outputs: list[dict]) -> dict:
    return {
        "name": name,
        "input": _read(input),
        "input_variance": _read(variance),
        "outputs": outputs,
    }


def write_config(folder: Path, config: dict) -> None:
    (folder / DENOISE_CONFIG).write_text(json.dumps(config))


def var_names(product: Usd.Prim) -> list[str]:
    return [paths.aov_name(var) for var in paths.render_vars(product)]


def var_kinds(product: Usd.Prim) -> list[tuple[str, str | None]]:
    """Each var's AOV name and LPE kind. Denoise's own vars go by their names:
    its albedo is an LPE through specular bounces, but isn't specular."""
    found = []
    for var in paths.render_vars(product):
        name = paths.aov_name(var)
        found.append((name, None if name in PASSES else lpe_kind(var)))
    return found


def add_passes(stage: Usd.Stage, configured: Usd.Stage, settings: str) -> None:
    """Append the render vars Denoise defined to the beauty product it denoises,
    and split its combined LPEs into halves the denoiser can filter.

    `configured` is the stage before Denoise, so its vars are the artist's. Other
    products, such as Cryptomatte or another settings' utility passes, keep
    their own vars.
    """
    rendered = paths.rendered_settings(stage, settings)
    product = paths.beauty_product(rendered, paths.products(rendered))
    rel = UsdRender.Product(product).GetOrderedVarsRel()
    for var in paths.render_vars(product):
        if split := halves(var):
            _split(rel, var, split)
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/Render")):
        if prim.IsA(paths.VAR) and not configured.GetPrimAtPath(prim.GetPath()):
            rel.AddTarget(prim.GetPath())
    # The config reads the beauty and alpha as RGB and A. Without asrgba they are
    # written as Ci and a: the alpha is lost, and the copy pass filters the
    # utility passes instead of copying them. Render layers saved before it
    # defaulted on keep it off.
    product.CreateAttribute(ASRGBA, Sdf.ValueTypeNames.Int).Set(1)


def _split(rel: Usd.Relationship, var: Usd.Prim, split: tuple[str, str]) -> None:
    """Render `var` as its two halves in its place. Its layer is their sum."""
    name = paths.aov_name(var)
    for (kind, suffix), expression in zip(HALVES.items(), split):
        path = var.GetPath().GetParentPath().AppendChild(var.GetName() + suffix)
        if var.GetStage().GetPrimAtPath(path):
            raise SendRefused(
                f"Denoise splits {var.GetPath()} into diffuse and specular, but "
                f"{path} already exists. Rename that render var."
            )
        half = UsdRender.Var.Define(var.GetStage(), path).GetPrim()
        # Keeps the var's format, data type and clear value.
        for attr in var.GetAuthoredAttributes():
            copy = half.CreateAttribute(attr.GetName(), attr.GetTypeName())
            copy.Set(attr.Get())
        half.GetAttribute(SOURCE_NAME).Set(expression)
        for attr in paths.AOV_NAMES:
            if half.HasAttribute(attr):
                half.GetAttribute(attr).Set(name + suffix)
        half.CreateAttribute(paths.AOV_NAMES[0], Sdf.ValueTypeNames.String).Set(
            name + suffix
        )
        rel.AddTarget(path)
    rel.RemoveTarget(var.GetPath())


def product(settings: UsdRender.Settings, products: list[Usd.Prim]) -> Usd.Prim:
    """The beauty product, which Denoise gave its passes."""
    found = paths.beauty_product(settings, products)
    if not PASSES <= set(var_names(found)):
        raise SendRefused(
            f"{found.GetPath()} lacks the passes Denoise adds, so it can't be "
            "denoised. Check that Denoise follows Configure and isn't bypassed."
        )
    return found
