from __future__ import annotations

import json
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from pipe.core.asset.paths import PAINTER_OFFSET_FILENAME

_TRANSLATE_KEY = "translate"


def shape_for_painter(model: Sdf.Layer, *, offset: Gf.Vec3d | None) -> Sdf.Layer:
    """A copy of `model` with only material subsets, translated by `offset` if given."""
    layer = Sdf.Layer.CreateAnonymous(Path(model.identifier).name)
    layer.TransferContent(model)
    strip_component_subsets(layer)
    if offset is not None:
        place_for_painter(layer, offset)
    return layer


def read_painter_offset(painter_dir: Path) -> Gf.Vec3d | None:
    """The translation `painter_dir/painter_offset.json` declares, or None without one.

    The file is `{"translate": [x, y, z]}` in the mesh's own units (centimetres):
    the project's painted bounding-box centre minus the model's current one.
    """
    path = painter_dir / PAINTER_OFFSET_FILENAME
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    x, y, z = data[_TRANSLATE_KEY]
    return Gf.Vec3d(float(x), float(y), float(z))


def place_for_painter(layer: Sdf.Layer, offset: Gf.Vec3d) -> None:
    """Translate the layer's default prim by `offset`."""
    stage = Usd.Stage.Open(layer)
    root = UsdGeom.Xformable(stage.GetDefaultPrim())
    ops = root.GetOrderedXformOps()
    if ops:
        raise ValueError(
            f"{stage.GetDefaultPrim().GetPath()} already has xformOps "
            f"({[op.GetOpName() for op in ops]}); the Painter offset expects a bare root."
        )
    root.AddTranslateOp().Set(offset)


def strip_component_subsets(layer: Sdf.Layer) -> int:
    """Remove every GeomSubset that is not a material partition; returns how many."""
    stage = Usd.Stage.Open(layer)
    doomed = [
        prim.GetPath()
        for prim in stage.Traverse()
        if prim.IsA("GeomSubset")
        and UsdGeom.Subset(prim).GetFamilyNameAttr().Get()
        != UsdShade.Tokens.materialBind
    ]
    for path in doomed:
        stage.RemovePrim(path)
    return len(doomed)
