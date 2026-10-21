from __future__ import annotations

from typing import TYPE_CHECKING

from env_sg import DB_Config
from nxt.tokens import register_token
from pipe.core.asset import AssetPaths, paths_for_asset
from pipe.core.shotgrid import ShotGrid

if TYPE_CHECKING:
    from nxt.nxt_layer import SpecLayer
    from nxt.nxt_node import Node
    from nxt.stage import Stage

PREFIX = "model::"

_conn = ShotGrid.connect(DB_Config)
_cached_asset_paths: dict[str, AssetPaths | None] = {}


def _get_asset_paths_from_name(name: str) -> AssetPaths | None:
    if name in _cached_asset_paths:
        return _cached_asset_paths[name]
    else:
        try:
            asset = _conn.get_asset(name=name)
            paths = paths_for_asset(asset)
            _cached_asset_paths[name] = paths
            return paths
        except Exception:
            _cached_asset_paths[name] = None
            return None


def detect_token_type(value):
    return value.startswith(PREFIX)


def resolve_token(
    stage: Stage, node: Node, value: str, layer: SpecLayer, **kwargs
) -> str:
    value: str = stage.resolve(node, value, layer, **kwargs)
    parts = value.split(":", 1)
    name = parts[0]
    if len(parts) > 1:
        variant = parts[1]
    else:
        variant = None
    asset_paths = _get_asset_paths_from_name(name)
    if asset_paths is None:
        return "<INVALID_ASSET>"

    if variant is None:
        return asset_paths.publish_source_model_usd.as_posix()

    return asset_paths.publish_source_variant_usd(variant).as_posix()


register_token(PREFIX, detect_token_type, resolve_token)
