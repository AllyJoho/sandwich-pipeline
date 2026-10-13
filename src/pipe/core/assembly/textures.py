"""A child's textures: links to the assembly's texture files in the UDIM tiles it uses."""

from __future__ import annotations

import math
import os
import re
import shutil
from pathlib import Path

from pxr import Usd, UsdGeom

from pipe.core.asset.paths import DEFAULT_GEOMETRY_VARIANT, AssetPaths
from pipe.core.assembly.model import AssemblyError

_UV_PRIMVAR = "st"
_UDIM_FILE = re.compile(r"\.(?P<tile>1\d{3})\.[^.]+$")
_PENDING_PREFIX = ".linking-"
_RETIRED_PREFIX = ".retired-"


def texture_link_problem(child_root: Path, variant: str) -> str | None:
    """Why the child's textures for `variant` cannot be linked, or None if they can."""
    folder = AssetPaths(child_root).publish_textures_variant_dir(variant)
    if not folder.is_symlink() and folder.exists() and not _is_link_folder(folder):
        return (
            f"'{child_root.name}' already has its own textures at {folder}, but a "
            "split piece shares its assembly's. Split under a different name, or "
            "delete that folder if it is a leftover."
        )
    return None


def link_used_textures(child_root: Path, variant: str, assembly_root: Path) -> None:
    """Rebuild the child's `publish/tex/<variant>` from the assembly's textures.

    Each file in a UDIM tile the variant's geometry uses is linked; a file
    named without a tile (`mat.json`) is always linked. Replaces the folder
    whole, so a reader sees the old links or the new ones, never a mix.
    """
    problem = texture_link_problem(child_root, variant)
    if problem is not None:
        raise AssemblyError(problem)
    paths = AssetPaths(child_root)
    folder = paths.publish_textures_variant_dir(variant)
    textures = AssetPaths(assembly_root).publish_textures_variant_dir(
        DEFAULT_GEOMETRY_VARIANT
    )
    tiles = used_tiles(paths.publish_source_variant_usd(variant))

    pending = folder.with_name(_PENDING_PREFIX + folder.name)
    _remove_link_folder(pending)
    pending.mkdir(parents=True)
    if textures.is_dir():
        _link_tiles(textures, pending, tiles, link_root=folder)
    _swap_in(pending, folder)


def used_tiles(source_layer: Path) -> set[int]:
    """The UDIM tiles the faces of every mesh in `source_layer` sit in, by UV centroid."""
    stage = Usd.Stage.Open(str(source_layer))
    tiles: set[int] = set()
    for prim in stage.Traverse():
        mesh = UsdGeom.Mesh(prim)
        if mesh:
            tiles |= _mesh_tiles(mesh)
    return tiles


def _mesh_tiles(mesh: UsdGeom.Mesh) -> set[int]:
    primvar = UsdGeom.PrimvarsAPI(mesh).GetPrimvar(_UV_PRIMVAR)
    uvs = primvar.ComputeFlattened() if primvar else None
    counts = mesh.GetFaceVertexCountsAttr().Get()
    if not uvs or not counts:
        return set()
    interpolation = primvar.GetInterpolation()
    if interpolation == UsdGeom.Tokens.faceVarying:
        corners = range(len(uvs))
    elif interpolation in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying):
        corners = mesh.GetFaceVertexIndicesAttr().Get()
    else:
        return set()

    tiles: set[int] = set()
    start = 0
    for count in counts:
        face = [uvs[corners[i]] for i in range(start, start + count)]
        start += count
        u = sum(uv[0] for uv in face) / count
        v = sum(uv[1] for uv in face) / count
        tiles.add(1001 + math.floor(u) + 10 * math.floor(v))
    return tiles


def _link_tiles(
    source: Path, pending: Path, tiles: set[int], *, link_root: Path
) -> None:
    """Mirror `source` under `pending` as relative links, keeping only `tiles`.

    Links are relative to where they will live, `link_root`, so the pair
    survives a move together and either mount of /job.
    """
    for directory, _, files in os.walk(source):
        relative = Path(directory).relative_to(source)
        (pending / relative).mkdir(exist_ok=True)
        for name in files:
            match = _UDIM_FILE.search(name)
            if match is not None and int(match.group("tile")) not in tiles:
                continue
            target = Path(directory) / name
            os.symlink(
                os.path.relpath(target, link_root / relative), pending / relative / name
            )


def _swap_in(pending: Path, folder: Path) -> None:
    if folder.is_symlink():
        folder.unlink()
    elif folder.exists():
        retired = folder.with_name(_RETIRED_PREFIX + folder.name)
        _remove_link_folder(retired)
        folder.rename(retired)
        pending.rename(folder)
        _remove_link_folder(retired)
        return
    pending.rename(folder)


def _remove_link_folder(folder: Path) -> None:
    """Delete a folder this module wrote; its files are links, so no texture goes with it."""
    if folder.is_symlink():
        folder.unlink()
    elif folder.exists():
        if not _is_link_folder(folder):
            raise AssemblyError(
                f"{folder} holds real files, so it was left alone. Move it aside "
                "and try again."
            )
        shutil.rmtree(folder)


def _is_link_folder(folder: Path) -> bool:
    """Whether every file under `folder` is a link, as a folder this module wrote is."""
    for directory, _, files in os.walk(folder):
        if any(not os.path.islink(os.path.join(directory, name)) for name in files):
            return False
    return True
