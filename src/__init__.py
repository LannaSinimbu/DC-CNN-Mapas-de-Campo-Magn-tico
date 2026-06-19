"""Pacote DC-CNN: continuação para baixo + CNN para mapas de campo Bz."""

from .physics import (
    get_k_grid,
    estimate_z0,
    downward_continuation,
    deconvolve_footprint,
)
from .data import (
    load_map,
    load_maps,
    force_shape,
    build_augmented_dataset,
    build_pairs,
    extract_patches,
    BzPatchDataset,
)
from .model import UNet2D, DoubleConv, crop_to_mult8
from .config import load_config, resolve_paths

__all__ = [
    "get_k_grid",
    "estimate_z0",
    "downward_continuation",
    "deconvolve_footprint",
    "load_map",
    "load_maps",
    "force_shape",
    "build_augmented_dataset",
    "build_pairs",
    "extract_patches",
    "BzPatchDataset",
    "UNet2D",
    "DoubleConv",
    "crop_to_mult8",
    "load_config",
    "resolve_paths",
]
