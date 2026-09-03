"""
HW1 data-quality triplestore CLI — student distribution.

STUDENT IMPLEMENTATION SURFACE:
    Students implement ONLY the 8 qualification-factor measurers behind the
    factor menu and their corresponding drop mask extractors:

        1. frame_clip_hi_fraction
        2. frame_clip_lo_fraction
        3. frame_high_frequency_depth_residual (+ _mask)
        4. frame_flying_pixel_ratio (+ _mask)
        5. frame_valid_tile_coverage (+ _mask)
        6. pair_identity_median_depth_change (+ _mask)
        7. pair_joint_valid_depth_ratio (+ _mask)
        8. pair_prior_warp_depth_residual (+ _mask)

    All RDF graph operations, CLI subcommands, validation schemas, and tamper-seals
    ship fully implemented and must not be modified.
"""

import argparse
import glob
import hashlib
import os
import re
import sys

import numpy as np
from PIL import Image

from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef, XSD

# =============================================================================
# Namespaces
# =============================================================================
NS = "http://taica.course/hw1/ontology#"
HW1 = Namespace(NS)
SCHEMA = Namespace("https://schema.org/")
QUDT = Namespace("http://qudt.org/schema/qudt/")
UNIT = Namespace("http://qudt.org/vocab/unit/")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
PROV = Namespace("http://www.w3.org/ns/prov#")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ONTOLOGY_TTL = os.path.join(_HERE, "ontology", "hw1.ttl")
_EXPERIMENT_DIR = os.path.join(_HERE, "experiments")
_DEPTH_SCALE = 1000.0

_IMMERKAER_M = np.array([[1.0, -2.0, 1.0],
                         [-2.0, 4.0, -2.0],
                         [1.0, -2.0, 1.0]])
_IMMERKAER_NORM = 6.0
_MAD_TO_SIGMA = 0.6745


def _value_channel(rgb_path):
    """V = max(R,G,B) per pixel, float64 HxW in [0,255]. HSV Value, no weighting."""
    arr = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float64)
    return arr.max(axis=2)


def frame_mean_value(rgb_path):
    """The BASELINE: mean of V = max(R,G,B) over one RGB frame. Not a quality factor."""
    return float(_value_channel(rgb_path).mean())


# =============================================================================
# TODO: Student Measurer 1 & 2 (RGB Quality Factors)
# =============================================================================
def frame_clip_hi_fraction(rgb_path, tau_hi):
    """RGB quality factor 1 (HighlightClipping) — blown-out pixel fraction.

    CONTRACT:
        In:     rgb_path — an RGB PNG path.
                tau_hi — highlight threshold on the 0-255 scale (float).
        Out:    float in [0, 1]: |{ V >= tau_hi }| / N, where V = max(R,G,B).
        Grade:  LowerIsBetter (Pass iff <= maxClipHiFraction).
    """
    # TODO: Student implementation
    # 1. Compute V = max(R, G, B) across pixels using _value_channel(rgb_path)
    # 2. Count fraction of pixels where V >= tau_hi
    raise NotImplementedError("TODO: Implement frame_clip_hi_fraction")


def frame_clip_lo_fraction(rgb_path, tau_lo):
    """RGB quality factor 2 (ShadowClipping) — crushed pixel fraction.

    CONTRACT:
        In:     rgb_path — an RGB PNG path.
                tau_lo — shadow threshold on the 0-255 scale (float).
        Out:    float in [0, 1]: |{ V <= tau_lo }| / N, where V = max(R,G,B).
        Grade:  LowerIsBetter (Pass iff <= maxClipLoFraction).
    """
    # TODO: Student implementation
    # 1. Compute V = max(R, G, B) across pixels using _value_channel(rgb_path)
    # 2. Count fraction of pixels where V <= tau_lo (all channels dark)
    raise NotImplementedError("TODO: Implement frame_clip_lo_fraction")


# =============================================================================
# Depth helper utilities
# =============================================================================
def _consumer_depth_metres_valid(depth_path):
    """Read one uint16-mm depth raster under consumer validity rule (depth > 0)."""
    raw = np.asarray(Image.open(depth_path))
    if raw.ndim != 2:
        raise ValueError(f"depth raster must be two-dimensional, got {raw.shape}")
    metres = raw.astype(np.float64) / _DEPTH_SCALE
    return metres, raw != 0


def _flag_mask(flagged):
    """Boolean drop flags -> on-disk mask convention (uint8 0/255)."""
    return np.where(flagged, np.uint8(255), np.uint8(0))


def _camera_intrinsics(intrinsics, shape):
    """Accept the capture dict, a (width, height, hfov) tuple, or a 3x3 K."""
    h, w = shape
    arr = np.asarray(intrinsics) if not isinstance(intrinsics, dict) else None
    if arr is not None and arr.shape == (3, 3):
        return float(arr[0, 0]), float(arr[1, 1]), float(arr[0, 2]), float(arr[1, 2])
    if isinstance(intrinsics, dict):
        width = int(intrinsics["width"])
        height = int(intrinsics["height"])
        hfov = float(intrinsics["hfov"])
    else:
        width, height, hfov = intrinsics
        width, height, hfov = int(width), int(height), float(hfov)
    if (height, width) != (h, w):
        raise ValueError(f"intrinsics ({width}x{height}) do not match raster ({w}x{h})")
    fx = fy = (width / 2.0) / np.tan(np.radians(hfov / 2.0))
    return fx, fy, width / 2.0, height / 2.0


# =============================================================================
# TODO: Student Measurers 3, 4, 5 (Depth Frame Quality Factors)
# =============================================================================
def _high_frequency_depth_residual(depth_path, residual_mask_k):
    """Compute high-frequency depth residual scalar and 0/255 drop mask.

    CONTRACT:
        Uses the Immerkaer 3x3 Laplacian mask (_IMMERKAER_M).
        Scale estimator: median absolute response / (_IMMERKAER_NORM * _MAD_TO_SIGMA)
        over fully valid 3x3 windows.
        Drop mask flags pixels where absolute residual > max(k * median_abs, 1.0 / _DEPTH_SCALE).
    """
    # TODO: Student implementation
    # 1. Load depth in metres and validity mask via _consumer_depth_metres_valid
    # 2. Convolve depth with _IMMERKAER_M over 3x3 windows where all 9 pixels are valid
    # 3. Compute MAD-based noise estimate and binary drop mask
    # 4. Return (value, uint8_mask)
    raise NotImplementedError("TODO: Implement _high_frequency_depth_residual")


def frame_high_frequency_depth_residual(depth_path, residual_mask_k=5.0):
    """Robust high-frequency depth residual in metres (LowerIsBetter)."""
    return _high_frequency_depth_residual(depth_path, residual_mask_k)[0]


def frame_high_frequency_depth_residual_mask(depth_path, residual_mask_k=5.0):
    """255 where HighFrequencyDepthResidual recommends dropping a pixel."""
    return _high_frequency_depth_residual(depth_path, residual_mask_k)[1]


def _flying_pixel_ratio(depth_path, window, planarity_tol):
    """Planar-discriminated flying pixels ratio and 0/255 drop mask.

    CONTRACT:
        Identifies mixed boundary pixels lying between distinct surfaces.
        Flags pixels that belong to neither local foreground nor background fitted planes.
    """
    # TODO: Student implementation
    # 1. Check local depth extrema within (window x window) neighbourhood
    # 2. Identify step edges exceeding 2 * planarity_tol
    # 3. Fit planes via least-squares and flag unassigned intermediate pixels
    # 4. Return (ratio, uint8_mask)
    raise NotImplementedError("TODO: Implement _flying_pixel_ratio")


def frame_flying_pixel_ratio(depth_path, flying_pixel_window=5,
                             flying_pixel_planarity_tol=0.03):
    """Fraction of planar-fit-discriminated flying pixels (LowerIsBetter)."""
    return _flying_pixel_ratio(depth_path, flying_pixel_window, flying_pixel_planarity_tol)[0]


def frame_flying_pixel_ratio_mask(depth_path, flying_pixel_window=5,
                                  flying_pixel_planarity_tol=0.03):
    """255 where FlyingPixelRatio recommends dropping a pixel."""
    return _flying_pixel_ratio(depth_path, flying_pixel_window, flying_pixel_planarity_tol)[1]


def _valid_tile_coverage(depth_path, tile_size, tile_valid_floor):
    """Tile-level valid-return coverage and 0/255 drop mask.

    CONTRACT:
        Splits depth map into grid tiles of size (tile_size x tile_size).
        A tile is valid if its valid return fraction >= tile_valid_floor.
        Returns fraction of valid tiles over total tiles, and flags invalid tiles in mask.
    """
    # TODO: Student implementation
    # 1. Partition valid depth into tiles of size tile_size
    # 2. Count tiles meeting tile_valid_floor
    # 3. Return (supported_tiles / total_tiles, uint8_mask)
    raise NotImplementedError("TODO: Implement _valid_tile_coverage")


def frame_valid_tile_coverage(depth_path, tile_size=64, tile_valid_floor=0.5):
    """Fraction of depth tiles meeting the declared floor (HigherIsBetter)."""
    return _valid_tile_coverage(depth_path, tile_size, tile_valid_floor)[0]


def frame_valid_tile_coverage_mask(depth_path, tile_size=64, tile_valid_floor=0.5):
    """255 over every tile below the declared valid-return floor."""
    return _valid_tile_coverage(depth_path, tile_size, tile_valid_floor)[1]


# =============================================================================
# TODO: Student Measurers 6, 7, 8 (Depth Pair Quality Factors)
# =============================================================================
def _identity_median_depth_change(d0_path, d1_path, change_mask_k):
    """Median absolute depth change at identity over jointly valid pixels.

    CONTRACT:
        Computes |D0 - D1| where both D0 > 0 and D1 > 0.
        Calculates median and MAD. Mask flags changes > median + k * 1.4826 * MAD.
        Returns (median_m, uint8_mask, count).
    """
    # TODO: Student implementation
    # 1. Find jointly valid pixels (d0 > 0 & d1 > 0)
    # 2. Compute median absolute depth difference |d0 - d1|
    # 3. Derive adaptive MAD threshold and flag outlier changes
    # 4. Return (median, uint8_mask, joint_count)
    raise NotImplementedError("TODO: Implement _identity_median_depth_change")


def pair_identity_median_depth_change(d0_path, d1_path, change_mask_k=3.0):
    """Median absolute depth change at identity in metres (LowerIsBetter)."""
    return _identity_median_depth_change(d0_path, d1_path, change_mask_k)[0]


def pair_identity_median_depth_change_mask(d0_path, d1_path, change_mask_k=3.0):
    """255 where IdentityMedianDepthChange recommends dropping a pixel."""
    return _identity_median_depth_change(d0_path, d1_path, change_mask_k)[1]


def _joint_valid_depth_ratio(d0_path, d1_path):
    """Fraction of pixels valid in both frames and drop mask for non-joint pixels.

    CONTRACT:
        Returns count(D0 > 0 & D1 > 0) / total_pixels.
        Mask flags all pixels that are NOT valid in both frames.
        Returns (ratio, uint8_mask, count).
    """
    # TODO: Student implementation
    # 1. Compute joint mask: (d0 > 0) & (d1 > 0)
    # 2. Ratio = sum(joint) / total_pixels
    # 3. Mask = ~joint (flags non-overlapping or missing depth)
    # 4. Return (ratio, uint8_mask, joint_count)
    raise NotImplementedError("TODO: Implement _joint_valid_depth_ratio")


def pair_joint_valid_depth_ratio(d0_path, d1_path):
    """Jointly-valid depth pixel fraction in [0, 1] (HigherIsBetter)."""
    return _joint_valid_depth_ratio(d0_path, d1_path)[0]


def pair_joint_valid_depth_ratio_mask(d0_path, d1_path):
    """255 where a pixel is NOT valid in both frames."""
    return _joint_valid_depth_ratio(d0_path, d1_path)[1]


def _prior_warp_depth_residual(d0_path, d1_path, prior_T, intrinsics, depth_gate):
    """Median depth residual under constant-velocity prior, plus drop mask and count.

    CONTRACT:
        Projects valid 3D points from frame 0 into frame 1 using prior_T.
        Compares warped Z against observed target depth D1 at projected pixels (u, v).
        Returns median absolute residual over visible supported pixels.
    """
    # TODO: Student implementation
    # 1. Unproject valid points from d0 into camera-0 3D coordinates
    # 2. Transform points by prior_T to camera-1 coordinates
    # 3. Project onto frame 1 image plane and evaluate |warped_z - d1[v, u]|
    # 4. Return (median_residual, uint8_mask, support_count)
    raise NotImplementedError("TODO: Implement _prior_warp_depth_residual")


def pair_prior_warp_depth_residual(d0_path, d1_path, prior_T, intrinsics,
                                   prior_warp_depth_gate=0.10):
    """Median depth residual under constant-velocity prior (LowerIsBetter)."""
    return _prior_warp_depth_residual(
        d0_path, d1_path, prior_T, intrinsics, prior_warp_depth_gate)[0]


def pair_prior_warp_depth_residual_mask(d0_path, d1_path, prior_T, intrinsics,
                                        prior_warp_depth_gate=0.10):
    """255 where prior-warp residual exceeds depth gate."""
    return _prior_warp_depth_residual(
        d0_path, d1_path, prior_T, intrinsics, prior_warp_depth_gate)[1]

# =============================================================================
# [REST OF api.py (INFRASTRUCTURE, CLI, RDF RUNNER) REMAINS UNMODIFIED]
# =============================================================================