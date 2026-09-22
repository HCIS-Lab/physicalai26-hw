import argparse
import csv
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np
from PIL import Image

from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef, XSD

# =============================================================================
# Namespaces  (must match ontology/hw1.ttl exactly — do not rename)
#   §1 freezes this list. QUDT/UNIT/SKOS/PROV are not decoration:
#   qudt:unit is what keeps "0.41" from being a unitless number nobody can check,
#   skos: is what marks the four closed term vocabularies (Status, Polarity,
#   SettingRole, SelectionMode) as vocabularies rather than classes of measurable
#   things, and prov:wasDerivedFrom is the edge from a corrupted capture back to
#   the capture it was made from — BETWEEN BATCHES, the only place it survives
#   (§6: there are no derived experiments).
#
#   `dqv:` is deliberately absent (§1). It carried the v1 Result /
#   Metric shape, which is gone: a hw1:QualityFactor is defined by its own four
#   properties and a run outcome is an ordinary observable.
# =============================================================================
NS = "http://taica.course/hw1/ontology#"
# Instance data uses a separate namespace from the ontology vocabulary. This
# keeps v5 Turtle qnames legal and readable; legacy v4 IRIs remain under NS.
DATA_NS = "http://taica.course/hw1/data/"
HW1 = Namespace(NS)
SCHEMA = Namespace("https://schema.org/")
QUDT = Namespace("http://qudt.org/schema/qudt/")
UNIT = Namespace("http://qudt.org/vocab/unit/")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
PROV = Namespace("http://www.w3.org/ns/prov#")

_HERE = os.path.dirname(os.path.abspath(__file__))

# Path to hw1/ontology/hw1.ttl relative to this file (this file lives in hw1/).
# It is the TBox, and it is also the AUTHORITY on which parameter and factor names
# exist: see load_parameter_declarations / load_quality_factors.
_ONTOLOGY_TTL = os.path.join(_HERE, "ontology", "hw1.ttl")

# Where `experiment` writes (§3).
_EXPERIMENT_DIR = os.path.join(_HERE, "experiments")

# Depth PNG encoding: uint16 millimetres.
_DEPTH_SCALE = 1000.0

# =============================================================================
# Immerkaer noise-estimation constants — the measurement contract of
# `frame_high_frequency_depth_residual`. FIXED: they are NOT yours to derive and
# they are NOT Parameters. Unlike the tau pair they need no per-experiment
# provenance, because they are not free parameters: they are the kernel and the
# two constants of one published estimator, and changing one does not measure the
# same quantity differently — it measures a different quantity. Read these names
# in your measurer; do not paste the numbers inline.
# =============================================================================
# Immerkaer 1996's 3x3 mask: the difference of two discrete Laplacians, so it
# annihilates any locally-linear depth surface and leaves the high-frequency
# residual behind.
_IMMERKAER_M = np.array([[1.0, -2.0, 1.0],
                         [-2.0, 4.0, -2.0],
                         [1.0, -2.0, 1.0]])
# ||M|| = sqrt(sum(M^2)) = sqrt(36) = 6: the factor by which the mask amplifies
# an i.i.d. noise standard deviation.
_IMMERKAER_NORM = 6.0
# Median of |N(0,1)| = Phi^-1(0.75). Converts a robust spread back to a sigma.
_MAD_TO_SIGMA = 0.6745


# =============================================================================
# Per-frame observable measurers
#   Every one of them is PURE and DETERMINISTIC: a path (plus parameters) in, one
#   documented scalar out. No I/O beyond reading the frame, no global state, no
#   randomness — that is what makes them autogradable and what makes a stored
#   observable reproducible from the file it was measured from.
# =============================================================================
def _value_channel(rgb_path):
    """V = max(R,G,B) per pixel, float64 HxW in [0,255]. Full contract: docs/factors.md."""
    arr = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float64)
    return arr.max(axis=2)


def frame_mean_value(rgb_path):
    """BASELINE (not a factor): mean of V over one RGB frame, [0,255]. Deliberately weak. Full contract: docs/factors.md."""
    return float(_value_channel(rgb_path).mean())


def frame_clip_hi_fraction(rgb_path, tau_hi):
    """HighlightClipping: |{V >= tau_hi}| / N, LowerIsBetter, Pass iff <= maxClipHiFraction. tau_hi has no default. Full contract: docs/factors.md."""
    V = _value_channel(rgb_path)
    return float(np.count_nonzero(V >= tau_hi)) / float(V.size)


def frame_clip_lo_fraction(rgb_path, tau_lo):
    """ShadowClipping: |{V <= tau_lo}| / N, LowerIsBetter, Pass iff <= maxClipLoFraction. tau_lo has no default. Full contract: docs/factors.md."""
    V = _value_channel(rgb_path)
    return float(np.count_nonzero(V <= tau_lo)) / float(V.size)


# =============================================================================
# Depth-frame and pair observable measurers
#   Same purity contract as the RGB measurers: path(s) (plus parameters) in, one
#   documented scalar out — and, for every depth factor, a 0/255 drop mask too.
#   numpy + Pillow only, no scipy, no point cloud, no pose (the one exception:
#   PriorWarpDepthResidual receives the constant-velocity prior transform from
#   the consumer, but still compares two RASTERS).
#
#   The pair measurers exist because frame factors cannot see the ONE thing that
#   actually breaks frame-to-frame ICP: what happened BETWEEN two frames. A pair
#   of individually perfect frames taken a metre apart registers no better than
#   a pair of bad ones. Of the four literature drivers of pairwise ICP failure
#   (semantic_layer_design.md §2.1) — D1 overlap, D2 initial misalignment /
#   motion, D3 geometric degeneracy, D4 depth noise — the frame factors carry D4
#   (HighFrequencyDepthResidual, FlyingPixelRatio) and per-frame coverage
#   (ValidTileCoverage); the pair factors carry D1 (JointValidDepthRatio) and D2
#   (IdentityMedianDepthChange, PriorWarpDepthResidual).
# =============================================================================
def _consumer_depth_metres_valid(depth_path):
    """One uint16-mm depth raster as (metres, valid mask) under the consumer rule raw != 0."""
    raw = np.asarray(Image.open(depth_path))
    if raw.ndim != 2:
        raise ValueError(f"depth raster must be two-dimensional, got {raw.shape}")
    metres = raw.astype(np.float64) / _DEPTH_SCALE
    return metres, raw != 0


def _flag_mask(flagged):
    """Boolean drop flags -> on-disk mask convention (uint8 0/255)."""
    return np.where(flagged, np.uint8(255), np.uint8(0))


def _fully_valid_3x3(valid):
    if valid.shape[0] < 3 or valid.shape[1] < 3:
        return np.zeros((0, 0), dtype=bool)
    return (valid[:-2, :-2] & valid[:-2, 1:-1] & valid[:-2, 2:] &
            valid[1:-1, :-2] & valid[1:-1, 1:-1] & valid[1:-1, 2:] &
            valid[2:, :-2] & valid[2:, 1:-1] & valid[2:, 2:])


def _high_frequency_depth_residual(depth_path, residual_mask_k):
    """Shared scalar/mask core of HighFrequencyDepthResidual (Immerkaer, metres, LowerIsBetter). Full contract: docs/factors.md."""
    metres, valid = _consumer_depth_metres_valid(depth_path)
    flagged = np.zeros(valid.shape, dtype=bool)
    windows = _fully_valid_3x3(valid)
    if not windows.any():
        return float("inf"), _flag_mask(flagged)

    response = (metres[:-2, :-2] - 2.0 * metres[:-2, 1:-1] + metres[:-2, 2:] -
                2.0 * metres[1:-1, :-2] + 4.0 * metres[1:-1, 1:-1] -
                2.0 * metres[1:-1, 2:] + metres[2:, :-2] -
                2.0 * metres[2:, 1:-1] + metres[2:, 2:])
    absolute = np.abs(response)
    contributing = absolute[windows]
    median_abs = float(np.median(contributing))
    value = median_abs / (_IMMERKAER_NORM * _MAD_TO_SIGMA)

    k = float(residual_mask_k)
    if not np.isfinite(k) or k < 0:
        raise ValueError("residualMaskK must be a finite non-negative number")
    # Quantised, perfectly planar input has median_abs == 0.  In that case a
    # literal zero threshold would turn every real step edge into noise, so use
    # one millimetre-response quantum as the minimum actionable residual.
    threshold = max(k * median_abs, 1.0 / _DEPTH_SCALE)
    centres = windows & (absolute > threshold)
    flagged[1:-1, 1:-1] = centres
    return value, _flag_mask(flagged)


def frame_high_frequency_depth_residual(depth_path, residual_mask_k=5.0):
    """Robust high-frequency depth residual in metres (LowerIsBetter). Student-implemented (README.md S5.2). Full contract: docs/factors.md."""
    return _high_frequency_depth_residual(depth_path, residual_mask_k)[0]


def frame_high_frequency_depth_residual_mask(depth_path, residual_mask_k=5.0):
    """255 where HighFrequencyDepthResidual recommends dropping a pixel. Full contract: docs/factors.md."""
    return _high_frequency_depth_residual(depth_path, residual_mask_k)[1]


def _local_extrema(values, valid, radius):
    """Window min/max without scipy; invalid samples never become extrema."""
    h, w = values.shape
    lo = np.full((h, w), np.inf, dtype=np.float64)
    hi = np.full((h, w), -np.inf, dtype=np.float64)
    padded_v = np.pad(values, radius, mode="edge")
    padded_ok = np.pad(valid, radius, mode="constant", constant_values=False)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            sample = padded_v[dy:dy + h, dx:dx + w]
            ok = padded_ok[dy:dy + h, dx:dx + w]
            lo = np.minimum(lo, np.where(ok, sample, np.inf))
            hi = np.maximum(hi, np.where(ok, sample, -np.inf))
    return lo, hi


def _flying_pixel_ratio(depth_path, window, planarity_tol):
    """Shared scalar/mask core of FlyingPixelRatio (fraction, LowerIsBetter). Full contract: docs/factors.md."""
    metres, valid = _consumer_depth_metres_valid(depth_path)
    flagged = np.zeros(valid.shape, dtype=bool)
    if not valid.any():
        return float("inf"), _flag_mask(flagged)

    size = int(round(float(window)))
    if size < 3 or size % 2 == 0:
        raise ValueError("flyingPixelWindow must be an odd integer >= 3")
    tol = float(planarity_tol)
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("flyingPixelPlanarityTol must be finite and > 0 metres")

    radius = size // 2
    local_lo, local_hi = _local_extrema(metres, valid, radius)
    candidates = np.argwhere(valid & ((local_hi - local_lo) > 2.0 * tol))
    h, w = valid.shape
    for y, x in candidates:
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        ok = valid[y0:y1, x0:x1].copy()
        ok[y - y0, x - x0] = False
        yy, xx = np.nonzero(ok)
        if len(yy) < 6:
            continue
        z = metres[y0:y1, x0:x1][ok]

        # Split at the largest depth gap.  A real discontinuity supplies two
        # locally planar populations; a mixed/flying centre belongs to neither.
        order = np.argsort(z)
        sorted_z = z[order]
        gaps = np.diff(sorted_z)
        if gaps.size == 0:
            continue
        split_at = int(np.argmax(gaps)) + 1
        if gaps[split_at - 1] <= 2.0 * tol:
            continue
        low_idx, high_idx = order[:split_at], order[split_at:]
        if len(low_idx) < 3 or len(high_idx) < 3:
            continue

        coords = np.column_stack([xx + x0 - x, yy + y0 - y,
                                  np.ones(len(xx), dtype=np.float64)])
        try:
            low_plane = np.linalg.lstsq(coords[low_idx], z[low_idx], rcond=None)[0]
            high_plane = np.linalg.lstsq(coords[high_idx], z[high_idx], rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        low_pred = float(low_plane[2])
        high_pred = float(high_plane[2])
        if abs(high_pred - low_pred) <= 2.0 * tol:
            continue
        centre = float(metres[y, x])
        if (min(low_pred, high_pred) - tol <= centre <=
                max(low_pred, high_pred) + tol and
                min(abs(centre - low_pred), abs(centre - high_pred)) > tol):
            flagged[y, x] = True

    return float(np.count_nonzero(flagged)) / float(flagged.size), _flag_mask(flagged)


def frame_flying_pixel_ratio(depth_path, flying_pixel_window=5,
                             flying_pixel_planarity_tol=0.03):
    """Planar-fit-discriminated flying-pixel fraction (LowerIsBetter). Student-implemented (S5.2). Full contract: docs/factors.md."""
    return _flying_pixel_ratio(
        depth_path, flying_pixel_window, flying_pixel_planarity_tol)[0]


def frame_flying_pixel_ratio_mask(depth_path, flying_pixel_window=5,
                                  flying_pixel_planarity_tol=0.03):
    """255 where FlyingPixelRatio recommends dropping a pixel. Full contract: docs/factors.md."""
    return _flying_pixel_ratio(
        depth_path, flying_pixel_window, flying_pixel_planarity_tol)[1]


def _valid_tile_coverage(depth_path, tile_size, tile_valid_floor):
    """Shared scalar/mask core of ValidTileCoverage (fraction, HigherIsBetter). Full contract: docs/factors.md."""
    metres, valid = _consumer_depth_metres_valid(depth_path)
    del metres
    size = int(round(float(tile_size)))
    if size <= 0:
        raise ValueError("tileSize must be a positive integer")
    floor = float(tile_valid_floor)
    if not 0.0 <= floor <= 1.0:
        raise ValueError("tileValidFloor must lie in [0, 1]")
    flagged = np.zeros(valid.shape, dtype=bool)
    h, w = valid.shape
    total = supported = 0
    for y0 in range(0, h, size):
        for x0 in range(0, w, size):
            tile = valid[y0:min(h, y0 + size), x0:min(w, x0 + size)]
            total += 1
            ok = bool(tile.size and float(np.mean(tile)) >= floor)
            supported += int(ok)
            if not ok:
                flagged[y0:min(h, y0 + size), x0:min(w, x0 + size)] = True
    value = 0.0 if total == 0 else float(supported) / float(total)
    return value, _flag_mask(flagged)


def frame_valid_tile_coverage(depth_path, tile_size=64, tile_valid_floor=0.5):
    """Supported-tile fraction (HigherIsBetter). Student-implemented (S5.2). Full contract: docs/factors.md."""
    return _valid_tile_coverage(depth_path, tile_size, tile_valid_floor)[0]


def frame_valid_tile_coverage_mask(depth_path, tile_size=64,
                                   tile_valid_floor=0.5):
    """255 over every tile below the valid-return floor. Full contract: docs/factors.md."""
    return _valid_tile_coverage(depth_path, tile_size, tile_valid_floor)[1]


def _identity_median_depth_change(d0_path, d1_path, change_mask_k):
    """Shared scalar/mask/count core of IdentityMedianDepthChange (metres, LowerIsBetter). Full contract: docs/factors.md."""
    d0, v0 = _consumer_depth_metres_valid(d0_path)
    d1, v1 = _consumer_depth_metres_valid(d1_path)
    flagged = np.zeros(d0.shape, dtype=bool)
    if d0.shape != d1.shape:
        return float("inf"), _flag_mask(flagged), 0
    joint = v0 & v1
    count = int(np.count_nonzero(joint))
    if count == 0:
        return float("inf"), _flag_mask(flagged), 0
    change = np.abs(d0 - d1)
    values = change[joint]
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    k = float(change_mask_k)
    if not np.isfinite(k) or k < 0:
        raise ValueError("changeMaskK must be a finite non-negative number")
    threshold = median + k * 1.4826 * mad
    # With a zero MAD, keep ordinary coherent motion and flag only values that
    # exceed the median by at least one quantisation step.
    threshold = max(threshold, median + 1.0 / _DEPTH_SCALE)
    flagged = joint & (change > threshold)
    return median, _flag_mask(flagged), count


def pair_identity_median_depth_change(d0_path, d1_path, change_mask_k=3.0):
    """Median |D0-D1| at identity in metres (LowerIsBetter). Student-implemented (S5.2). Full contract: docs/factors.md."""
    return _identity_median_depth_change(d0_path, d1_path, change_mask_k)[0]


def pair_identity_median_depth_change_mask(d0_path, d1_path, change_mask_k=3.0):
    """255 where IdentityMedianDepthChange recommends dropping a pixel. Full contract: docs/factors.md."""
    return _identity_median_depth_change(d0_path, d1_path, change_mask_k)[1]


def _joint_valid_depth_ratio(d0_path, d1_path):
    """Shared scalar/mask/count core of JointValidDepthRatio ([0,1], HigherIsBetter). Full contract: docs/factors.md."""
    d0, v0 = _consumer_depth_metres_valid(d0_path)
    d1, v1 = _consumer_depth_metres_valid(d1_path)
    flagged = np.ones(d0.shape, dtype=bool)
    if d0.shape != d1.shape or d0.size == 0:
        return 0.0, _flag_mask(flagged), 0
    joint = v0 & v1
    count = int(np.count_nonzero(joint))
    return float(count) / float(joint.size), _flag_mask(~joint), count


def pair_joint_valid_depth_ratio(d0_path, d1_path):
    """Jointly-valid depth fraction in [0,1] (HigherIsBetter). Student-implemented (S5.2). Full contract: docs/factors.md."""
    return _joint_valid_depth_ratio(d0_path, d1_path)[0]


def pair_joint_valid_depth_ratio_mask(d0_path, d1_path):
    """255 where a pixel is NOT valid in both frames. Full contract: docs/factors.md."""
    return _joint_valid_depth_ratio(d0_path, d1_path)[1]


def _camera_intrinsics(intrinsics, shape):
    """Accept the capture dict, a (width,height,hfov) tuple, or a 3x3 K."""
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
        raise ValueError(
            f"intrinsics ({width}x{height}) do not match depth raster ({w}x{h})")
    fx = fy = (width / 2.0) / np.tan(np.radians(hfov / 2.0))
    return fx, fy, width / 2.0, height / 2.0


def _prior_warp(d0_m, d1_m, prior_T, intrinsics, depth_gate):
    """Warp depth-0 pixels into depth 1; residual/support/projected rasters at source coordinates. Full contract: docs/factors.md."""
    d0 = np.asarray(d0_m, dtype=np.float64)
    d1 = np.asarray(d1_m, dtype=np.float64)
    if d0.shape != d1.shape or d0.ndim != 2:
        raise ValueError("prior-warp depth rasters must be same-shape HxW arrays")
    gate = float(depth_gate)
    if not np.isfinite(gate) or gate <= 0:
        raise ValueError("priorWarpDepthGate must be finite and > 0 metres")
    fx, fy, cx, cy = _camera_intrinsics(intrinsics, d0.shape)
    transform = np.asarray(prior_T, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("prior_T must be a finite 4x4 transform")

    source_valid = d0 > 0
    y, x = np.nonzero(source_valid)
    residual = np.full(d0.shape, np.nan, dtype=np.float64)
    support = np.zeros(d0.shape, dtype=bool)
    projected = np.zeros(d0.shape, dtype=bool)
    if len(x) == 0:
        return residual, support, projected

    z = d0[y, x]
    xyz1 = np.vstack([(x - cx) * z / fx, (y - cy) * z / fy, z,
                      np.ones(len(z), dtype=np.float64)])
    warped = transform @ xyz1
    wz = warped[2]
    in_front = wz > 0
    u = np.rint(fx * warped[0] / np.where(in_front, wz, 1.0) + cx).astype(int)
    v = np.rint(fy * warped[1] / np.where(in_front, wz, 1.0) + cy).astype(int)
    h, w = d0.shape
    inside = in_front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    src_y, src_x = y[inside], x[inside]
    dst_y, dst_x = v[inside], u[inside]
    warped_z = wz[inside]
    target_z = d1[dst_y, dst_x]
    target_valid = target_z > 0
    src_y, src_x = src_y[target_valid], src_x[target_valid]
    warped_z, target_z = warped_z[target_valid], target_z[target_valid]
    if len(src_x) == 0:
        return residual, support, projected

    r = np.abs(warped_z - target_z)
    projected[src_y, src_x] = True
    residual[src_y, src_x] = r
    # A point substantially behind the observed surface is occluded and is not
    # evidence about the prior residual distribution.  Points in front remain
    # contributing outliers and are exactly the ones the filter can remove.
    visible = warped_z <= target_z + gate
    support[src_y[visible], src_x[visible]] = True
    return residual, support, projected


def _prior_warp_depth_residual(d0_path, d1_path, prior_T, intrinsics, depth_gate):
    """Shared scalar/mask/count core of PriorWarpDepthResidual (metres, LowerIsBetter). Full contract: docs/factors.md."""
    d0, _ = _consumer_depth_metres_valid(d0_path)
    d1, _ = _consumer_depth_metres_valid(d1_path)
    if d0.shape != d1.shape:
        return float("inf"), _flag_mask(np.zeros(d0.shape, dtype=bool)), 0
    residual, support, projected = _prior_warp(
        d0, d1, prior_T, intrinsics, depth_gate)
    count = int(np.count_nonzero(support))
    if count == 0:
        return float("inf"), _flag_mask(np.zeros(d0.shape, dtype=bool)), 0
    value = float(np.median(residual[support]))
    flagged = projected & np.isfinite(residual) & (residual > float(depth_gate))
    return value, _flag_mask(flagged), count


def pair_prior_warp_depth_residual(d0_path, d1_path, prior_T, intrinsics,
                                   prior_warp_depth_gate=0.10):
    """Median residual under the constant-velocity prior (LowerIsBetter). Student-implemented (S5.2). Full contract: docs/factors.md."""
    return _prior_warp_depth_residual(
        d0_path, d1_path, prior_T, intrinsics, prior_warp_depth_gate)[0]


def pair_prior_warp_depth_residual_mask(d0_path, d1_path, prior_T, intrinsics,
                                        prior_warp_depth_gate=0.10):
    """255 where the prior-warp residual exceeds the depth gate. Full contract: docs/factors.md."""
    return _prior_warp_depth_residual(
        d0_path, d1_path, prior_T, intrinsics, prior_warp_depth_gate)[1]


# =============================================================================
# Frame pairing  (rgb/*.png <-> depth/*.png by integer stem, iterate sorted by int)
# =============================================================================
def _stem(path):
    """Integer filename stem of a frame path ('.../17.png' -> 17)."""
    return int(os.path.splitext(os.path.basename(path))[0])


def _pair_frames(data_dir):
    """[(stem, rgb, depth)] paired by int stem, sorted. Full strategy note: docs/triplestore.md."""
    rgb_dir = os.path.join(data_dir, "rgb")
    depth_dir = os.path.join(data_dir, "depth")
    if not os.path.isdir(rgb_dir) or not os.path.isdir(depth_dir):
        raise ValueError(f"data-dir must contain rgb/ and depth/ subdirs: {data_dir!r}")

    rgb = {_stem(p): p for p in glob.glob(os.path.join(rgb_dir, "*.png"))}
    depth = {_stem(p): p for p in glob.glob(os.path.join(depth_dir, "*.png"))}
    common = sorted(set(rgb) & set(depth))
    if not common:
        raise ValueError(f"No frames present in BOTH rgb/ and depth/ under: {data_dir!r}")
    return [(str(s), rgb[s], depth[s]) for s in common]


# =============================================================================
# IRI helpers — the frozen scheme of §2, in ONE place
#   Every module that needs an IRI of this assignment calls a function from this
#   section. Nobody, in any file, writes an f-string with `batch/` in it, and
#   nobody writes `str(iri).split("/")[-1]`: the scheme is a contract between
#   `batch2ttl` (which mints the structural IRIs), `experiment` (which measures
#   those subjects) and `reconstruct.py` (which reads them back), and a contract
#   duplicated across three files is a contract that will disagree with itself the
#   first time anyone renames a segment.
#
#   TWO TIERS, AND THE RULE THAT MUST NEVER BE RELAXED (§2).
#   Batch, frame and image IRIs are SHARED STRUCTURE: they name pixels on disk, so
#   every experiment over one capture reaches the same nodes. Everything an
#   experiment mints — settings, annotations, pairs, runs — is EXPERIMENT-SCOPED,
#   i.e. hangs under `<ns>experiment/<expname>/`. That scoping is what replaced v1's
#   named graphs: two experiments over one batch produce two disjoint annotation /
#   pair / run sets over the SAME frame IRIs, so nothing collides and nothing is
#   overwritten. A loader can forget a `to_graph` argument; it cannot forget the
#   experiment name, because the name is inside the subject.
# =============================================================================

# The two modality segments of an annotation IRI and of `component_iri` — frozen
# (§2). They are the SAME two strings on purpose: an annotation's
# `<kind>` names the image node it describes, so `annotation_iri(name, n, kind)`
# and `component_iri(batch, n, kind)` line up segment for segment.
_ANNOTATION_KINDS = ("rgb", "depth")

# An experiment name is the declaration file's stem AND the IRI tail (§2/§6), so
# it has to survive both a filesystem and an IRI without quoting: letters, digits,
# underscore and hyphen. A name with a slash in it would silently restructure the
# IRI scheme; a name with a space in it would not survive a Turtle IRI at all.
_EXPNAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def batch_name(data_dir, floor):
    """Floor-qualified batch name floor{floor}_{basename}. Full strategy note: docs/triplestore.md."""
    return f"floor{int(floor)}_{os.path.basename(os.path.normpath(data_dir))}"


def batch_iri(name):
    """Batch node IRI <ns>batch/<name>. Full strategy note: docs/triplestore.md."""
    return URIRef(f"{NS}batch/{name}")


def frame_iri(name, idx):
    """Frame node IRI, <n> decimal unpadded. Full strategy note: docs/triplestore.md."""
    return URIRef(f"{NS}batch/{name}/frame/{idx}")


def component_iri(name, idx, kind):
    """Image node IRI (rgb|depth); carries contentUrl ONLY, never observables. Full strategy note: docs/triplestore.md."""
    return URIRef(f"{NS}batch/{name}/frame/{idx}/{kind}")


def generation_setting_iri(name, param):
    """Batch-scoped generation-setting IRI. Full strategy note: docs/triplestore.md."""
    return URIRef(f"{NS}batch/{name}/setting/{param}")


def experiment_iri(expname):
    """Experiment IRI <ns>experiment/<expname>; the isolation prefix. Full strategy note: docs/triplestore.md."""
    return URIRef(f"{NS}experiment/{expname}")


def setting_iri(expname, factor_local, param_local):
    """Experiment-scoped machine-minted setting IRI (defaults completing the required set). Full strategy note: docs/triplestore.md."""
    return URIRef(f"{NS}experiment/{expname}/setting/{factor_local}/{param_local}")


def data_batch_iri(name):
    return URIRef(f"{DATA_NS}batch/{name}")


def data_frame_iri(name, idx):
    return URIRef(f"{DATA_NS}batch/{name}/frame/{int(idx)}")


def data_component_iri(name, idx, kind):
    if kind not in ("rgb", "depth"):
        raise ValueError(f"unknown image kind {kind!r}")
    return URIRef(f"{DATA_NS}batch/{name}/{kind}/{int(idx)}")


def data_experiment_iri(expname):
    return URIRef(f"{DATA_NS}experiment/{expname}")


def data_setting_iri(expname, factor_local, param_local):
    return URIRef(f"{DATA_NS}experiment/{expname}/setting/{factor_local}_{param_local}")


def data_run_iri(expname, mode):
    return URIRef(f"{DATA_NS}experiment/{expname}/run/{mode}")


def data_factor_iri(expname, factor_local, current_idx, previous_idx=None):
    local = (f"{factor_local}_{int(previous_idx)}_{int(current_idx)}"
             if previous_idx is not None else f"{factor_local}_{int(current_idx)}")
    return URIRef(f"{DATA_NS}experiment/{expname}/factor/{local}")


def annotation_iri(expname, idx, kind):
    """Per-modality annotation IRI; the OBSERVATION node (values+statuses). Full strategy note: docs/triplestore.md."""
    if kind not in _ANNOTATION_KINDS:
        raise ValueError(
            f"annotation kind {kind!r} is not one of {', '.join(_ANNOTATION_KINDS)}; "
            f"the <kind> segment of an annotation IRI is frozen (§2)")
    return URIRef(f"{NS}experiment/{expname}/annotation/{idx}/{kind}")


def pair_iri(expname, i, j):
    """Ordered experiment-scoped pair IRI <i>_<j>, stems not ordinals. Full strategy note: docs/triplestore.md."""
    return URIRef(f"{NS}experiment/{expname}/pair/{i}_{j}")


def run_iri(expname, mode):
    """Run IRI <exp>/run/<mode> (baseline|selected|masked). Full strategy note: docs/triplestore.md."""
    return URIRef(f"{NS}experiment/{expname}/run/{mode}")


def factor_iri(expname, factor_local, current_idx, previous_idx=None):
    """Readable, experiment-scoped IRI for one Factor occurrence.

    The definition is part of the key (not merely a label): two selected
    definitions over the same image are distinct occurrences.  Pair keys retain
    the ordered source/target stems.  Keeping this helper beside the other IRI
    constructors prevents consumers from inferring occurrence identity from
    predicates or mask paths.
    """
    local = f"{factor_local}_{int(previous_idx)}_{int(current_idx)}" if previous_idx is not None \
        else f"{factor_local}_{int(current_idx)}"
    return URIRef(f"{NS}experiment/{expname}/factor/{local}")


def frame_index_from_iri(iri):
    """Frame/annotation IRI -> int stem. THE tail parse; do not re-derive. Full strategy note: docs/triplestore.md."""
    text = str(iri)
    expected = (f"{NS}batch/<name>/frame/<n> or "
                f"{DATA_NS}batch/<name>/frame/<n> or "
                f"{NS}experiment/<expname>/annotation/<n>/<kind>")
    if (text.startswith(f"{NS}batch/") or text.startswith(f"{DATA_NS}batch/")) and "/frame/" in text:
        tail = text.split("/frame/", 1)[1]
    elif (text.startswith(f"{NS}experiment/") or text.startswith(f"{DATA_NS}experiment/")) and "/annotation/" in text:
        tail, _, kind = text.split("/annotation/", 1)[1].partition("/")
        if kind not in _ANNOTATION_KINDS:
            raise ValueError(
                f"annotation IRI {text!r} ends in modality segment {kind!r}; "
                f"expected one of {', '.join(_ANNOTATION_KINDS)} ({expected})")
    else:
        raise ValueError(
            f"not a frame or annotation IRI of this assignment: {text!r} "
            f"(expected {expected})")
    if not tail.isdigit():
        raise ValueError(
            f"IRI tail {tail!r} is not a decimal frame index in {text!r} "
            f"(expected {expected})")
    return int(tail)


def batch_name_from_frame_iri(iri):
    """Frame IRI -> embedded batch name (batch-match guard). Full strategy note: docs/triplestore.md."""
    text = str(iri)
    prefix = f"{NS}batch/"
    data_prefix = f"{DATA_NS}batch/"
    marker = "/frame/"
    if not ((text.startswith(prefix) or text.startswith(data_prefix)) and marker in text):
        raise ValueError(
            f"not a frame IRI of this assignment: {text!r} "
            f"(expected {NS}batch/<name>/frame/<n>)")
    base = prefix if text.startswith(prefix) else data_prefix
    return text[len(base):].split(marker, 1)[0]


def _batch_name_from_batch_iri(iri):
    """Batch IRI -> its <name>. Private."""
    text = str(iri)
    prefix = f"{NS}batch/"
    data_prefix = f"{DATA_NS}batch/"
    if not (text.startswith(prefix) or text.startswith(data_prefix)):
        raise ValueError(f"not a batch IRI of this assignment: {text!r} "
                         f"(expected {NS}batch/<name>)")
    base = prefix if text.startswith(prefix) else data_prefix
    return text[len(base):]


def _experiment_name_from_iri(iri, path=None):
    """Experiment IRI -> <expname>, validated against _EXPNAME_RE. Full strategy note: docs/triplestore.md."""
    text = str(iri)
    prefix = f"{NS}experiment/"
    data_prefix = f"{DATA_NS}experiment/"
    where = f"{path}: " if path else ""
    if not (text.startswith(prefix) or text.startswith(data_prefix)):
        raise ValueError(f"{where}not an experiment IRI of this assignment: {text!r} "
                         f"(expected {NS}experiment/<expname>)")
    base = prefix if text.startswith(prefix) else data_prefix
    name = text[len(base):]
    if not _EXPNAME_RE.match(name):
        raise ValueError(
            f"{where}experiment IRI tail {name!r} is not a legal experiment name "
            f"(§2: [A-Za-z0-9_-]+, and it must equal the declaration "
            f"file's stem)")
    return name


# Backwards-compatible aliases. The two helpers were private (`_frame_iri`,
# `_component_iri`) while this file was the only caller; §9 makes
# them public because reconstruct.py and the tests import them. The old names
# stay bound so nothing that already imports them breaks on the rename.
_frame_iri = frame_iri
_component_iri = component_iri


# =============================================================================
# The TBox as the authority on names
#   Parameter names, factor names, polarities and thresholds are NOT string
#   constants in this file. They are read out of ontology/hw1.ttl at run time, so
#   the ontology is a thing the code OBEYS rather than a document that describes
#   it. That is the whole argument for having a TBox in a pipeline this small:
#   a declared `hw1:settingParameter hw1:tuaHi` is a hard error with the declared
#   list printed, not a seventeenth FactorSetting nobody ever reads recording a
#   treatment nobody ran.
#
#   Three loaders, one status rule: `load_parameter_declarations` (what may be
#   set, and how it parses), `load_quality_factors` (what is measured, which way
#   is better, and which parameter is its threshold) and `status_for` (the ONE
#   implementation of the §4.5 Pass rule — `experiment` calls it for
#   frame and pair observables, `write_run` for mapMeanL2 and coverageF, and
#   nobody anywhere re-derives `>=` versus `<=`).
# =============================================================================
_SETTING_ROLES = ("GenerationSetting", "MeasurementSetting", "QualificationSetting")
_VALUE_KINDS = ("double", "integer", "string")

# The two roles a setting may have ON AN EXPERIMENT, and therefore the two roles
# the selection-scoped completeness rule of §4.2 draws from. Generation
# is deliberately absent: those settings live on the Batch, they describe the pixels,
# and they are never defaulted — see `_resolve_generation_settings` and
# `read_declaration`, which rejects a Generation parameter in a declaration outright.
_EXPERIMENT_ROLES = ("MeasurementSetting", "QualificationSetting")


def _local(term):
    """Local name of a term in the hw1 namespace."""
    text = str(term)
    if text.startswith(NS):
        return text[len(NS):]
    return text.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def _storable(value):
    """Round a double to the precision the .ttl file holds (rdflib: 7 sig digits); non-finite passes through. Full strategy note: docs/triplestore.md."""
    return float(f"{float(value):e}")


def _double_literal(value):
    """The one way to write xsd:double (S4.6): graded precision, XSD-valid INF/-INF/NaN. Full strategy note: docs/triplestore.md."""
    v = _storable(value)
    # NaN is the only value that is not equal to itself — the cheapest exact test,
    # and it avoids importing `math` for three comparisons.
    if v != v:
        return Literal("NaN", datatype=XSD.double, normalize=False)
    if v == float("inf"):
        return Literal("INF", datatype=XSD.double, normalize=False)
    if v == float("-inf"):
        return Literal("-INF", datatype=XSD.double, normalize=False)
    return Literal(v, datatype=XSD.double)


def _setting_value_literal(value, kind):
    """Typed literal for a setting value by parameter kind. Full strategy note: docs/triplestore.md."""
    if kind == "string":
        return Literal(str(value), datatype=XSD.string)
    if kind == "integer":
        return Literal(int(value), datatype=XSD.integer)
    return _double_literal(value)


def load_parameter_declarations(path=_ONTOLOGY_TTL):
    """TBox authority #1: what may be set and how it parses. Full strategy note: docs/triplestore.md."""
    g = Graph()
    g.parse(path, format="turtle")
    decls = {}
    for param in g.subjects(RDF.type, HW1.Parameter):
        local = _local(param)

        role_term = g.value(param, HW1.paramRole)
        role = _local(role_term) if role_term is not None else None
        if role not in _SETTING_ROLES:
            raise ValueError(
                f"{path}: parameter {local!r} declares hw1:paramRole {role!r}; "
                f"expected one of {', '.join(_SETTING_ROLES)}")

        kind_term = g.value(param, HW1.paramValueKind)
        kind = str(kind_term) if kind_term is not None else None
        if kind not in _VALUE_KINDS:
            raise ValueError(
                f"{path}: parameter {local!r} declares hw1:paramValueKind {kind!r}; "
                f"expected one of {', '.join(_VALUE_KINDS)}")

        # Single-valued and mandatory: it is a segment of the setting IRI and of
        # the digest line, so "several" and "none" are equally unusable.
        primaries = sorted(g.objects(param, HW1.paramPrimaryFactor), key=str)
        if len(primaries) != 1:
            raise ValueError(
                f"{path}: parameter {local!r} has {len(primaries)} "
                f"hw1:paramPrimaryFactor values; exactly one is required "
                f"(§5) because it names the setting IRI and the digest line")
        affects = sorted((URIRef(str(a)) for a in g.objects(param, HW1.paramAffectsFactor)),
                         key=str)

        default_term = g.value(param, HW1.paramDefault)
        if default_term is None:
            default = None
        else:
            raw = default_term.toPython() if hasattr(default_term, "toPython") \
                else str(default_term)
            try:
                default = {"double": float, "integer": int, "string": str}[kind](raw)
            except (TypeError, ValueError):
                raise ValueError(
                    f"{path}: parameter {local!r} declares hw1:paramDefault "
                    f"{str(default_term)!r}, which is not a {kind}") from None

        decls[local] = {
            "iri": URIRef(str(param)),
            "role": role,
            "kind": kind,
            "primary": _local(primaries[0]),
            "primaryIri": URIRef(str(primaries[0])),
            "affects": tuple(_local(a) for a in affects),
            "affectsIris": tuple(affects),
            "default": default,
        }
    return decls


def parse_setting_arg(arg, decls):
    """Parse one NAME=VALUE setting against the TBox declarations. Full strategy note: docs/triplestore.md."""
    if "=" not in arg:
        raise ValueError(f"a setting expects NAME=VALUE, got {arg!r}")
    name, _, text = arg.partition("=")
    name = name.strip()
    if name not in decls:
        raise ValueError(
            f"undeclared parameter {name!r}. Parameter names come from the TBox, not "
            f"from the command line; declared parameters are: "
            f"{', '.join(sorted(decls)) or '(none)'}")
    kind = decls[name]["kind"]
    if kind == "string":
        return name, text
    try:
        return name, (int(text) if kind == "integer" else float(text))
    except (TypeError, ValueError):
        raise ValueError(
            f"parameter {name!r} is declared hw1:paramValueKind {kind!r}; "
            f"{text!r} does not parse as one") from None


def load_quality_factors(path=_ONTOLOGY_TTL):
    """TBox authority #2: what is measured, polarity, threshold link. Full strategy note: docs/triplestore.md."""
    g = Graph()
    g.parse(path, format="turtle")
    out = {}
    for f in g.subjects(RDF.type, HW1.QualityFactor):
        local = _local(f)
        over = g.value(f, HW1.overProperty)
        status = g.value(f, HW1.statusProperty)
        pol = g.value(f, HW1.polarity)
        qual = g.value(f, HW1.qualifiedBy)
        # Semantic schema uses generic hw1:value/hw1:status on occurrences.
        # Keep the internal observable key for measurement dispatch and grading;
        # it is no longer serialized as a per-metric RDF predicate.
        semantic = over is None and status is None and local in (
            set(_FRAME_OBSERVABLES) | set(_PAIR_OBSERVABLES) |
            {"HighlightClipping", "ShadowClipping", "HighFrequencyDepthResidual",
             "FlyingPixelRatio", "ValidTileCoverage", "IdentityMedianDepthChange",
             "JointValidDepthRatio", "PriorWarpDepthResidual"})
        # Run-level definitions may also adopt generic result predicates.  They
        # are retained for write_run compatibility, so use their own local name
        # as the internal observable key when the new TBox omits legacy links.
        if over is None and status is None and _semantic_schema_enabled(path):
            semantic = True
        if semantic:
            aliases = {"HighlightClipping": "clipHiFraction", "ShadowClipping": "clipLoFraction",
                       "HighFrequencyDepthResidual": "highFrequencyDepthResidual", "FlyingPixelRatio": "flyingPixelRatio",
                       "ValidTileCoverage": "validTileCoverage", "IdentityMedianDepthChange": "identityMedianDepthChange",
                       "JointValidDepthRatio": "jointValidDepthRatio", "PriorWarpDepthResidual": "priorWarpDepthResidual"}
            over = URIRef(f"{NS}{aliases.get(local, local)}")
            status = HW1.status
        missing = [n for n, t in (("hw1:overProperty", over),
                                  ("hw1:statusProperty", status),
                                  ("hw1:polarity", pol),
                                  ("hw1:qualifiedBy", qual)) if t is None]
        if missing:
            raise ValueError(
                f"{path}: hw1:QualityFactor {local!r} is missing {', '.join(missing)}. "
                f"All four links are required (§4.5): without them the "
                f"factor cannot be graded, and a factor that cannot be graded reads as "
                f"a factor that always passes.")
        polarity = {"HigherIsBetter": "higher", "LowerIsBetter": "lower"}.get(_local(pol))
        if polarity is None:
            raise ValueError(
                f"{path}: factor {local!r} declares hw1:polarity {_local(pol)!r}; "
                f"expected hw1:HigherIsBetter or hw1:LowerIsBetter")
        out[local] = {"over": _local(over), "status": _local(status),
                      "polarity": polarity, "qualifiedBy": _local(qual)}
    return out


def status_for(value_property_local, value, settings, factors=None):
    """THE Pass rule (S4.5): value vs threshold by polarity, fail-closed on non-finite. Full strategy note: docs/triplestore.md."""
    factors = load_quality_factors() if factors is None else factors
    over = sorted(k for k, info in factors.items()
                  if info["over"] == value_property_local)
    if not over:
        raise ValueError(
            f"no hw1:QualityFactor is declared over hw1:{value_property_local} in "
            f"{_ONTOLOGY_TTL}, so it has no polarity, no threshold and no status. "
            f"hw1:meanValue is the deliberate case (§4.3); anything else "
            f"is a missing TBox declaration.")
    if len(over) > 1:
        raise ValueError(
            f"{len(over)} quality factors ({', '.join(over)}) are declared over "
            f"hw1:{value_property_local}. One observable has one verdict; two factors "
            f"over it would write two contradictory {value_property_local}Status "
            f"triples onto one node.")
    info = factors[over[0]]

    param = info["qualifiedBy"]
    if param not in settings:
        raise ValueError(
            f"factor {over[0]!r} is qualified by hw1:{param}, which is not in the "
            f"setting vector of this experiment. Every Qualification parameter is "
            f"recorded on every experiment (§5), so this means the vector "
            f"is not total — and guessing a threshold would bake a verdict nobody can "
            f"reproduce.")

    v = float(value)
    t = float(settings[param])
    passing = (v >= t) if info["polarity"] == "higher" else (v <= t)
    return HW1.Pass if passing else HW1.Fail


# =============================================================================
# Experiment files — two sections, one marker, one seal  (§3.1)
#   An experiment file is not machine-generated any more. It is a STUDENT
#   DECLARATION with a machine section appended to it:
#
#       <student turtle: Experiment node, factor selection, settings>
#       # ============ MACHINE SECTION (regenerated by api.py — do not edit) ===
#       <machine turtle: default-filled settings, annotations, pairs, runs>
#
#   Three writers/readers touch that structure and they have to agree to the
#   byte: `cmd_experiment` appends the marker and stamps the seal,
#   `read_experiment` verifies it, `write_run` verifies it and re-serializes the
#   half below the marker. That is why the split and the digest live HERE, once,
#   and why none of them does its own `text.find(...)`.
#
#   v2's DIGEST IDENTITY IS DELETED (§6/§11) — `experiment_digest`,
#   `hw1:experimentId`, `--exp-id` and the `<id>.ttl` filenames with it. An
#   experiment is identified by its NAME now, and what the digest used to buy —
#   "one treatment, one file, nothing overwritten" — is bought by write-once plus
#   the seal below, which is a stronger guarantee: it survives hand editing.
# =============================================================================

# The frozen marker line (§3.1), exported as part of the public
# surface (§8) because the tests and `explore` split files on it too. Matched as
# a WHOLE LINE, em dash and all: a "close enough" spelling would either cut a
# file at a line that is not the marker, or fail to find the marker and report a
# sealed experiment as an unassessed declaration.
MACHINE_MARKER = "# ============ MACHINE SECTION (regenerated by api.py — do not edit) ============"


def _marker_offset(text):
    """Locate MACHINE_MARKER in an experiment file. Full strategy note: docs/triplestore.md."""
    start = 0
    while True:
        idx = text.find(MACHINE_MARKER, start)
        if idx < 0:
            return None
        end = idx + len(MACHINE_MARKER)
        at_line_start = idx == 0 or text[idx - 1] == "\n"
        at_line_end = end == len(text) or text[end] == "\n"
        if at_line_start and at_line_end:
            return idx
        start = idx + 1


def _split_sections(text):
    """Split declaration vs machine section at the marker. Full strategy note: docs/triplestore.md."""
    idx = _marker_offset(text)
    if idx is None:
        return text, None
    rest = text[idx + len(MACHINE_MARKER):]
    if rest.startswith("\n"):
        rest = rest[1:]
    return text[:idx], rest


def _declaration_digest(student_bytes):
    """SHA-256 seal of the declaration half. Full strategy note: docs/triplestore.md."""
    return hashlib.sha256(student_bytes).hexdigest()


def _read_text(path):
    """The whole file as text. Turtle is UTF-8 by specification (and so is this)."""
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _verify_seal(g, exp, student_text, path):
    """Check the stored seal against the declaration; hard error on mismatch. Full strategy note: docs/triplestore.md."""
    stored = g.value(exp, HW1.declarationDigest)
    if stored is None:
        raise ValueError(
            f"{path}: experiment {exp} carries no hw1:declarationDigest. Every "
            f"assessed file is sealed by `api.py experiment` (§3.1), "
            f"so a machine section without a seal was not written by this program.")
    actual = _declaration_digest(student_text.encode("utf-8"))
    if str(stored) != actual:
        raise ValueError(
            f"{path}: THE DECLARATION WAS EDITED AFTER ASSESSMENT. The recorded "
            f"hw1:declarationDigest ({str(stored)[:12]}…) does not match the bytes "
            f"above the marker ({actual[:12]}…), so the verdicts below the marker "
            f"were computed for a treatment this file no longer declares. An "
            f"edited declaration IS A NEW EXPERIMENT AND NEEDS A NEW FILE: copy "
            f"the declaration to a new name and run `api.py experiment "
            f"<newname>.ttl` (§3.1 — experiments are write-once).")


# =============================================================================
# Experiment files — the ONE reader and the ONE writer of run triples
#   `read_experiment` (§8.2) is the only code in the project that
#   turns an ASSESSED experiment Turtle into Python, and `write_run` (§8.2) is the
#   only code that emits a hw1:ReconstructionRun triple. reconstruct.py,
#   completeness.py and any other evaluator go through these two: none of them
#   parses Turtle, none of them re-derives the IRI scheme, none of them re-derives
#   the usable-link rule of §4.5, and none of them re-derives the Pass/Fail rule
#   (that one lives in `status_for`, and `write_run` calls it).
#
#   BOTH VERIFY THE SEAL FIRST (§3.1, new in v3), and `write_run`
#   rewrites ONLY below the marker. `read_declaration` — further down, with the
#   `experiment` command — is the reader for the other half of the file: the
#   student's declaration, before any of this exists.
#
#   v1's `write_result` / `hw1:Result` / `load_result_measures` shape is DELETED
#   (§11). A reconstruction outcome is now an ordinary observable —
#   `hw1:mapMeanL2` + `hw1:mapMeanL2Status` on a hw1:ReconstructionRun, graded by
#   the ordinary hw1:ReconstructionAccuracy factor — so nothing in this section
#   is special-cased for results any more.
# =============================================================================
def _sole_experiment(g, path):
    """Exactly one Experiment subject per file, else error. Full strategy note: docs/triplestore.md."""
    exps = sorted(set(g.subjects(RDF.type, HW1.Experiment)), key=str)
    if len(exps) != 1:
        raise ValueError(
            f"{path}: expected exactly one hw1:Experiment subject, found {len(exps)}"
            + (f": {', '.join(str(e) for e in exps)}" if exps else ""))
    return exps[0]


def _setting_value_to_python(lit, path, param_local):
    """Setting literal -> python value by parameter kind. Full strategy note: docs/triplestore.md."""
    dt = lit.datatype
    if dt == XSD.integer:
        return int(lit)
    if dt in (XSD.double, XSD.decimal, XSD.float):
        return float(lit)
    if dt is None or dt == XSD.string:
        return str(lit)
    raise ValueError(
        f"{path}: hw1:settingValue of {param_local!r} carries datatype {dt}; "
        f"§5 declares exactly one of xsd:double / xsd:integer / "
        f"xsd:string, one per hw1:paramValueKind")


def _experiment_settings(g, exp, path):
    """Experiment's full setting vector (declared + defaulted). Full strategy note: docs/triplestore.md."""
    settings = {}
    for setting in g.objects(exp, HW1.hasFactorSetting):
        param = g.value(setting, HW1.settingParameter)
        if param is None:
            raise ValueError(
                f"{path}: hw1:FactorSetting {setting} has no hw1:settingParameter")
        local = _local(param)
        lit = g.value(setting, HW1.settingValue)
        if lit is None:
            raise ValueError(
                f"{path}: hw1:FactorSetting for {local!r} has no hw1:settingValue")
        value = _setting_value_to_python(lit, path, local)
        if local in settings and settings[local] != value:
            raise ValueError(
                f"{path}: parameter {local!r} is recorded twice with different values "
                f"({settings[local]!r} and {value!r}); one experiment is ONE treatment "
                f"(§3.1: two levels are two experiments, and the second "
                f"one needs its own declaration file)")
        settings[local] = value
    return settings


def _annotation_frame_index(g, ann, path):
    """Annotation node -> int frame stem. Full strategy note: docs/triplestore.md."""
    idx = g.value(ann, HW1.frameIndex)
    if idx is not None:
        return int(idx)
    frame = g.value(ann, HW1.annotatesFrame)
    if frame is None:
        raise ValueError(
            f"{path}: hw1:FrameAnnotation {ann} carries neither hw1:frameIndex nor "
            f"hw1:annotatesFrame, so nothing says which frame it is about")
    return frame_index_from_iri(frame)


def _mask_factor_from_path(mask_file, path):
    """Extract the factor directory from ``.../masks/<factor>/<file>.png``."""
    parts = str(mask_file).replace("\\", "/").split("/")
    try:
        index = parts.index("masks")
        factor = parts[index + 1]
    except (ValueError, IndexError):
        raise ValueError(
            f"{path}: hw1:maskFile {str(mask_file)!r} does not follow "
            "<experiment>/masks/<factor>/<file>.png") from None
    if not factor:
        raise ValueError(f"{path}: hw1:maskFile has an empty factor directory")
    return factor


_USABLE_LINKS_QUERY = """\
PREFIX hw1: <%s>
SELECT ?pairIndex ?source ?target WHERE {
  ?experiment hw1:producesPair ?pair .
  ?pair hw1:pairIndex ?pairIndex ;
        hw1:sourceFrame ?source ;
        hw1:targetFrame ?target ;
        hw1:qualificationStatus hw1:Pass .
  FILTER NOT EXISTS {
    ?experiment hw1:producesAnnotation ?sourceAnnotation .
    ?sourceAnnotation hw1:annotatesFrame ?source ;
                      hw1:qualificationStatus hw1:Fail .
  }
  FILTER NOT EXISTS {
    ?experiment hw1:producesAnnotation ?targetAnnotation .
    ?targetAnnotation hw1:annotatesFrame ?target ;
                      hw1:qualificationStatus hw1:Fail .
  }
}
ORDER BY ?pairIndex
""" % NS


def query_graph(graph, query_text):
    """Run a read-only SPARQL query against a local graph. Full strategy note: docs/triplestore.md."""
    try:
        return graph.query(query_text)
    except Exception as exc:
        raise ValueError(f"invalid or unsupported SPARQL query: {exc}") from None


def usable_links_from_query(graph, exp_iri):
    """Usable-link set from a SPARQL result. Full strategy note: docs/triplestore.md."""
    rows = query_graph(graph, _USABLE_LINKS_QUERY)
    selected = []
    for row in rows:
        # The query can see multiple experiments if a caller gives it a combined
        # graph. `read_experiment` supplies one, but keep this helper safe for
        # other API consumers by verifying the pair belongs to this experiment.
        pair = None
        for candidate in graph.subjects(HW1.pairIndex, row.pairIndex):
            if ((exp_iri, HW1.producesPair, candidate) in graph and
                    graph.value(candidate, HW1.sourceFrame) == row.source and
                    graph.value(candidate, HW1.targetFrame) == row.target):
                pair = candidate
                break
        if pair is not None:
            selected.append((int(row.pairIndex),
                             (frame_index_from_iri(row.source),
                              frame_index_from_iri(row.target))))
    selected.sort(key=lambda value: value[0])
    return [link for _, link in selected]


def _read_semantic_experiment(g, exp, b, path):
    """Read explicit Factor occurrences and expose the legacy policy adapter."""
    # Keep completeness semantics in one shared validator. The adapter fields
    # below remain for reconstruct.py compatibility, while the report is the
    # authoritative closed-world diagnostic for new-schema readers.
    from semantic_model import validate_experiment
    completion = validate_experiment(g, exp, materialize=False)
    frames = []
    for f in g.objects(b, HW1.hasFrame):
        idx = g.value(f, HW1.frameIndex)
        rgb = g.value(f, HW1.hasRGBImage); dep = g.value(f, HW1.hasDepthImage)
        if idx is None or rgb is None or dep is None:
            raise ValueError(f"{path}: every Batch Frame needs frameIndex, RGBImage and DepthImage")
        frames.append((int(idx), f, rgb, dep))
    frames.sort(key=lambda x: x[0])
    memberships = {x[2]: x[0] for x in frames} | {x[3]: x[0] for x in frames}
    selected = sorted({_local(x) for x in g.objects(exp, HW1.evaluatesFactor)})
    factors = list(g.subjects(HW1.inExperiment, exp))
    seen = set(); frame_status = {idx: True for idx, *_ in frames}; pair_status = {}; masks = {}
    for node in factors:
        typ = g.value(node, HW1.hasDefinition)
        if typ is None:
            typ = g.value(node, HW1.factorType)
        cur = g.value(node, HW1.hasCurrentFrame); prev = g.value(node, HW1.hasPrevious)
        if typ is None or cur not in memberships or (prev is not None and prev not in memberships):
            raise ValueError(f"{path}: Factor {node} has foreign or incomplete image links")
        state = g.value(node, HW1.evaluationState)
        if state == HW1.Measured and (g.value(node, HW1.value) is None or g.value(node, HW1.status) is None):
            raise ValueError(f"{path}: measured Factor {node} must have exactly one value and status")
        if prev is None:
            idx = memberships[cur]; frame_status[idx] = frame_status[idx] and g.value(node, HW1.status) == HW1.Pass
        else:
            key = (memberships[prev], memberships[cur]); st = g.value(node, HW1.status)
            pair_status[key] = pair_status.get(key, True) and st == HW1.Pass
        for mf in g.objects(node, HW1.maskFile):
            local = _local(typ); key = (memberships[prev], memberships[cur]) if prev is not None else memberships[cur]
            masks.setdefault(local, {"frames": {}, "pairs": {}})["pairs" if prev is not None else "frames"][key] = str(mf)
    ordered = sorted(pair_status)
    usable = [(i, j) for i, j in ordered if pair_status[(i, j)] and frame_status.get(i, True) and frame_status.get(j, True)]
    complete = completion.complete
    return {"exp_iri": URIRef(str(exp)), "exp_name": _experiment_name_from_iri(exp, path),
            "batch_name": _batch_name_from_batch_iri(b), "batch_iri": URIRef(str(b)),
            "selected": selected, "settings": _experiment_settings(g, exp, path),
            "frame_status": frame_status, "pair_status": pair_status,
            "usable_links": usable, "mask_files": masks, "complete": complete,
            "completion": completion.to_dict(), "graph": g}


def read_experiment(path):
    """Canonical reader: declaration, settings, values+statuses, usable-link conjunction. Full strategy note: docs/triplestore.md."""
    text = _read_text(path)
    student_text, machine_text = _split_sections(text)
    if machine_text is None:
        raise ValueError(
            f"{path}: this file carries no MACHINE SECTION marker, so it is a "
            f"DECLARATION that has not been assessed yet — there are no values and "
            f"no verdicts in it to read. Run `api.py experiment {path}` first "
            f"(`api.py explore {path}` shows what it declares in the meantime).")

    # ONE GRAPH, BOTH SECTIONS (§3.1). The machine section re-opens
    # the student's Experiment subject, so only the whole file carries the whole
    # experiment: the selection and the student's own settings are above the
    # marker, the measurements and the defaulted settings below it.
    g = Graph()
    g.parse(data=text, format="turtle")
    exp = _sole_experiment(g, path)
    _verify_seal(g, exp, student_text, path)

    b = g.value(exp, HW1.onBatch)
    if b is None:
        raise ValueError(f"{path}: experiment {exp} has no hw1:onBatch")

    if _graph_uses_semantic_schema(g, exp):
        return _read_semantic_experiment(g, exp, b, path)

    selected = sorted({_local(f) for f in g.objects(exp, HW1.evaluatesFactor)})
    settings = _experiment_settings(g, exp, path)

    # ── frame verdicts: the conjunction over that frame's annotations ─────────
    per_frame = {}
    mask_files = {}
    for ann in g.objects(exp, HW1.producesAnnotation):
        idx = _annotation_frame_index(g, ann, path)
        st = g.value(ann, HW1.qualificationStatus)
        if st is None:
            raise ValueError(
                f"{path}: hw1:FrameAnnotation {ann} has no hw1:qualificationStatus; "
                f"§4.3 makes the aggregate total on every annotation, so this file "
                f"was written by something that is not `api.py experiment`")
        per_frame[idx] = per_frame.get(idx, True) and (st == HW1.Pass)
        for mask_file in g.objects(ann, HW1.maskFile):
            factor = _mask_factor_from_path(mask_file, path)
            mask_files.setdefault(factor, {"frames": {}, "pairs": {}})[
                "frames"][idx] = str(mask_file)

    # ── pair verdicts, in capture order ───────────────────────────────────────
    ordered = []
    for pair in g.objects(exp, HW1.producesPair):
        src = g.value(pair, HW1.sourceFrame)
        tgt = g.value(pair, HW1.targetFrame)
        if src is None or tgt is None:
            raise ValueError(
                f"{path}: hw1:FramePair {pair} is missing hw1:sourceFrame or "
                f"hw1:targetFrame")
        st = g.value(pair, HW1.qualificationStatus)
        if st is None:
            raise ValueError(
                f"{path}: hw1:FramePair {pair} has no hw1:qualificationStatus; §4.3 "
                f"makes it total on every pair — vacuously hw1:Pass when no pair "
                f"factor was selected")
        pidx = g.value(pair, HW1.pairIndex)
        if pidx is None:
            raise ValueError(
                f"{path}: hw1:FramePair {pair} has no hw1:pairIndex, so nothing puts "
                f"it in capture order (§2: order by pairIndex, never by IRI string)")
        pair_key = (frame_index_from_iri(src), frame_index_from_iri(tgt))
        for mask_file in g.objects(pair, HW1.maskFile):
            factor = _mask_factor_from_path(mask_file, path)
            mask_files.setdefault(factor, {"frames": {}, "pairs": {}})[
                "pairs"][pair_key] = str(mask_file)
        ordered.append((int(pidx),
                        pair_key,
                        st == HW1.Pass))
    ordered.sort(key=lambda row: row[0])

    # A frame that appears only as a pair endpoint is VACUOUSLY usable: with no
    # annotation there is no claim about it to fail (§4.5). v2 raised here instead,
    # because v2 annotated every frame; under selection that would reject every
    # pair-only experiment.
    endpoints = {end for _, ends, _ in ordered for end in ends}
    frame_status = {idx: per_frame.get(idx, True)
                    for idx in sorted(set(per_frame) | endpoints)}

    pair_status = {}
    usable_links = []
    for _, (i, j), ok in ordered:
        pair_status[(i, j)] = ok
    # §4.5's usable-link rule is a SPARQL query shared with reconstruct.py.
    # Keep the validation and the derived status maps above: malformed input must
    # still fail loudly instead of merely disappearing from a query result.
    usable_links = usable_links_from_query(g, exp)

    return {"exp_iri": URIRef(str(exp)),
            "exp_name": _experiment_name_from_iri(exp, path),
            "batch_name": _batch_name_from_batch_iri(b),
            "batch_iri": URIRef(str(b)),
            "selected": selected,
            "settings": settings,
            "frame_status": frame_status,
            "pair_status": pair_status,
            "usable_links": usable_links,
            "mask_files": mask_files,
            "graph": g}


# The IRI `<mode>` segment of §2 pinned to the hw1:SelectionMode
# individual of §4.4. One dict, so "baseline" cannot end up meaning GoodSegments in
# one caller and FullBatch in another.
_SELECTION_MODE = {"baseline": HW1.FullBatch,
                   "selected": HW1.GoodSegments,
                   "masked": HW1.MaskFiltered}

# Statusless reconstruction-mechanism evidence (§14).  These are
# diagnostics, not quality factors: there is no threshold to game and no baked
# Pass/Fail cache.  Keep the allow-list here so a misspelling cannot silently mint
# an RDF predicate that no reader knows about.
_RUN_METADATA = {
    "gatedSteps": XSD.integer,
    "spliceCount": XSD.integer,
    "maxGapLength": XSD.integer,
}


def _write_machine_section(path, student_text, machine_graph):
    """Re-serialize only the machine half (declaration bytes preserved). Full strategy note: docs/triplestore.md."""
    body = machine_graph.serialize(format="turtle")
    if isinstance(body, bytes):                      # rdflib < 6 returned bytes
        body = body.decode("utf-8")
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(student_text)
        fh.write(MACHINE_MARKER + "\n\n")
        fh.write(body if body.endswith("\n") else body + "\n")
    os.replace(tmp, path)


def write_run(exp_path, mode, values, used_frames=None, frame_count=None,
              metadata=None, mask_factor=None, diagnostic_file=None):
    """THE one writer of hw1:ReconstructionRun triples (S8.2), idempotent per key. Full strategy note: docs/triplestore.md."""
    if mode not in _SELECTION_MODE:
        raise ValueError(
            f"mode must be one of {sorted(_SELECTION_MODE)}, got {mode!r}; the mode is "
            f"the last segment of the run IRI (§2) and is not free text")
    if used_frames is not None and mode not in ("selected", "masked"):
        raise ValueError(
            f"used_frames given with mode={mode!r}: hw1:usedFrame is asserted by "
            f"selected or masked subset runs only. A full-batch run states "
            f"its size with hw1:runFrameCount and lists no frames.")
    if mode == "masked" and not mask_factor:
        raise ValueError("mode='masked' requires mask_factor provenance")
    if mode != "masked" and mask_factor is not None:
        raise ValueError("mask_factor is legal only for mode='masked'")
    metadata = {} if metadata is None else dict(metadata)
    unknown_metadata = sorted(set(metadata) - set(_RUN_METADATA))
    if unknown_metadata:
        raise ValueError(
            f"unknown run metadata {unknown_metadata}; allowed statusless mechanism "
            f"properties: {', '.join(sorted(_RUN_METADATA))}")
    for key, value in metadata.items():
        if isinstance(value, bool) or int(value) != value or int(value) < 0:
            raise ValueError(f"run metadata {key} must be a non-negative integer, "
                             f"got {value!r}")

    text = _read_text(exp_path)
    student_text, machine_text = _split_sections(text)
    if machine_text is None:
        raise ValueError(
            f"{exp_path}: no machine section — a run cannot be written into a "
            f"declaration that was never assessed. Run `api.py experiment "
            f"{exp_path}` first.")

    # The WHOLE file answers "which experiment, under which thresholds" (the
    # student's own settings are above the marker); the MACHINE half is what gets
    # mutated and written back. Two parses of one small file, and no possibility of
    # grading against half a setting vector.
    full = Graph()
    full.parse(data=text, format="turtle")
    exp = _sole_experiment(full, exp_path)
    _verify_seal(full, exp, student_text, exp_path)
    expname = _experiment_name_from_iri(exp, exp_path)
    settings = _experiment_settings(full, exp, exp_path)
    # Runs live in the experiment's own namespace tier: v5 (semantic) experiments
    # mint under DATA_NS, legacy v4 ones under NS. Readers resolve runs through
    # hw1:hasRun and accept both prefixes, so this branch changes nothing they see.
    if _graph_uses_semantic_schema(full, exp):
        run = data_run_iri(expname, mode)
        legacy_run = run_iri(expname, mode)
    else:
        run = run_iri(expname, mode)
        legacy_run = None

    # The status PROPERTY comes from the factor's hw1:statusProperty, never from
    # string concatenation. The naming rule (value + "Status") is frozen, but a
    # concatenating writer would happily invent `hw1:somethingStatus` for a key no
    # factor grades, and the resulting triple would be invisible to every reader —
    # they all find statuses through `?factor hw1:statusProperty ?p`.
    factors = load_quality_factors()
    if mask_factor is not None:
        selected = {_local(f) for f in full.objects(exp, HW1.evaluatesFactor)}
        if mask_factor not in selected:
            raise ValueError(
                f"mask_factor {mask_factor!r} was not selected by this experiment")
    try:
        status_property = {f["over"]: f["status"] for f in factors.values()}
    except KeyError as exc:
        raise ValueError(
            f"load_quality_factors() must return the §8 shape "
            f"{{'over', 'status', 'polarity', 'qualifiedBy'}}; key {exc} is missing"
        ) from None

    # Resolve EVERY key, and every status, before touching the graph: a typo in the
    # second key must not leave the first one half-written into the file.
    planned = []
    for key in values:
        if key not in status_property:
            raise ValueError(
                f"{key!r} is not graded by any hw1:QualityFactor; write_run writes a "
                f"value together with its status, so a value nothing grades has no "
                f"business in a run node. Graded value properties: "
                f"{', '.join(sorted(status_property))}")
        # ROUND FIRST, GRADE SECOND (§4.5). `_storable` is the value as
        # the .ttl will actually hold it; grading the unrounded argument would bake a
        # verdict the stored number re-grades the other way. mapMeanL2 = 0.4000000001
        # against maxMapMeanL2 = 0.4 is the case: it stores as `4e-01`, so a `Fail`
        # graded from the raw double sits next to a number that reads Pass.
        stored = _storable(values[key])
        planned.append((HW1[key], HW1[status_property[key]], stored,
                        status_for(key, stored, settings, factors)))

    # hw1:usedFrame points at the SHARED structural frames of the batch (§2), not at
    # anything experiment-scoped, so the frame IRIs are minted from the batch name —
    # which is why an experiment without hw1:onBatch cannot record provenance at all.
    frames = None
    if used_frames is not None:
        b = full.value(exp, HW1.onBatch)
        if b is None:
            raise ValueError(
                f"{exp_path}: experiment {exp} has no hw1:onBatch, so the frame IRIs "
                f"for hw1:usedFrame cannot be minted")
        name = _batch_name_from_batch_iri(b)
        mint_frame = (data_frame_iri if legacy_run is not None else frame_iri)
        frames = [mint_frame(name, int(idx)) for idx in used_frames]

    # Everything a run node consists of lives BELOW the marker, so the mutation
    # happens on the machine section alone and the declaration is never re-serialized.
    g = Graph()
    g.parse(data=machine_text, format="turtle")

    # ── remove exactly what is about to be written, and nothing else ──────────
    # On semantic experiments also clear a legacy-NS run node of the same mode:
    # runs written before the DATA_NS migration would otherwise linger as a stale
    # second node reachable through hw1:hasRun.
    runs_to_clear = (run,) if legacy_run is None else (run, legacy_run)
    for old_run in runs_to_clear:
        for value_p, status_p, _, _ in planned:
            g.remove((old_run, value_p, None))
            g.remove((old_run, status_p, None))
        if frames is not None:
            g.remove((old_run, HW1.usedFrame, None))
        if frame_count is not None:
            g.remove((old_run, HW1.runFrameCount, None))
        if mask_factor is not None:
            g.remove((old_run, HW1.maskFactor, None))
        if diagnostic_file is not None:
            g.remove((old_run, HW1.diagnosticFile, None))
        for key in metadata:
            g.remove((old_run, HW1[key], None))
        g.remove((old_run, HW1.selectionMode, None))
        if old_run is not run:
            g.remove((old_run, RDF.type, None))
            g.remove((exp, HW1.hasRun, old_run))

    # ── write ─────────────────────────────────────────────────────────────────
    g.add((run, RDF.type, HW1.ReconstructionRun))
    g.add((run, HW1.selectionMode, _SELECTION_MODE[mode]))
    g.add((exp, HW1.hasRun, run))
    for value_p, status_p, v, status in planned:
        # `_double_literal` is the one spelling of an xsd:double in this project, and
        # it rounds through `_storable` too — so the number written here is, byte for
        # byte, the number `status` was computed from.
        g.add((run, value_p, _double_literal(v)))
        g.add((run, status_p, status))
    if frame_count is not None:
        g.add((run, HW1.runFrameCount, Literal(int(frame_count), datatype=XSD.integer)))
    if frames is not None:
        for f in frames:
            g.add((run, HW1.usedFrame, f))
    if mask_factor is not None:
        g.add((run, HW1.maskFactor, HW1[mask_factor]))
    if diagnostic_file is not None:
        g.add((run, HW1.diagnosticFile, Literal(str(diagnostic_file))))
    for key, value in metadata.items():
        g.add((run, HW1[key], Literal(int(value), datatype=_RUN_METADATA[key])))

    _write_machine_section(exp_path, student_text, g)


def write_pair_measurements(exp_path, factor_local, measurements):
    """Write back PriorWarpDepthResidual pair evidence below the marker. Full strategy note: docs/triplestore.md."""
    factors = load_quality_factors()
    if factor_local not in factors:
        raise ValueError(f"unknown QualityFactor {factor_local!r}")
    value_local = factors[factor_local]["over"]
    spec = _PAIR_OBSERVABLES.get(value_local)
    if spec is None or not spec.get("deferred"):
        raise ValueError(
            f"{factor_local} is not a deferred pair factor; experiment measures it")

    text = _read_text(exp_path)
    student_text, machine_text = _split_sections(text)
    if machine_text is None:
        raise ValueError(f"{exp_path}: cannot write pair evidence before assessment")
    full = Graph()
    full.parse(data=text, format="turtle")
    exp = _sole_experiment(full, exp_path)
    _verify_seal(full, exp, student_text, exp_path)
    selected = {_local(f) for f in full.objects(exp, HW1.evaluatesFactor)}
    if factor_local not in selected:
        raise ValueError(
            f"{exp_path}: cannot write {factor_local}; it was not selected")
    settings = _experiment_settings(full, exp, exp_path)

    # Explicit-occurrence schema: update only returned canonical keys.  Absent
    # evidence remains Pending, never an invented infinity/Measured result.
    if _graph_uses_semantic_schema(full, exp):
        machine = Graph(); machine.parse(data=machine_text, format="turtle")
        factors = load_quality_factors(); by_pair = {
            (int(m["source"]), int(m["target"])): dict(m) for m in measurements}
        info = factors[factor_local]; over = info.get("over", factor_local)
        names = {"HighFrequencyDepthResidual": "highFrequencyDepthResidual", "FlyingPixelRatio": "flyingPixelRatio",
                 "ValidTileCoverage": "validTileCoverage", "IdentityMedianDepthChange": "identityMedianDepthChange",
                 "JointValidDepthRatio": "jointValidDepthRatio", "PriorWarpDepthResidual": "priorWarpDepthResidual"}
        over = names.get(factor_local, over)
        image_index = {}
        for frame in machine.objects(full.value(exp, HW1.onBatch), HW1.hasFrame):
            idx = full.value(frame, HW1.frameIndex)
            for image in (full.value(frame, HW1.hasDepthImage), full.value(frame, HW1.hasRGBImage)):
                if idx is not None and image is not None: image_index[image] = int(idx)
        written = 0
        for node in list(machine.subjects(HW1.inExperiment, exp)):
            node_def = machine.value(node, HW1.hasDefinition)
            if node_def is None:
                node_def = machine.value(node, HW1.factorType)
            if _local(node_def) != factor_local:
                continue
            prev, cur = machine.value(node, HW1.hasPrevious), machine.value(node, HW1.hasCurrentFrame)
            if prev is None or cur is None:
                continue
            if prev not in image_index or cur not in image_index:
                raise ValueError(f"{exp_path}: deferred Factor {node} points outside Batch structure")
            i, j = image_index[prev], image_index[cur]
            m = by_pair.get((i, j)); machine.remove((node, HW1.value, None)); machine.remove((node, HW1.status, None)); machine.remove((node, HW1.evaluationState, None))
            if m is None:
                machine.add((node, HW1.evaluationState, HW1.Pending)); continue
            value = _storable(float(m.get("value", float("nan"))))
            machine.add((node, HW1.value, _double_literal(value))); machine.add((node, HW1.status, status_for(over, value, settings, factors))); machine.add((node, HW1.evaluationState, HW1.Measured)); written += 1
            if "count" in m: machine.remove((node, HW1.supportCount, None)); machine.add((node, HW1.supportCount, Literal(int(m["count"]), datatype=XSD.integer)))
        # Recompute completion from explicit state; preserve declaration bytes.
        machine.remove((exp, RDF.type, HW1.FullEvaluatedFrames))
        occurrences = list(machine.subjects(HW1.inExperiment, exp))
        if occurrences and all(machine.value(n, HW1.evaluationState) == HW1.Measured for n in occurrences): machine.add((exp, RDF.type, HW1.FullEvaluatedFrames))
        _write_machine_section(exp_path, student_text, machine)
        return written

    by_pair = {(int(m["source"]), int(m["target"])): dict(m)
               for m in measurements}
    g = Graph()
    g.parse(data=machine_text, format="turtle")
    expname = _experiment_name_from_iri(exp, exp_path)
    artifact_root = os.path.splitext(os.path.abspath(exp_path))[0]
    relative_to = os.path.dirname(os.path.abspath(exp_path))
    count_property = spec.get("count_property")

    written = 0
    for pair in g.objects(exp, HW1.producesPair):
        src = g.value(pair, HW1.sourceFrame)
        tgt = g.value(pair, HW1.targetFrame)
        key = (frame_index_from_iri(src), frame_index_from_iri(tgt))
        measurement = by_pair.get(key)
        value = (float("inf") if measurement is None
                 else float(measurement.get("value", float("inf"))))
        count = 0 if measurement is None else int(measurement.get("count", 0))

        g.remove((pair, HW1[value_local], None))
        g.remove((pair, HW1[factors[factor_local]["status"]], None))
        if count_property:
            g.remove((pair, HW1[count_property], None))
        _write_observable(g, pair, value_local, value, settings, factors)
        if count_property:
            g.add((pair, HW1[count_property], Literal(count, datatype=XSD.integer)))

        # Remove only this factor's prior artifact reference; other selected
        # factor masks share hw1:maskFile on the same pair node.
        for old in list(g.objects(pair, HW1.maskFile)):
            if _mask_factor_from_path(old, exp_path) == factor_local:
                g.remove((pair, HW1.maskFile, old))
        mask = None if measurement is None else measurement.get("mask")
        mask_file = _write_mask_artifact(
            mask, artifact_root, relative_to, factor_local,
            f"{key[0]}_{key[1]}.png")
        if mask_file is not None:
            g.add((pair, HW1.maskFile, Literal(mask_file)))
        written += 1

    selected_pair_values = [
        factors[name]["over"] for name in selected
        if name in factors and factors[name]["over"] in _PAIR_OBSERVABLES]
    for pair in g.objects(exp, HW1.producesPair):
        passed = []
        for over in selected_pair_values:
            status = g.value(pair, _status_property(over, factors))
            passed.append(status == HW1.Pass if status is not None else False)
        g.set((pair, HW1.qualificationStatus,
               HW1.Pass if all(passed) else HW1.Fail))

    _write_machine_section(exp_path, student_text, g)
    return written


# =============================================================================
# batch2ttl  — the STRUCTURE of one capture, and not one measured number
# =============================================================================
def _resolve_generation_settings(gen_args, decls):
    """Generation sidecar settings for a batch. Full strategy note: docs/triplestore.md."""
    given = {}
    for arg in gen_args or []:
        name, value = parse_setting_arg(arg, decls)
        role = decls[name]["role"]
        if role != "GenerationSetting":
            gen_names = sorted(n for n in decls if decls[n]["role"] == "GenerationSetting")
            raise ValueError(
                f"--gen {name}=... : {name!r} is declared hw1:paramRole hw1:{role}, not "
                f"hw1:GenerationSetting. A Generation setting describes the PIXELS and "
                f"lives on the Batch; Measurement and Qualification settings describe a "
                f"measurement pass and live in the DECLARATION, as hw1:FactorSetting "
                f"nodes on the experiment. "
                f"Generation parameters are: {', '.join(gen_names) or '(none)'}")
        if name in given:
            raise ValueError(
                f"--gen {name!r} given twice ({given[name]!r} then {value!r}). One "
                f"capture directory was produced under one level of each generation "
                f"parameter; two levels are two captures.")
        given[name] = value
    return given


def build_batch_graph(data_dir, floor, generation=None, derived_from=None):
    """Batch/file structural graph (frames, images, gaps warned). Full strategy note: docs/triplestore.md."""
    g = Graph()
    name = batch_name(data_dir, floor)
    b = batch_iri(name)

    g.add((b, RDF.type, HW1.Batch))
    g.add((b, HW1.batchName, Literal(name)))
    g.add((b, HW1.batchPath, Literal(data_dir)))
    g.add((b, HW1.floor, Literal(int(floor), datatype=XSD.integer)))

    # Corruption provenance, on the BATCH because it describes the pixels and
    # survives every re-measurement of them (§4.2). Applied by nothing
    # here: the capture directory already embodies it, and pretending otherwise is
    # what the role exists to prevent.
    decls = load_parameter_declarations(_ONTOLOGY_TTL)
    for param in sorted(generation or {}):
        decl = decls[param]
        s = generation_setting_iri(name, param)
        g.add((b, HW1.hasGenerationSetting, s))
        g.add((s, RDF.type, HW1.FactorSetting))
        g.add((s, HW1.settingParameter, decl["iri"]))
        # The role is DENORMALISED onto the setting node on purpose: the attribution
        # query then reads the verdict ("regenerate the data") off the node it
        # already has, without a second join into the TBox.
        g.add((s, HW1.settingRole, HW1[decl["role"]]))
        # settingForFactor is SINGLE-VALUED — the primary factor only, here exactly
        # as on experiment-scoped settings (§4.5). v2 wrote the affected
        # factors out too, so that attribution could blame brightnessGain for CRUSHED
        # shadows and not only for blown highlights — the failure a low_light batch
        # actually produces. v3 keeps that blame and drops the duplication: `explore`
        # reaches the affected factors by traversing hw1:paramAffectsFactor in the
        # TBox, which is the one edge that was being copied here.
        g.add((s, HW1.settingForFactor, decl["primaryIri"]))
        g.add((s, HW1.settingValue, _setting_value_literal(generation[param], decl["kind"])))

    # prov:wasDerivedFrom BETWEEN BATCHES is the only surviving prov: term
    # (§6): "these pixels came from those pixels, corrupted". It is NOT
    # asserted between experiments — there are no derived experiments in v2.
    if derived_from:
        g.add((b, PROV.wasDerivedFrom, batch_iri(derived_from)))

    frames = _pair_frames(data_dir)
    for stem, rgb_path, depth_path in frames:
        f = frame_iri(name, stem)
        g.add((b, HW1.hasFrame, f))
        g.add((f, RDF.type, HW1.Frame))
        g.add((f, HW1.frameIndex, Literal(int(stem), datatype=XSD.integer)))

        # An image node carries schema:contentUrl and NOTHING ELSE in v2
        # (§4.3): the observables moved to the experiment-scoped
        # FrameAnnotation, because two experiments cannot both own one image node.
        rc = component_iri(name, stem, "rgb")
        g.add((f, HW1.hasRGBImage, rc))
        g.add((rc, RDF.type, HW1.RGBImage))
        g.add((rc, SCHEMA.contentUrl, Literal(rgb_path)))

        dc = component_iri(name, stem, "depth")
        g.add((f, HW1.hasDepthImage, dc))
        g.add((dc, RDF.type, HW1.DepthImage))
        g.add((dc, SCHEMA.contentUrl, Literal(depth_path)))

    _warn_stem_gaps(frames, prefix="[batch2ttl]")
    return g, name, b


def _warn_stem_gaps(frames, prefix):
    """Log consecutive-stem holes in a paired capture. Shared by declare/batch2ttl."""
    gaps = [(int(s0), int(s1))
            for (s0, _, _), (s1, _, _) in zip(frames, frames[1:])
            if int(s1) - int(s0) != 1]
    if not gaps:
        return gaps
    shown = ", ".join(f"{i}->{j}" for i, j in gaps[:12])
    more = f" ... (+{len(gaps) - 12} more)" if len(gaps) > 12 else ""
    print(f"{prefix} WARNING: {len(gaps)} stem gap(s) in the paired sequence: "
          f"{shown}{more}")
    print(f"{prefix} WARNING: every experiment over this batch will pair across "
          f"those gaps, spanning more than one capture step. A stem missing from "
          f"rgb/ or depth/ changes which pairs exist, and the pair observables will "
          f"read as larger motion — correctly, but check it is the capture and not a "
          f"lost file.")
    return gaps


def cmd_batch2ttl(args):
    """Optional generation-provenance sidecar batch.ttl (deprecated as a required step). Full strategy note: docs/triplestore.md."""
    print("[batch2ttl] DEPRECATED: `declare` and `experiment` take the capture "
          "directory directly (`declare --data-dir <dir> --floor <n>`). This "
          "command is now only the optional writer of generation provenance into "
          "<data_dir>/batch.ttl.", file=sys.stderr)
    decls = load_parameter_declarations(_ONTOLOGY_TTL)
    generation = _resolve_generation_settings(getattr(args, "gen", None), decls)

    g, name, _ = build_batch_graph(args.data_dir, args.floor,
                                   generation=generation,
                                   derived_from=args.derived_from)
    out = args.out or os.path.join(args.data_dir, "batch.ttl")
    g.serialize(destination=out, format="turtle")
    n_frames = len(set(g.subjects(RDF.type, HW1.Frame)))
    print(f"[batch2ttl] batch {name!r}: {n_frames} frames, 0 measured values and 0 "
          f"pairs (by contract — pairs are minted per experiment)")
    if generation:
        print(f"[batch2ttl] generation settings recorded (provenance only, applied by "
              f"nothing): "
              f"{', '.join(f'{k}={generation[k]!r}' for k in sorted(generation))}")
    else:
        print(f"[batch2ttl] generation settings recorded: (none). Absence means NOT "
              f"ASSERTED, not 'uncorrupted' — pass --gen NAME=VALUE for a corrupted "
              f"capture, because nothing defaults a claim about pixels.")
    if args.derived_from:
        print(f"[batch2ttl] prov:wasDerivedFrom {batch_iri(args.derived_from)}")
    print(f"[batch2ttl] wrote {len(g)} triples -> {out}")
    return 0



# =============================================================================
# declare  — scaffold a declaration Turtle so nobody starts from a blank page
#
#   The declaration is still the one piece of RDF the student AUTHORS
#   (§4.2): this command only writes the boilerplate — prefixes, the
#   Experiment node whose IRI tail equals the file stem, the batch join
#   (hw1:batchFile = the capture directory, hw1:onBatch derived from --floor +
#   basename), a factor selection, and the PREDICTION block as comments.
#   Everything it writes is the student's to edit until `api.py experiment`
#   seals the file; it never assesses and never overwrites an existing
#   declaration.
# =============================================================================
def _factor_menu_comment(factors, decls):
    """Factor menu comment block for declarations."""
    menu = _selectable_factors(factors)
    width = max(len(name) for name in menu)
    lines = []
    for name in sorted(menu):
        info = factors[name]
        op = "<=" if info["polarity"] == "lower" else ">="
        default = decls[info["qualifiedBy"]]["default"]
        lines.append(f"#   hw1:{name:<{width}}  Pass iff hw1:{info['over']} "
                     f"{op} hw1:{info['qualifiedBy']} (default {default})")
    return lines


def _declare_capture_args(args):
    """Capture args recorded in a declaration."""
    data_dir_arg = getattr(args, "data_dir", None)
    batch_file_arg = getattr(args, "batch_file", None)
    floor = int(getattr(args, "floor", 1) or 1)
    if data_dir_arg and batch_file_arg:
        raise SystemExit(
            "[declare] pass --data-dir (the capture directory) or the deprecated "
            "--batch-file, not both.")
    if data_dir_arg:
        resolved = _resolve_batch_file(data_dir_arg)
        if not _is_capture_dir(resolved):
            raise SystemExit(
                f"[declare] --data-dir {data_dir_arg!r} does not resolve to a capture "
                f"directory with rgb/ and depth/ subdirs ({resolved!r}; relative "
                f"paths resolve against the current working directory).")
        return resolved, floor, data_dir_arg.replace(os.sep, "/")
    if not batch_file_arg:
        raise SystemExit(
            "[declare] --data-dir is required (the capture directory containing "
            "rgb/ and depth/). `batch2ttl` is no longer a step in this loop.")
    print("[declare] --batch-file is deprecated: pass --data-dir <capture-dir> "
          "instead. The scaffold will name the capture directory in hw1:batchFile.",
          file=sys.stderr)
    resolved = _resolve_batch_file(batch_file_arg)
    if _is_capture_dir(resolved):
        return resolved, floor, batch_file_arg.replace(os.sep, "/")
    if os.path.isfile(resolved):
        _g, _batch, _name, stored_path, stored_floor = _parse_batch_ttl(resolved)
        data_dir = _capture_dir(stored_path, batch_file_arg)
        if stored_floor is not None:
            floor = stored_floor
        # Write the capture directory into the declaration, not the sidecar.
        return data_dir, floor, os.path.relpath(data_dir, os.getcwd()).replace(os.sep, "/")
    raise SystemExit(
        f"[declare] --batch-file {batch_file_arg!r} does not resolve to a capture "
        f"directory or a batch.ttl ({resolved!r}). Pass --data-dir <capture-dir>.")


def cmd_declare(args):
    """Scaffold a student declaration Turtle (never assesses, never overwrites). Full strategy note: docs/triplestore.md."""
    name = args.name
    if not _EXPNAME_RE.match(name):
        raise SystemExit(
            f"[declare] {name!r} is not a legal experiment name (§2: "
            f"[A-Za-z0-9_-]+). It becomes both the file stem and the tail of the "
            f"Experiment IRI, which must stay equal.")

    out = args.out or os.path.join(_EXPERIMENT_DIR, f"{name}.ttl")
    stem = os.path.splitext(os.path.basename(out))[0]
    if stem != name:
        raise SystemExit(
            f"[declare] --out names the file {stem!r}.ttl but --name says {name!r}. "
            f"The file stem must equal the Experiment IRI tail (§2), so "
            f"the two flags must agree.")
    if os.path.exists(out):
        raise SystemExit(
            f"[declare] {out} already exists and hw1/experiments/ is append-only "
            f"(§3.1): a new tuning idea is a NEW declaration under a "
            f"new name, and an existing file — assessed or not — is never "
            f"regenerated over.")

    data_dir, floor, batch_file_text = _declare_capture_args(args)
    try:
        frames = _pair_frames(data_dir)
    except ValueError as exc:
        raise SystemExit(f"[declare] {exc}")
    _warn_stem_gaps(frames, prefix="[declare]")
    name_of_batch = batch_name(data_dir, floor)
    batch = data_batch_iri(name_of_batch)

    decls = load_parameter_declarations(_ONTOLOGY_TTL)
    factors = load_quality_factors(_ONTOLOGY_TTL)
    menu = _selectable_factors(factors)
    if args.factor:
        selected = []
        for raw in args.factor:
            local = raw[len("hw1:"):] if raw.startswith("hw1:") else raw
            if local not in menu:
                raise SystemExit(
                    f"[declare] hw1:{local} is not a factor of the menu. The menu is "
                    f"{', '.join('hw1:' + f for f in sorted(menu))} "
                    f"(§4.2); factor names come from the TBox, not the command line.")
            if local not in selected:
                selected.append(local)
        selected.sort()
    else:
        # The full menu: a legal 8-factor selection that assesses as-is. The
        # printed hint (and the TODO in the file) says to trim it — choosing the
        # selection is part of DESIGNING the experiment, not part of the scaffold.
        selected = sorted(menu)

    selection_text = " ,\n        ".join(f"hw1:{f}" for f in selected)
    lines = [
        "# =============================================================================",
        f"# EXPERIMENT DECLARATION — {name}   (scaffolded by `api.py declare`)",
        "#",
        "# This file is YOURS until `api.py experiment` seals it: edit the label, trim",
        "# the factor selection, add overrides. After assessment the student section is",
        "# digest-sealed — a new idea is a NEW file under a NEW name (§3.1).",
        "#",
        "# PREDICTION (write BEFORE assessing — the seal makes it non-retractable):",
        "#   input:    TODO — which frames/factors you expect to fail, and why",
        "#   baseline: TODO — expected mapMeanL2 band and verdict for the full-batch run",
        "#   selected: TODO — expected differential vs baseline, and the mechanism",
        "# =============================================================================",
        "",
        "@prefix hw1:  <http://taica.course/hw1/ontology#> .",
        f"@prefix batch: <{DATA_NS}batch/> .",
        f"@prefix exp:   <{DATA_NS}experiment/> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .",
        "",
        f"exp:{name}",
        "    a hw1:Experiment ;",
        "    hw1:schemaVersion \"5.0.0\" ;",
        f"    rdfs:label \"TODO: one line naming the condition this experiment tests\"@en ;",
        f"    hw1:batchFile \"{batch_file_text}\" ;",
        f"    hw1:onBatch batch:{name_of_batch} ;   # {name_of_batch}",
        "    hw1:evaluatesFactor",
        f"        {selection_text} .",
        "",
        "# THE FACTOR MENU (from ontology/hw1.ttl — select 1..8 above; TODO: trim the",
        "# selection to the factors your hypothesis actually needs):",
        *_factor_menu_comment(factors, decls),
        "#",
        "# OVERRIDES — every setting you do not declare is filled from its TBox default",
        "# at assessment time (preview with `explore` before committing). To override",
        "# one, replace the final \".\" above with \";\" and append a blank node, e.g.:",
        "#",
        "#     hw1:hasFactorSetting [",
        "#         a hw1:FactorSetting ;",
        "#         hw1:settingParameter hw1:maxHighFrequencyDepthResidual ;",
        "#         hw1:settingRole hw1:QualificationSetting ;",
        "#         hw1:settingForFactor hw1:HighFrequencyDepthResidual ;",
        "#         hw1:settingValue \"0.010\"^^xsd:double",
        "#     ] .",
        "#",
        "# hw1:settingRole must match the parameter's TBox hw1:paramRole:",
        "#   MeasurementSetting    how the observable is computed (e.g. hw1:tauHi)",
        "#   QualificationSetting  where the Pass line is cut (e.g. hw1:maxClipHiFraction)",
        "# Run-level re-cut: hw1:settingParameter hw1:maxMapMeanL2 with",
        "# hw1:settingForFactor hw1:ReconstructionAccuracy.",
        "",
    ]

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # The generator's own output must pass the validator it will later be read
    # by; a scaffold this command cannot re-read is a bug here, not user error.
    try:
        decl = read_declaration(out)
    except ValueError as exc:
        os.remove(out)
        raise SystemExit(
            f"[declare] scaffold failed its own validation and was removed — this "
            f"is a bug in `declare`, not in your input: {exc}")

    defaulted = sorted(set(decl["required"]) - set(decl["given"]))
    print(f"[declare] wrote {out}")
    print(f"[declare] selection ({len(selected)}/{len(menu)}): "
          + ", ".join(f"hw1:{f}" for f in selected)
          + ("" if args.factor else "  — the FULL menu; trim it to your hypothesis"))
    print(f"[declare] {len(defaulted)} setting(s) WILL BE DEFAULTED at assessment; "
          f"preview them with:  api.py explore {out}")
    print(f"[declare] before assessing: replace the PREDICTION TODOs and the "
          f"rdfs:label — the seal makes them non-retractable. Then:  "
          f"api.py experiment {out}")
    return 0


# =============================================================================
# experiment  — ONE declaration in, ONE machine section out, ONCE
#
#   This is the command the whole layout exists for, and in v3 it is the command
#   that reads what a STUDENT wrote. The declaration (§4.2) names a
#   capture directory and says which frames to grade, on which factors, under
#   which thresholds; this command measures exactly that from the rasters on
#   disk and appends the numbers, the verdicts, the defaulted settings and the
#   tamper seal to the declaration's OWN FILE, below the marker (§3.1/§4.3).
#
#   WRITE-ONCE. A file that already carries `MACHINE_MARKER` is a hard error and
#   there is no `--force`: an experiment is one treatment, and the way to try
#   another threshold is to copy the declaration to a new name. That is what turns
#   `hw1/experiments/` into an append-only lab notebook whose series of files IS
#   the report, and it restores v2's "nothing is ever overwritten" guarantee by
#   immutability instead of by digest identity (§6).
#
#   EVERYTHING v2 CARRIED ON THE COMMAND LINE IS DELETED (§7): `--batch-dir`,
#   `--floor`, `--set`, `--exp-id`, `--label`, `--no-pairs`, `--out`. The command
#   takes exactly one positional argument, the declaration path; the batch comes
#   from `hw1:batchFile`, the settings are FactorSettings, the label is
#   `rdfs:label`, and pairs are always minted.
# =============================================================================

# How often the measuring loops print a progress line. 387 frames x up to 5
# observables plus 386 pairs x 2 is well under a minute but far too long to look
# alive, and a command that prints nothing for forty seconds gets killed by the
# person running it. Progress goes to STDERR, not stdout: it is a status display,
# not a result, and `api.py experiment ... > log` must keep the summary readable.
_PROGRESS_EVERY = 25

# WHICH OBSERVABLE GOES ON WHICH NODE, AND WHAT MEASURES IT (§4.3,
# frozen). Two tables, read three ways:
#
#   * PLACEMENT. `clipHiFraction` / `clipLoFraction` are rgb-annotation
#     properties, the three active depth-frame observables are depth-annotation
#     properties, and the three pair observables live on the FramePair. That placement is
#     structural rather than cosmetic: an annotation exists per frame PER
#     MODALITY, and a modality with no selected factor gets no node at all.
#   * MEASUREMENT. `params` is what the measurer needs out of the setting vector
#     — checked against the vector before a single PNG is opened — and `measure`
#     is the one call site of each measurer in this file.
#   * THE MENU. `_selectable_factors` derives the eight-factor menu of §4.2 from
#     these keys: a QualityFactor whose `hw1:overProperty` appears here is
#     selectable; one whose does not (mapMeanL2, coverageF) is a RUN factor,
#     never selected and always evaluated. Adding a menu factor is therefore
#     adding a TBox declaration and one row here, and no per-factor branch
#     anywhere else.
#
# `hw1:meanValue` is deliberately absent: no QualityFactor is declared over it, it
# carries no status and it takes no part in any aggregate. It is written beside
# the clip factors as the baseline they must beat (§4.3), and putting it in this
# table would quietly promote the deliberately-weak baseline to a criterion.
_FRAME_OBSERVABLES = {
    "highFrequencyDepthResidual": {
        "modality": "depth",
        "params": ("residualMaskK",),
        "measure_mask": lambda rgb, depth, s: _high_frequency_depth_residual(
            depth, float(s["residualMaskK"]))},
    "flyingPixelRatio": {
        "modality": "depth",
        "params": ("flyingPixelWindow", "flyingPixelPlanarityTol"),
        "measure_mask": lambda rgb, depth, s: _flying_pixel_ratio(
            depth, int(s["flyingPixelWindow"]),
            float(s["flyingPixelPlanarityTol"]))},
    "validTileCoverage": {
        "modality": "depth",
        "params": ("tileSize", "tileValidFloor"),
        "measure_mask": lambda rgb, depth, s: _valid_tile_coverage(
            depth, int(s["tileSize"]), float(s["tileValidFloor"]))},
    "clipHiFraction": {
        "modality": "rgb",
        "params": ("tauHi",),
        "measure": lambda rgb, depth, s: frame_clip_hi_fraction(rgb, float(s["tauHi"]))},
    "clipLoFraction": {
        "modality": "rgb",
        "params": ("tauLo",),
        "measure": lambda rgb, depth, s: frame_clip_lo_fraction(rgb, float(s["tauLo"]))},
}

_PAIR_OBSERVABLES = {
    "identityMedianDepthChange": {
        "params": ("changeMaskK",),
        "count_property": "identityMedianDepthChangePixelCount",
        "measure_mask": lambda d0, d1, s: _identity_median_depth_change(
            d0, d1, float(s["changeMaskK"]))},
    "jointValidDepthRatio": {
        "params": (),
        "count_property": "jointValidDepthPixelCount",
        "measure_mask": lambda d0, d1, s: _joint_valid_depth_ratio(d0, d1)},
    # This Tier-B factor is measured from the actual constant-velocity state by
    # utils.reconstruct before fitting the current link.  experiment still puts
    # its settings in the vector and mints the pair; write_pair_measurements adds
    # the value/status/count/mask after the baseline consumer run.
    "priorWarpDepthResidual": {
        "params": ("priorWarpDepthGate",),
        "count_property": "priorWarpDepthResidualPixelCount",
        "deferred": True},
}

# The declaration whitelist of §4.2, as two tuples. Anything else in
# the student section — an annotation, a status, an observable value, a run — is a
# hard error, because VERDICTS ARE COMPUTED, NEVER DECLARED. A file that could
# assert its own `hw1:qualificationStatus` would let a student write the answer
# they wanted next to the data that disagrees with it, and nothing downstream
# could tell that apart from a measurement.
_DECLARATION_EXPERIMENT_PREDICATES = (
    RDF.type, RDFS.label, HW1.schemaVersion, HW1.batchFile, HW1.onBatch, HW1.evaluatesFactor,
    HW1.hasFactorSetting)
_DECLARATION_SETTING_PREDICATES = (
    RDF.type, HW1.settingParameter, HW1.settingRole, HW1.settingForFactor,
    HW1.settingValue)


def _fmt_term(term):
    """One RDF term as it would be written in the declaration. For error messages."""
    text = str(term)
    if isinstance(term, URIRef):
        return f"hw1:{text[len(NS):]}" if text.startswith(NS) else f"<{text}>"
    try:
        return term.n3()
    except Exception:                                    # pragma: no cover
        return repr(text)


def _fmt_triple(s, p, o):
    """`<s> <p> <o> .` in prefixed form — the offending triple every §4.2 error names."""
    return f"{_fmt_term(s)} {_fmt_term(p)} {_fmt_term(o)} ."


def _selectable_factors(factors):
    """The 1..8 selectable input factors."""
    placed = set(_FRAME_OBSERVABLES) | set(_PAIR_OBSERVABLES)
    return {name: info["over"] for name, info in factors.items()
            if info["over"] in placed}


def _run_level_factors(factors):
    """Non-selectable always-on run factors."""
    run_observables = {"mapMeanL2", "coverageF"}
    return {name: info["over"] for name, info in factors.items()
            if info["over"] in run_observables}


def _run_factor_parameters(decls, factors):
    """Run-factor parameters (icpBackend, maxMapMeanL2, ...). Full strategy note: docs/triplestore.md."""
    run_factors = set(_run_level_factors(factors))
    return sorted(name for name, d in decls.items()
                  if d["role"] in _EXPERIMENT_ROLES and d["primary"] in run_factors)


def _required_parameters(selected, decls, factors):
    """Selection-scoped required parameter set (S4.2 completeness). Full strategy note: docs/triplestore.md."""
    required = set(_run_factor_parameters(decls, factors))
    for factor in selected:
        required.add(factors[factor]["qualifiedBy"])
    for name, d in decls.items():
        if d["role"] != "MeasurementSetting":
            continue
        if set((d["primary"],) + tuple(d["affects"])) & set(selected):
            required.add(name)
    return required


def _resolve_batch_file(batch_file):
    """Resolve hw1:batchFile to a readable capture. Full strategy note: docs/triplestore.md."""
    return batch_file if os.path.isabs(batch_file) else os.path.abspath(batch_file)


def _is_capture_dir(path):
    """True iff `path` is a capture directory: rgb/ and depth/ subdirs exist."""
    return (os.path.isdir(path)
            and os.path.isdir(os.path.join(path, "rgb"))
            and os.path.isdir(os.path.join(path, "depth")))


_FLOOR_BATCH_NAME_RE = re.compile(r"^floor(\d+)_(.+)$")


def _floor_from_batch_name(name):
    """Floor number from a floor-qualified batch name."""
    match = _FLOOR_BATCH_NAME_RE.match(name)
    if not match:
        raise ValueError(
            f"batch name {name!r} is not floor-qualified (expected "
            f"'floor<N>_<capture-dir-basename>', e.g. 'floor1_baseline')")
    return int(match.group(1))


def _sidecar_batch_ttl(data_dir):
    """Optional `<data_dir>/batch.ttl` written by the deprecated `batch2ttl`."""
    path = os.path.join(data_dir, "batch.ttl")
    return path if os.path.isfile(path) else None


def _parse_batch_ttl(path):
    """Read one Batch out of a (legacy / sidecar) batch.ttl.

    Returns (graph, batch_iri, batch_name, batch_path_literal, floor_or_None).
    """
    g = Graph()
    g.parse(path, format="turtle")
    batches = sorted(set(g.subjects(RDF.type, HW1.Batch)), key=str)
    if len(batches) != 1:
        raise ValueError(
            f"{path}: expected exactly one hw1:Batch subject, found {len(batches)}"
            + (f": {', '.join(str(b) for b in batches)}" if batches else "")
            + ". A batch file describes exactly one capture (§4.1).")
    batch = batches[0]
    name_lit = g.value(batch, HW1.batchName)
    path_lit = g.value(batch, HW1.batchPath)
    floor_lit = g.value(batch, HW1.floor)
    if name_lit is None or path_lit is None:
        raise ValueError(
            f"{path}: the batch carries no hw1:batchName and/or no hw1:batchPath, "
            f"so neither the frame IRIs nor the pixels can be reached from it "
            f"(§4.1).")
    floor = int(floor_lit) if floor_lit is not None else None
    return g, URIRef(str(batch)), str(name_lit), str(path_lit), floor


def _capture_dir(batch_path, batch_file):
    """Capture dir for a batch name. Full strategy note: docs/triplestore.md."""
    candidates = [os.path.abspath(batch_path),
                  os.path.dirname(_resolve_batch_file(batch_file))]
    for candidate in candidates:
        if _is_capture_dir(candidate):
            return candidate
    raise ValueError(
        f"the batch's hw1:batchPath {batch_path!r} does not resolve to a capture "
        f"directory with rgb/ and depth/ subdirs. Tried "
        f"{', '.join(repr(c) for c in candidates)} (relative paths resolve against "
        f"the current working directory, {os.getcwd()!r}). Point hw1:batchFile at "
        f"the capture directory itself.")


def _resolve_declared_capture(batch_file, on_batch, path, exp, bf, *, semantic=False):
    """Resolve the declaration's capture to a data dir. Full strategy note: docs/triplestore.md."""
    resolved = _resolve_batch_file(batch_file)
    if _is_capture_dir(resolved):
        if on_batch is None:
            expected = data_batch_iri(batch_name(resolved, 1)) if semantic else batch_iri(batch_name(resolved, 1))
            raise ValueError(
                f"{path}: the experiment declares no hw1:onBatch. State the batch "
                f"IRI of the capture at {batch_file!r} — it is checked against the "
                f"directory, and that check is what catches 'measured capture A, "
                f"wrote into capture B'. Add:\n"
                f"    {_fmt_term(exp)} hw1:onBatch {_fmt_term(expected)} .")
        declared_name = _batch_name_from_batch_iri(on_batch)
        try:
            floor = _floor_from_batch_name(declared_name)
        except ValueError as exc:
            raise ValueError(
                f"{path}: hw1:onBatch names {_fmt_term(on_batch)}, which is not "
                f"a floor-qualified batch IRI of this assignment "
                f"(expected {DATA_NS}batch/floor<N>_<capture-dir-basename>). {exc} "
                f"Offending triple:\n    {_fmt_triple(exp, HW1.onBatch, on_batch)}")
        expected_name = batch_name(resolved, floor)
        expected_iri = data_batch_iri(expected_name) if semantic else batch_iri(expected_name)
        if URIRef(str(on_batch)) != expected_iri:
            raise ValueError(
                f"{path}: hw1:onBatch names {_fmt_term(on_batch)} but the capture "
                f"at {batch_file!r} is {_fmt_term(expected_iri)}. That is the "
                f"'measured capture A, wrote into capture B' bug at declaration "
                f"time — an error, not a warning (§4.2). Offending "
                f"triple:\n    {_fmt_triple(exp, HW1.onBatch, on_batch)}")
        return resolved, expected_iri, expected_name

    if os.path.isfile(resolved):
        _g, batch, name, stored_path, _floor = _parse_batch_ttl(resolved)
        data_dir = _capture_dir(stored_path, batch_file)
        if on_batch is None:
            raise ValueError(
                f"{path}: the experiment declares no hw1:onBatch. State the batch "
                f"IRI you believe {batch_file!r} describes — it is checked against "
                f"the file, and that check is what catches 'measured capture A, "
                f"wrote into capture B'. Add:\n"
                f"    {_fmt_term(exp)} hw1:onBatch {_fmt_term(batch)} .")
        if URIRef(str(on_batch)) != batch:
            raise ValueError(
                f"{path}: hw1:onBatch names {_fmt_term(on_batch)} but {batch_file!r} "
                f"describes {_fmt_term(batch)}. That is the 'measured capture A, "
                f"wrote into capture B' bug at declaration time — an error, not a "
                f"warning (§4.2). Offending triple:\n"
                f"    {_fmt_triple(exp, HW1.onBatch, on_batch)}")
        return data_dir, batch, name

    raise ValueError(
        f"{path}: hw1:batchFile {batch_file!r} does not resolve to a capture "
        f"directory with rgb/ and depth/ subdirs, or to a (deprecated) batch.ttl "
        f"({resolved!r}; relative paths resolve against the current working "
        f"directory, {os.getcwd()!r}). Point it at the capture directory. "
        f"Offending triple:\n    {_fmt_triple(exp, HW1.batchFile, bf)}")


def _declaration_setting_value(node, param_local, decl, lit, path):
    """One declared setting value, parsed by kind. Full strategy note: docs/triplestore.md."""
    kind = decl["kind"]
    dt = lit.datatype
    triple = _fmt_triple(node, HW1.settingValue, lit)
    numeric = (XSD.double, XSD.decimal, XSD.float, XSD.integer)
    if kind == "string":
        if dt is not None and dt != XSD.string:
            raise ValueError(
                f"{path}: hw1:{param_local} is declared hw1:paramValueKind \"string\", "
                f"but its hw1:settingValue carries datatype {dt}. Offending triple:\n"
                f"    {triple}")
        return str(lit)
    if kind == "integer":
        if dt != XSD.integer:
            raise ValueError(
                f"{path}: hw1:{param_local} is declared hw1:paramValueKind \"integer\" "
                f"— write it as a bare integer (e.g. 20) or "
                f"\"20\"^^xsd:integer, not as {dt or 'an untyped literal'}. Offending "
                f"triple:\n    {triple}")
        return int(lit)
    if dt not in numeric:
        raise ValueError(
            f"{path}: hw1:{param_local} is declared hw1:paramValueKind \"double\", but "
            f"its hw1:settingValue is {dt or 'an untyped literal'}. Write "
            f"\"0.05\"^^xsd:double (or the bare Turtle form 0.05). Offending triple:\n"
            f"    {triple}")
    return float(lit)


def read_declaration(path):
    """Validate a student declaration: batch, selection, settings, prediction TODOs. Full strategy note: docs/triplestore.md."""
    text = _read_text(path)
    student_text, _machine_text = _split_sections(text)

    g = Graph()
    try:
        g.parse(data=student_text, format="turtle")
    except Exception as exc:
        raise ValueError(
            f"{path}: the declaration (everything above the machine marker) is not "
            f"valid Turtle: {exc}") from None

    decls = load_parameter_declarations(_ONTOLOGY_TTL)
    factors = load_quality_factors(_ONTOLOGY_TTL)
    menu = _selectable_factors(factors)
    run_factors = _run_level_factors(factors)

    # ── the one Experiment subject, and its name ──────────────────────────────
    exp = _sole_experiment(g, path)
    exp_name = _experiment_name_from_iri(exp, path)
    stem = os.path.splitext(os.path.basename(path))[0]
    if exp_name != stem:
        raise ValueError(
            f"{path}: the experiment names itself {exp_name!r} but the file is called "
            f"{stem!r}.ttl. The IRI tail IS the experiment's identity and so is the "
            f"file name (§2/§6); they must agree, and choosing one for "
            f"you would silently rename your experiment. Offending triple:\n"
            f"    {_fmt_triple(exp, RDF.type, HW1.Experiment)}")

    # ── the whitelist: verdicts are computed, never declared ──────────────────
    setting_nodes = set(g.objects(exp, HW1.hasFactorSetting))
    for s, p, o in g:
        if s == exp:
            if p not in _DECLARATION_EXPERIMENT_PREDICATES:
                raise ValueError(
                    f"{path}: {_fmt_term(p)} may not appear on a declared "
                    f"hw1:Experiment. A declaration states its batch, its factor "
                    f"selection and its settings; values, statuses, annotations, "
                    f"pairs and runs are COMPUTED by `api.py experiment` and written "
                    f"below the marker (§4.2). Declarable predicates: "
                    f"{', '.join(_fmt_term(q) for q in _DECLARATION_EXPERIMENT_PREDICATES)}. "
                    f"Offending triple:\n    {_fmt_triple(s, p, o)}")
        elif s in setting_nodes:
            if p not in _DECLARATION_SETTING_PREDICATES:
                raise ValueError(
                    f"{path}: {_fmt_term(p)} may not appear on a hw1:FactorSetting. "
                    f"Declarable predicates: "
                    f"{', '.join(_fmt_term(q) for q in _DECLARATION_SETTING_PREDICATES)} "
                    f"(§4.2). Offending triple:\n    {_fmt_triple(s, p, o)}")
        else:
            raise ValueError(
                f"{path}: {_fmt_term(s)} is neither the declared hw1:Experiment nor "
                f"one of its hw1:hasFactorSetting nodes, so nothing in this project "
                f"reads it. A declaration describes ONE experiment and its settings "
                f"and nothing else (§4.2). Offending triple:\n"
                f"    {_fmt_triple(s, p, o)}")

    # ── the batch: declaration -> batchFile -> capture directory -> pixels ──
    bf = g.value(exp, HW1.batchFile)
    if bf is None:
        raise ValueError(
            f"{path}: the experiment declares no hw1:batchFile, so nothing says which "
            f"capture to measure. Add e.g.\n"
            f"    {_fmt_term(exp)} hw1:batchFile \"eval/first_floor\" .\n"
            f"(§4.2: the path to the capture directory, resolved against "
            f"the current working directory when relative)")
    batch_file = str(bf)
    on_batch = g.value(exp, HW1.onBatch)
    schema_version = str(g.value(exp, HW1.schemaVersion) or "4.0.0")
    data_dir, batch, batch_name_str = _resolve_declared_capture(
        batch_file, on_batch, path, exp, bf,
        semantic=schema_version.startswith("5"))

    # ── the selection ─────────────────────────────────────────────────────────
    selected_terms = list(g.objects(exp, HW1.evaluatesFactor))
    if not selected_terms:
        raise ValueError(
            f"{path}: the experiment selects no factor. hw1:evaluatesFactor is "
            f"required and takes 1..{len(menu)} of the menu (§4.2): "
            f"{', '.join('hw1:' + f for f in sorted(menu))}. An empty selection would "
            f"measure nothing and qualify nothing.")
    selected = []
    for term in selected_terms:
        local = _local(term)
        if local in run_factors:
            raise ValueError(
                f"{path}: hw1:{local} is a RUN-LEVEL factor and is not selectable — it "
                f"is always evaluated, by `reconstruct.py`, on the reconstruction "
                f"outcome rather than on the input rasters. hw1:evaluatesFactor scopes "
                f"INPUT-quality factors only (§4.2). Offending triple:\n"
                f"    {_fmt_triple(exp, HW1.evaluatesFactor, term)}")
        if local not in menu:
            raise ValueError(
                f"{path}: hw1:{local} is not a factor of the menu. The menu is "
                f"{', '.join('hw1:' + f for f in sorted(menu))} (§4.2); "
                f"factor names come from the TBox, not from the declaration. Offending "
                f"triple:\n    {_fmt_triple(exp, HW1.evaluatesFactor, term)}")
        selected.append(local)
    selected = sorted(set(selected))

    # ── the settings ──────────────────────────────────────────────────────────
    run_params = set(_run_factor_parameters(decls, factors))
    given = {}
    for node in sorted(setting_nodes, key=str):
        param_term = g.value(node, HW1.settingParameter)
        if param_term is None:
            raise ValueError(
                f"{path}: a hw1:FactorSetting has no hw1:settingParameter, so nothing "
                f"says what it sets. Offending triple:\n"
                f"    {_fmt_triple(exp, HW1.hasFactorSetting, node)}")
        local = _local(param_term)
        if local not in decls:
            raise ValueError(
                f"{path}: {_fmt_term(param_term)} is not a declared hw1:Parameter. "
                f"Parameter names come from the TBox ({_ONTOLOGY_TTL}), not from the "
                f"declaration; a typo surviving as an extra FactorSetting would record "
                f"a treatment nobody ran. Declared parameters: "
                f"{', '.join(sorted(decls))}. Offending triple:\n"
                f"    {_fmt_triple(node, HW1.settingParameter, param_term)}")
        decl = decls[local]

        if decl["role"] == "GenerationSetting":
            raise ValueError(
                f"{path}: hw1:{local} is declared hw1:paramRole hw1:GenerationSetting. "
                f"It describes how the PIXELS were produced, so it lives on the Batch: "
                f"`api.py batch2ttl --data-dir <dir> --floor <n> --gen {local}=<value>` "
                f"(optional sidecar). Generation settings never enter an experiment, "
                f"and they are never defaulted — absence means NOT ASSERTED "
                f"(§5). Offending triple:\n"
                f"    {_fmt_triple(node, HW1.settingParameter, param_term)}")

        role_term = g.value(node, HW1.settingRole)
        if role_term is None or _local(role_term) != decl["role"]:
            raise ValueError(
                f"{path}: the setting of hw1:{local} declares hw1:settingRole "
                f"{_fmt_term(role_term) if role_term is not None else '(absent)'}, but "
                f"the TBox declares hw1:paramRole hw1:{decl['role']}. The role says "
                f"WHERE a setting lives and WHAT TO FIX when it is the culprit, so it "
                f"is checked rather than copied. Offending triple:\n"
                f"    {_fmt_triple(node, HW1.settingRole, role_term)}")

        for_factors = list(g.objects(node, HW1.settingForFactor))
        if len(for_factors) != 1 or URIRef(str(for_factors[0])) != decl["primaryIri"]:
            found = ", ".join(_fmt_term(f) for f in for_factors) or "(none)"
            primary = _fmt_term(decl["primaryIri"])
            raise ValueError(
                f"{path}: the setting of hw1:{local} must carry exactly one "
                f"hw1:settingForFactor, its parameter's hw1:paramPrimaryFactor "
                f"{primary} — it is single-valued in v3 (§4.5) and "
                f"readers traverse hw1:paramAffectsFactor in the TBox for the rest. "
                f"Found: {found}. Write:\n"
                f"    {_fmt_triple(node, HW1.settingForFactor, decl['primaryIri'])}")

        lit = g.value(node, HW1.settingValue)
        if lit is None or not isinstance(lit, Literal):
            raise ValueError(
                f"{path}: the setting of hw1:{local} carries no hw1:settingValue "
                f"literal, so it records no level at all. Offending triple:\n"
                f"    {_fmt_triple(exp, HW1.hasFactorSetting, node)}")
        value = _declaration_setting_value(node, local, decl, lit, path)

        touched = set((decl["primary"],) + tuple(decl["affects"]))
        if not touched & set(selected) and local not in run_params:
            raise ValueError(
                f"{path}: hw1:{local} is set, but none of the factors it touches "
                f"({', '.join('hw1:' + f for f in sorted(touched))}) is selected by "
                f"hw1:evaluatesFactor ({', '.join('hw1:' + f for f in selected)}). That "
                f"records a level that changed nothing, which is as false as a "
                f"defaulted Generation setting (§4.2) — select the factor, "
                f"or drop the setting. Offending triple:\n"
                f"    {_fmt_triple(node, HW1.settingParameter, param_term)}")

        stored = _storable(value) if decl["kind"] == "double" else value
        if local in given and given[local] != stored:
            raise ValueError(
                f"{path}: hw1:{local} is set twice, to {given[local]!r} and to "
                f"{stored!r}. One parameter has one level per experiment; two levels "
                f"are two experiments, and the second one needs its own declaration "
                f"file (§3.1). Offending triple:\n"
                f"    {_fmt_triple(node, HW1.settingValue, lit)}")
        given[local] = stored

    # ── completeness: the selection-scoped required set (§4.2) ────────────────
    # `given` is a subset of this by construction — the scope check above rejects
    # any setting that touches nothing selected — so the union only makes
    # "student-given wins" total rather than adding anything.
    required_names = _required_parameters(selected, decls, factors) | set(given)
    required = {}
    for param in sorted(required_names):
        if param in given:
            required[param] = given[param]
            continue
        decl = decls[param]
        if decl["default"] is None:
            raise ValueError(
                f"{path}: hw1:{param} is required by this selection "
                f"(§4.2) but declares no hw1:paramDefault, and the declaration does not "
                f"set it. Either add a hw1:FactorSetting for it or declare a default "
                f"in {_ONTOLOGY_TTL}.")
        required[param] = (_storable(decl["default"]) if decl["kind"] == "double"
                           else decl["default"])

    return {"exp_iri": URIRef(str(exp)),
            "exp_name": exp_name,
            "schema_version": schema_version,
            "batch_file": batch_file,
            "batch_iri": URIRef(str(batch)),
            "batch_name": batch_name_str,
            "batch_path": data_dir,
            "selected": selected,
            "given": given,
            "required": required}


def _require_settings(settings, names, what):
    """Complete the required setting set with recorded defaults. Full strategy note: docs/triplestore.md."""
    missing = [n for n in names if n not in settings]
    if missing:
        raise ValueError(
            f"{what} needs parameter(s) {', '.join(missing)}, which the selection-scoped "
            f"required set (§4.2) did not put in the setting vector. That "
            f"means the TBox in {_ONTOLOGY_TTL} does not declare them as affecting the "
            f"selected factor — fix the declaration links there, not with a default here.")


def _status_property(value_local, factors):
    """Status predicate for an observable property. Full strategy note: docs/triplestore.md."""
    for info in factors.values():
        if info["over"] == value_local:
            return HW1[info["status"]]
    raise ValueError(
        f"no hw1:QualityFactor declares hw1:overProperty hw1:{value_local} in "
        f"{_ONTOLOGY_TTL}, so it has no hw1:statusProperty to write.")


def _write_observable(g, node, value_local, value, settings, factors):
    """Write one value + baked status triple pair. Full strategy note: docs/triplestore.md."""
    stored = _storable(value)
    g.add((node, HW1[value_local], _double_literal(stored)))
    status = status_for(value_local, stored, settings, factors)
    g.add((node, _status_property(value_local, factors), status))
    return status == HW1.Pass


def _fail_closed_value(value_local, factors):
    """Fail-closed scalar for a failed measurement (inf or 0.0 by polarity). Full strategy note: docs/triplestore.md."""
    for info in factors.values():
        if info["over"] == value_local:
            return float("-inf") if info["polarity"] == "higher" else float("inf")
    return float("nan")


def _measured(value_local, factors, where, measure, *args):
    """Run one frame measurer, fail-closed. Full strategy note: docs/triplestore.md."""
    try:
        return measure(*args)
    except Exception as exc:                             # noqa: BLE001 — deliberate
        value = _fail_closed_value(value_local, factors)
        print(f"[experiment] WARNING: hw1:{value_local} on {where} raised "
              f"{type(exc).__name__}: {exc} — storing the fail-closed value {value!r} "
              f"(§4.3: a failing measurer writes its fail-closed value, "
              f"never nothing)", file=sys.stderr, flush=True)
        return value


def _measured_with_mask(value_local, factors, where, measure, *args):
    """Run one depth measurer with its drop mask, fail-closed. Full strategy note: docs/triplestore.md."""
    try:
        result = measure(*args)
        if not isinstance(result, tuple) or len(result) not in (2, 3):
            raise TypeError("mask measurer must return (value, mask[, count])")
        value, mask = result[:2]
        array = np.asarray(mask)
        if array.ndim != 2 or array.dtype != np.uint8:
            raise TypeError("mask measurer must return an HxW uint8 mask")
        count = None if len(result) == 2 else int(result[2])
        return value, array, count
    except Exception as exc:                             # noqa: BLE001 — fail closed
        value = _fail_closed_value(value_local, factors)
        print(f"[experiment] WARNING: hw1:{value_local} on {where} raised "
              f"{type(exc).__name__}: {exc} — storing {value!r} without a mask",
              file=sys.stderr, flush=True)
        return value, None, 0


def _factor_for_observable(value_local, factors):
    matches = [name for name, info in factors.items() if info["over"] == value_local]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one QualityFactor over hw1:{value_local}, got {matches}")
    return matches[0]


def _write_mask_artifact(mask, mask_root, relative_to, factor_local, filename):
    """Persist one drop-mask PNG artifact. Full strategy note: docs/triplestore.md."""
    if mask_root is None or mask is None:
        return None
    factor_dir = os.path.join(mask_root, "masks", factor_local)
    os.makedirs(factor_dir, exist_ok=True)
    path = os.path.join(factor_dir, filename)
    Image.fromarray(np.asarray(mask, dtype=np.uint8)).save(path)
    base = os.getcwd() if relative_to is None else relative_to
    return os.path.relpath(path, base).replace(os.sep, "/")


def _check_batch_file(batch_ttl, name, expected_frames):
    """Validate hw1:batchFile against the capture dir. Full strategy note: docs/triplestore.md."""
    g = Graph()
    g.parse(batch_ttl, format="turtle")
    declared_names = sorted({str(o) for o in g.objects(None, HW1.batchName)})
    if declared_names != [name]:
        raise ValueError(
            f"{batch_ttl} declares hw1:batchName {declared_names!r}, but this "
            f"experiment measures batch {name!r}. A batch file describes exactly one "
            f"batch, and that name is inside every frame IRI the two files join on.")

    have_frames = {str(f) for f in g.objects(batch_iri(name), HW1.hasFrame)}
    want_frames = {str(f) for f in expected_frames}

    problems = []
    missing = sorted(want_frames - have_frames)
    extra = sorted(have_frames - want_frames)
    if missing:
        problems.append(f"{len(missing)} frame(s) on disk that batch.ttl does not "
                        f"declare (e.g. {missing[0]})")
    if extra:
        problems.append(f"{len(extra)} frame(s) in batch.ttl that are not on disk "
                        f"(e.g. {extra[0]})")
    if problems:
        raise ValueError(
            f"{batch_ttl} is stale: " + "; ".join(problems) + ". Annotations join to "
            f"frames on these IRIs and on nothing else, so a mismatch makes every "
            f"report show zero rows without erroring. Re-run `api.py batch2ttl "
            f"--data-dir <dir> --floor <n>`.")


def _semantic_schema_enabled(path=_ONTOLOGY_TTL):
    """Return whether the installed TBox advertises explicit Factor occurrences."""
    try:
        ontology = Graph()
        ontology.parse(path, format="turtle")
        return ((HW1.Factor, RDF.type, RDFS.Class) in ontology or
                any(ontology.triples((None, RDFS.subClassOf, HW1.Factor))))
    except Exception:
        return False


def _graph_uses_semantic_schema(graph, experiment):
    """Opt into the Factor model per artifact, while retaining legacy reads."""
    version = graph.value(experiment, HW1.schemaVersion)
    if version is not None:
        try:
            return int(str(version).split('.', 1)[0]) >= 5
        except (TypeError, ValueError):
            return False
    return any(graph.subjects(HW1.inExperiment, experiment))


def _semantic_factor_info(factors, factor_local):
    info = factors.get(factor_local, {})
    over = info.get("over", factor_local)
    return over, info


def _bind_readable_namespaces(g, batch_name_str, expname):
    """Bind short Turtle prefixes so output stays readable."""
    g.bind("hw1", HW1); g.bind("schema", SCHEMA); g.bind("xsd", XSD)
    g.bind("batch", Namespace(f"{DATA_NS}batch/"))
    g.bind("frame", Namespace(f"{DATA_NS}batch/{batch_name_str}/frame/"))
    g.bind("rgb", Namespace(f"{DATA_NS}batch/{batch_name_str}/rgb/"))
    g.bind("depth", Namespace(f"{DATA_NS}batch/{batch_name_str}/depth/"))
    g.bind("exp", Namespace(f"{DATA_NS}experiment/"))
    g.bind("factor", Namespace(f"{DATA_NS}experiment/{expname}/factor/"))
    g.bind("setting", Namespace(f"{DATA_NS}experiment/{expname}/setting/"))
    g.bind("run", Namespace(f"{DATA_NS}experiment/{expname}/run/"))


def _build_semantic_machine_graph(decl, data_dir, digest, decls, factors,
                                  artifact_root=None, artifact_relative_to=None):
    """Semantic (v5 DATA_NS) machine graph builder. Full strategy note: docs/triplestore.md."""
    exp, expname = decl["exp_iri"], decl["exp_name"]
    name, settings, selected = decl["batch_name"], decl["required"], decl["selected"]
    frames = _pair_frames(data_dir); g = Graph()
    _bind_readable_namespaces(g, name, expname)
    b = data_batch_iri(name)
    g.add((exp, HW1.declarationDigest, Literal(digest))); g.add((exp, HW1.onBatch, b))
    g.add((b, RDF.type, HW1.Batch)); g.add((b, HW1.batchName, Literal(name)))
    g.add((b, HW1.batchPath, Literal(data_dir)))
    for stem, rgb_path, depth_path in frames:
        f = data_frame_iri(name, stem); rgb = data_component_iri(name, stem, "rgb"); dep = data_component_iri(name, stem, "depth")
        g.add((b, HW1.hasFrame, f)); g.add((f, RDF.type, HW1.Frame)); g.add((f, HW1.frameIndex, Literal(int(stem), datatype=XSD.integer)))
        g.add((f, HW1.hasRGBImage, rgb)); g.add((rgb, RDF.type, HW1.RGBImage)); g.add((rgb, SCHEMA.contentUrl, Literal(rgb_path)))
        g.add((f, HW1.hasDepthImage, dep)); g.add((dep, RDF.type, HW1.DepthImage)); g.add((dep, SCHEMA.contentUrl, Literal(depth_path)))
    for param in sorted(set(settings) - set(decl["given"])):
        d = decls[param]; s = data_setting_iri(expname, d["primary"], param)
        for triple in ((exp, HW1.hasFactorSetting, s), (s, RDF.type, HW1.FactorSetting), (s, HW1.settingParameter, d["iri"]),
                       (s, HW1.settingRole, HW1[d["role"]]), (s, HW1.settingForFactor, d["primaryIri"])): g.add(triple)
        g.add((s, HW1.settingValue, _setting_value_literal(settings[param], d["kind"])))
    count = 0; measured = 0
    for factor_local in selected:
        over, _ = _semantic_factor_info(factors, factor_local)
        if over in _FRAME_OBSERVABLES:
            spec = _FRAME_OBSERVABLES[over]
            for stem, rgb_path, depth_path in frames:
                node = data_factor_iri(expname, factor_local, stem); image = data_component_iri(name, stem, spec["modality"])
                g.add((node, RDF.type, HW1.Factor)); g.add((node, RDF.type, HW1.SingleImageFactor)); g.add((node, HW1.inExperiment, exp)); g.add((node, HW1.hasDefinition, HW1[factor_local])); g.add((node, HW1.hasCurrentFrame, image))
                # targetKind/evaluationPhase describe the reusable FactorDefinition
                # in the TBox; the occurrence carries only its
                # hasDefinition link and measured result.
                if "measure_mask" in spec:
                    value, mask, _count = _measured_with_mask(over, factors, f"frame {stem}", spec["measure_mask"], rgb_path, depth_path, settings)
                else:
                    value, mask = _measured(over, factors, f"frame {stem}", spec["measure"], rgb_path, depth_path, settings), None
                g.add((node, HW1.value, _double_literal(value))); g.add((node, HW1.status, status_for(over, _storable(value), settings, factors))); g.add((node, HW1.evaluationState, HW1.Measured)); measured += 1; count += 1
                if mask is not None:
                    mf = _write_mask_artifact(mask, artifact_root, artifact_relative_to, factor_local, f"{stem}.png")
                    if mf: g.add((node, HW1.maskFile, Literal(mf)))
        elif over in _PAIR_OBSERVABLES:
            spec = _PAIR_OBSERVABLES[over]
            for (s0, _, d0), (s1, _, d1) in zip(frames, frames[1:]):
                i, j = int(s0), int(s1); node = data_factor_iri(expname, factor_local, j, i)
                g.add((node, RDF.type, HW1.Factor)); g.add((node, RDF.type, HW1.DepthPairFactor)); g.add((node, HW1.inExperiment, exp)); g.add((node, HW1.hasDefinition, HW1[factor_local]))
                g.add((node, HW1.hasPrevious, data_component_iri(name, s0, "depth"))); g.add((node, HW1.hasCurrentFrame, data_component_iri(name, s1, "depth"))); count += 1
                if spec.get("deferred"):
                    g.add((node, HW1.evaluationState, HW1.Pending))
                else:
                    value, mask, support = _measured_with_mask(over, factors, f"pair {i}_{j}", spec["measure_mask"], d0, d1, settings)
                    g.add((node, HW1.value, _double_literal(value))); g.add((node, HW1.status, status_for(over, _storable(value), settings, factors))); g.add((node, HW1.evaluationState, HW1.Measured)); measured += 1
                    if mask is not None:
                        mf = _write_mask_artifact(mask, artifact_root, artifact_relative_to, factor_local, f"{i}_{j}.png")
                        if mf: g.add((node, HW1.maskFile, Literal(mf)))
                    if spec.get("count_property"): g.add((node, HW1.supportCount, Literal(int(support or 0), datatype=XSD.integer)))
    if count and count == measured: g.add((exp, RDF.type, HW1.FullEvaluatedFrames))
    return g, {"frames": len(frames), "factors": count, "annotations": 0, "pairs": 0}


def build_machine_graph(decl, data_dir, digest, decls=None, factors=None,
                        artifact_root=None, artifact_relative_to=None):
    """Full machine section: settings, annotations, pairs, statuses. Full strategy note: docs/triplestore.md."""
    decls = load_parameter_declarations(_ONTOLOGY_TTL) if decls is None else decls
    factors = load_quality_factors(_ONTOLOGY_TTL) if factors is None else factors

    # New TBoxes advertise hw1:Factor; route only that schema through the
    # occurrence writer.  Legacy artifacts continue through the container adapter
    # below, which is important for historical course fixtures.
    # The declaration's explicit schemaVersion selects the writer.  This keeps
    # historical v4 declarations readable while new scaffolds emit v5 Factors.
    if str(decl.get("schema_version", "")).startswith("5"):
        return _build_semantic_machine_graph(
            decl, data_dir, digest, decls, factors,
            artifact_root=artifact_root, artifact_relative_to=artifact_relative_to)

    exp = decl["exp_iri"]
    expname = decl["exp_name"]
    name = decl["batch_name"]
    settings = decl["required"]
    selected = decl["selected"]

    # The selection, translated once from factor names to the observables that
    # carry them, in table order so the file is deterministic.
    menu = _selectable_factors(factors)
    chosen = {menu[f] for f in selected}
    frame_values = [v for v in _FRAME_OBSERVABLES if v in chosen]
    pair_values = [v for v in _PAIR_OBSERVABLES if v in chosen]
    modalities = [k for k in _ANNOTATION_KINDS
                  if any(_FRAME_OBSERVABLES[v]["modality"] == k for v in frame_values)]

    # Every parameter every selected measurer needs, checked against the vector
    # BEFORE the first PNG is opened.
    for value_local in frame_values:
        _require_settings(settings, _FRAME_OBSERVABLES[value_local]["params"],
                          f"hw1:{value_local}")
    for value_local in pair_values:
        _require_settings(settings, _PAIR_OBSERVABLES[value_local]["params"],
                          f"hw1:{value_local}")

    g = Graph()

    # ── the seal (§3.1) ──────────────────────────────────────────
    # First triple of the machine section, and the reason the rest of it can be
    # trusted to describe the declaration above it.
    g.add((exp, HW1.declarationDigest, Literal(digest)))

    # ── the defaulted half of the setting vector (§4.2) ───────────────────────
    for param in sorted(set(settings) - set(decl["given"])):
        d = decls[param]
        s = setting_iri(expname, d["primary"], param)
        g.add((exp, HW1.hasFactorSetting, s))
        g.add((s, RDF.type, HW1.FactorSetting))
        g.add((s, HW1.settingParameter, d["iri"]))
        # The role is DENORMALISED onto the node on purpose: `explore` then reads
        # the verdict ("regenerate the data" / "re-measure" / "re-qualify") off the
        # node it already has, with no second join into the TBox.
        g.add((s, HW1.settingRole, HW1[d["role"]]))
        # SINGLE-VALUED in v3 (§4.5): the primary factor only. Readers traverse
        # hw1:paramAffectsFactor in the TBox for the rest, which is also what makes
        # a student's blank-node setting legal — nothing has to re-open it.
        g.add((s, HW1.settingForFactor, d["primaryIri"]))
        g.add((s, HW1.settingValue, _setting_value_literal(settings[param], d["kind"])))

    # ── one FrameAnnotation per frame per SELECTED modality ───────────────────
    frames = _pair_frames(data_dir)
    total = len(frames)
    n_annotations = 0
    for n, (stem, rgb_path, depth_path) in enumerate(frames, start=1):
        for kind in modalities:
            ann = annotation_iri(expname, stem, kind)
            g.add((exp, HW1.producesAnnotation, ann))
            g.add((ann, RDF.type, HW1.FrameAnnotation))
            g.add((ann, HW1.annotatesFrame, frame_iri(name, stem)))
            # frameIndex is repeated here although the Frame already carries it: it
            # is what lets a report order by capture order without loading the batch
            # file at all (§2).
            g.add((ann, HW1.frameIndex, Literal(int(stem), datatype=XSD.integer)))
            # describesImage (new in v3): the annotation's link to the raster it
            # measured. annotatesFrame stays for frame-level joins; this one says
            # WHICH OF THE TWO IMAGES the numbers on this node are about.
            g.add((ann, HW1.describesImage, component_iri(name, stem, kind)))

            passed = []
            for value_local in frame_values:
                if _FRAME_OBSERVABLES[value_local]["modality"] != kind:
                    continue
                spec = _FRAME_OBSERVABLES[value_local]
                if "measure_mask" in spec:
                    value, mask, _count = _measured_with_mask(
                        value_local, factors, f"frame {stem} ({kind})",
                        spec["measure_mask"], rgb_path, depth_path, settings)
                    factor_local = _factor_for_observable(value_local, factors)
                    mask_file = _write_mask_artifact(
                        mask, artifact_root, artifact_relative_to, factor_local,
                        f"{stem}.png")
                    if mask_file is not None:
                        g.add((ann, HW1.maskFile, Literal(mask_file)))
                else:
                    value = _measured(value_local, factors,
                                      f"frame {stem} ({kind})", spec["measure"],
                                      rgb_path, depth_path, settings)
                passed.append(_write_observable(g, ann, value_local, value,
                                                settings, factors))
            if kind == "rgb":
                # meanValue: the value and NO status, whenever an rgb annotation
                # exists (§4.3). There is no QualityFactor over it, so there is no
                # polarity and no threshold — asking `status_for` for one raises by
                # design. It ships so a student can check "is the picture dark on
                # average?" against the factors that actually predict reconstruction
                # failure, and discover that it does not.
                g.add((ann, HW1.meanValue, _double_literal(
                    _measured("meanValue", factors, f"frame {stem} (rgb)",
                              lambda p: frame_mean_value(p), rgb_path))))
            # The aggregate of §4.5, on the node that says what was aggregated:
            # Pass iff every SELECTED factor of THIS modality passed.
            g.add((ann, HW1.qualificationStatus,
                   HW1.Pass if all(passed) else HW1.Fail))
            n_annotations += 1
        if n % _PROGRESS_EVERY == 0 or n == total:
            print(f"[experiment] frames {n}/{total}", file=sys.stderr, flush=True)

    # ── one FramePair per consecutive pair, ALWAYS ────────────────────────────
    steps = list(zip(frames, frames[1:]))
    total_pairs = len(steps)
    # `enumerate(..., start=0)` is the hw1:pairIndex: a 0-based ORDINAL over the
    # pairs in ascending order, NOT a frame stem (§4.3). The stems are
    # in the IRI, where a gap in the capture stays visible as `41_43`; the ordinal
    # is what a reader sorts by, because unpadded stems sort lexicographically as
    # 0, 1, 10, 100, 11 and a time series read in that order looks like noise.
    for pair_index, ((s0, _, d0), (s1, _, d1)) in enumerate(steps):
        i, j = int(s0), int(s1)
        p = pair_iri(expname, i, j)
        g.add((exp, HW1.producesPair, p))
        g.add((p, RDF.type, HW1.FramePair))
        g.add((p, HW1.sourceFrame, frame_iri(name, s0)))
        g.add((p, HW1.targetFrame, frame_iri(name, s1)))
        g.add((p, HW1.pairIndex, Literal(pair_index, datatype=XSD.integer)))

        pair_passed = []
        for value_local in pair_values:
            spec = _PAIR_OBSERVABLES[value_local]
            if spec.get("deferred"):
                continue
            value, mask, count = _measured_with_mask(
                value_local, factors, f"pair {i}_{j}", spec["measure_mask"],
                d0, d1, settings)
            pair_passed.append(_write_observable(g, p, value_local, value,
                                                 settings, factors))
            if spec.get("count_property") is not None:
                g.add((p, HW1[spec["count_property"]],
                       Literal(int(count or 0), datatype=XSD.integer)))
            factor_local = _factor_for_observable(value_local, factors)
            mask_file = _write_mask_artifact(
                mask, artifact_root, artifact_relative_to, factor_local,
                f"{i}_{j}.png")
            if mask_file is not None:
                g.add((p, HW1.maskFile, Literal(mask_file)))
        # `all([])` is True, and that is the contract: a pair carries a TOTAL
        # qualificationStatus, vacuously Pass when no pair factor was selected
        # (§4.3/§4.5) — the pair is minted for its adjacency, not for a verdict
        # nobody asked for.
        g.add((p, HW1.qualificationStatus,
               HW1.Pass if all(pair_passed) else HW1.Fail))
        if pair_values and ((pair_index + 1) % _PROGRESS_EVERY == 0
                            or pair_index + 1 == total_pairs):
            print(f"[experiment] pairs {pair_index + 1}/{total_pairs}",
                  file=sys.stderr, flush=True)

    return g, {"frames": total, "annotations": n_annotations, "pairs": total_pairs}


def cmd_experiment(args):
    """Measure + append the machine section ONCE (hard error if sealed). Full strategy note: docs/triplestore.md."""
    path = args.declaration
    if not os.path.isfile(path):
        raise ValueError(f"{path}: no such declaration file")

    text = _read_text(path)
    if _marker_offset(text) is not None:
        raise ValueError(
            f"{path} HAS ALREADY BEEN ASSESSED — it carries the machine marker, so it "
            f"already holds one experiment's values and verdicts. Experiments are "
            f"WRITE-ONCE (§3.1): there is no --force and no re-assess "
            f"path, because a re-assessment would silently replace the evidence in "
            f"your lab notebook. To try a different threshold, factor selection or "
            f"batch: COPY THE DECLARATION — every line ABOVE the machine marker — "
            f"into a NEW file, edit it there (including the hw1:Experiment IRI tail, "
            f"which must equal the new file's stem), and run "
            f"`api.py experiment <new-name>.ttl`. The old file stays as it is: that "
            f"is what makes hw1/experiments/ a lab notebook.")

    decls = load_parameter_declarations(_ONTOLOGY_TTL)
    factors = load_quality_factors(_ONTOLOGY_TTL)
    decl = read_declaration(path)

    required = decl["required"]
    given = decl["given"]
    defaulted = sorted(set(required) - set(given))
    shown_given = ", ".join("{}={!r}".format(k, given[k]) for k in sorted(given))
    shown_default = ", ".join("{}={!r}".format(k, required[k]) for k in defaulted)
    print(f"[experiment] declaration {path}")
    print(f"[experiment] experiment {decl['exp_name']!r} on batch "
          f"{decl['batch_name']!r} (via {decl['batch_file']})")
    print(f"[experiment] evaluates {len(decl['selected'])} factor(s): "
          f"{', '.join(decl['selected'])}")
    print(f"[experiment] settings given:      {shown_given or '(none)'}")
    print(f"[experiment] settings defaulted:  {shown_default or '(none)'}")
    n_meas = sum(1 for k in required if decls[k]["role"] == "MeasurementSetting")
    n_qual = sum(1 for k in required if decls[k]["role"] == "QualificationSetting")
    print(f"[experiment] setting vector: {len(required)} setting(s) — {n_meas} "
          f"Measurement (how pixels are scored) + {n_qual} Qualification (where the "
          f"Pass line sits), total over THIS SELECTION plus the three run-factor "
          f"parameters (§4.2). Generation settings live on the batch.")

    data_dir = decl["batch_path"]
    frames = _pair_frames(data_dir)
    # Legacy declarations still name a batch.ttl: keep the stale-file check so a
    # sidecar that drifted from the pixels cannot silently join to nothing.
    declared = _resolve_batch_file(decl["batch_file"])
    if os.path.isfile(declared) and not _is_capture_dir(declared):
        _check_batch_file(declared, decl["batch_name"],
                          [frame_iri(decl["batch_name"], stem)
                           for stem, _, _ in frames])

    # THE SEAL COVERS THE BYTES THAT WILL BE ABOVE THE MARKER. If the declaration
    # does not end in a newline, one is appended so the marker starts its own line —
    # and the digest is computed over the text INCLUDING that newline, because that
    # is what the file will hold and what `read_experiment` will re-hash.
    student_text = text if text.endswith("\n") else text + "\n"
    digest = _declaration_digest(student_text.encode("utf-8"))

    artifact_root = os.path.splitext(os.path.abspath(path))[0]
    g, counts = build_machine_graph(
        decl, data_dir, digest, decls=decls, factors=factors,
        artifact_root=artifact_root,
        artifact_relative_to=os.path.dirname(os.path.abspath(path)))

    body = g.serialize(format="turtle")
    if isinstance(body, bytes):                          # rdflib < 6 returned bytes
        body = body.decode("utf-8")
    # APPEND, so no failure in this command can touch the student's declaration.
    with open(path, "a", encoding="utf-8") as fh:
        if not text.endswith("\n"):
            fh.write("\n")
        fh.write(MACHINE_MARKER + "\n\n")
        fh.write(body if body.endswith("\n") else body + "\n")

    # The verdict counts, printed because they are the number a student actually
    # wants next and because a selection that will turn out empty is visible here
    # rather than three commands later.
    annotations = set(g.subjects(RDF.type, HW1.FrameAnnotation))
    pairs = set(g.subjects(RDF.type, HW1.FramePair))
    bad_ann = sum(1 for a in annotations
                  if g.value(a, HW1.qualificationStatus) == HW1.Fail)
    bad_pairs = sum(1 for p in pairs
                    if g.value(p, HW1.qualificationStatus) == HW1.Fail)
    n_ann, n_pairs = len(annotations), len(pairs)
    modalities = ", ".join(_modalities_of(g)) or "no modality selected"
    print(f"[experiment] batch {decl['batch_name']!r}: {counts['frames']} frames -> "
          f"{n_ann} annotation(s) [{modalities}], {n_pairs} pair(s)")
    print(f"[experiment] verdicts: {n_ann - bad_ann}/{n_ann} annotations Pass, "
          f"{n_pairs - bad_pairs}/{n_pairs} pairs Pass (under the {n_qual} thresholds "
          f"recorded above)")
    print(f"[experiment] experiment IRI  {decl['exp_iri']}")
    print(f"[experiment] declarationDigest {digest}")
    print(f"[experiment] appended {len(g)} triples below the marker -> {path}")
    print(f"[experiment] this file is now SEALED: edit it above the marker and every "
          f"reader will refuse it. Next: `api.py explore {path}`, then "
          f"`reconstruct.py --data_root {data_dir} --experiment {path}`.")
    return 0


def _modalities_of(g):
    """Modalities covered by a factor selection."""
    kinds = []
    for kind in _ANNOTATION_KINDS:
        if any(str(a).endswith(f"/{kind}")
               for a in g.subjects(RDF.type, HW1.FrameAnnotation)):
            kinds.append(kind)
    return kinds


# =============================================================================
# explore  — read-only terminal tables (§7.1)
#   Four views: a batch file, an unassessed declaration, an assessed experiment
#   (settings, per-frame table, summary and the computed VERDICT section), and
#   several experiments side by side. It measures nothing and writes nothing, and
#   it re-implements no rule: `read_declaration` and `read_experiment` above are
#   its two inputs, and the TBox wiring (`overProperty` / `statusProperty` /
#   `polarity` / `qualifiedBy` / `paramPrimaryFactor` / `paramAffectsFactor`) is
#   how it resolves factors and culprit settings generically — exactly as the
#   deleted SPARQL queries did (§10).
#
#   READ-ONLY, AND THAT IS AN INVARIANT OF THIS WHOLE SECTION. Nothing below
#   opens a file for writing, calls `write_run`, serializes a graph to disk or
#   mutates a parsed graph. The graphs it parses are throwaway projections of
#   files on disk; the files themselves are never reopened after being read.
#
#   THE SEAL IS NOT SHORT-CIRCUITED. View 3 goes through `read_experiment`,
#   which verifies `hw1:declarationDigest` before returning anything (§3.1), so a
#   file whose declaration was edited after assessment raises here instead of
#   printing a pretty table of stale numbers.
#
#   THE WIDE-TABLE DECISION (§7.1 asks for one, so it is stated here rather than
#   left to the reader): the per-frame table PRINTS EVERY ROW and keeps every
#   COLUMN narrow. A 6-factor selection over a 387-frame batch is 387 lines of
#   117 characters (measured) — one line per frame, value and status merged into
#   cell ("0.0312 P"), long observable names abbreviated with a legend printed
#   above the table. Rows are cheap: a terminal has scrollback, and plain ASCII
#   one-line-per-frame means `explore … | grep ' F'` finds every failing frame.
#   Columns are not cheap: a wrapped line destroys the alignment that makes a
#   column scannable at all. The alternative — eliding all-Pass rows — was
#   REJECTED because `query` is deleted (§10) and the frozen command surface (§7)
#   gives `explore` no flag with which to ask for an elided row back, so an
#   elision here would hide a measured value with no way to recover it. Nothing
#   in this section truncates anything silently; where a list is long it is
#   printed in full or the count of what was left out is printed with it.
# =============================================================================

# What to do about a culprit setting, by its parameter's hw1:paramRole
# (§4.2/§7.1). Keyed by role — NOT by factor and NOT by parameter:
# adding a menu factor, or a parameter for one, must require no edit here.
_ROLE_FIX = {
    "GenerationSetting": "regenerate the data (this level is baked into the pixels)",
    "MeasurementSetting": "change the number, re-measure -> a NEW experiment",
    "QualificationSetting": "change the threshold, re-qualify -> a NEW experiment",
}

# Width at which an observable's local name is abbreviated in a per-frame column
# header. `_abbrev` falls back to full names if truncation would collide, so this
# is a cosmetic bound and never an ambiguity.
_COL_ABBREV = 10

_INDENT = "  "

# Prose is wrapped at this column; tables never are (see the section header).
_WRAP = 92


# ── output primitives: plain ASCII, aligned columns, no dependency ────────────
def _rule(title):
    """A top-level section heading."""
    print()
    print(title)
    print("=" * len(title))


def _sub(title):
    """A block heading inside a view."""
    print()
    print(title)
    print("-" * len(title))


def _kv(label, value):
    """One `label   value` line of a header block.

    Deliberately NOT wrapped: the values here are IRIs and file paths, and a
    wrapped path cannot be copied out of a terminal in one go.

    The gap is a MINIMUM, not a fixed width: experiment names are student-chosen
    and routinely overrun 20 columns, and a name run flush into the path it labels
    is unreadable exactly where the reader most needs to tell them apart.
    """
    gap = max(20 - len(label), 2)
    print(f"{_INDENT}{label}{' ' * gap}{value}")


def _note(text, indent=_INDENT):
    """Prose, greedy-wrapped to `_WRAP` columns. Tables are never wrapped; notes are.

    A wrapped TABLE loses the column alignment that makes it scannable, so the
    per-frame table stays on one line per frame however wide it gets (see the
    section header). A wrapped SENTENCE loses nothing, and the explanations
    around these tables carry the contract references a reader needs.
    """
    line = ""
    for word in text.split():
        if line and len(indent) + len(line) + 1 + len(word) > _WRAP:
            print(indent + line)
            line = word
        else:
            line = f"{line} {word}" if line else word
    if line:
        print(indent + line)


def _render_table(headers, rows, right=()):
    """Aligned ASCII table -> a list of lines. `right` = column indices to right-align.

    Deliberately hand-rolled: `explore` must run under the same rdflib-only stack
    as everything else (§7/§10 removed the last non-stdlib
    dependency this project had beyond numpy/Pillow/rdflib), and a table is
    twelve lines of `str.ljust`.
    """
    head = [str(h) for h in headers]
    body = [["" if c is None else str(c) for c in row] for row in rows]
    widths = [len(h) for h in head]
    for row in body:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells):
        out = [(cell.rjust(widths[i]) if i in right else cell.ljust(widths[i]))
               for i, cell in enumerate(cells)]
        return (_INDENT + "  ".join(out)).rstrip()

    lines = [fmt(head), _INDENT + "  ".join("-" * w for w in widths)]
    lines.extend(fmt(row) for row in body)
    return lines


def _print_table(headers, rows, right=()):
    """`_render_table`, printed. An empty table says so rather than printing nothing."""
    if not rows:
        print(f"{_INDENT}(no rows)")
        return
    for line in _render_table(headers, rows, right):
        print(line)


def _fmt_num(term):
    """One numeric literal as a narrow fixed-shape string. INF / -INF / NaN survive.

    The fail-closed sentinels of §9 are values a reader must SEE — an `inf`
    printed as `1.0e+308` or, worse, silently reformatted to something finite,
    would hide exactly the rows the convention exists to surface.
    """
    if term is None:
        return "-"
    try:
        v = float(term)
    except (TypeError, ValueError):
        return str(term)
    if v != v:
        return "NaN"
    if v == float("inf"):
        return "INF"
    if v == float("-inf"):
        return "-INF"
    if v == 0.0:
        return "0.0000"
    if 1e-3 <= abs(v) < 1e5:
        return f"{v:.4f}"
    return f"{v:.2e}"


def _fmt_setting(value):
    """One recorded setting level, printed the way its Python type reads."""
    if isinstance(value, float):
        return _fmt_num(value)
    return str(value)


def _fmt_status(term, short=False):
    """hw1:Pass / hw1:Fail / absent -> "Pass" | "Fail" | "-" (or P / F / - ).

    Compared against the two hw1:Status individuals (§4.5), never against a
    string: a status is a term, and `str(term).endswith("Pass")` would also match
    an IRI from some other vocabulary that happens to end that way.
    """
    if term == HW1.Pass:
        return "P" if short else "Pass"
    if term == HW1.Fail:
        return "F" if short else "Fail"
    if term is None:
        return "-"
    return _local(term)


def _abbrev(names, width=_COL_ABBREV):
    """{name: short name} for column headers, or identity if truncation collides."""
    short = {n: n[:width] for n in names}
    if len(set(short.values())) != len(short):
        return {n: n for n in names}
    return short


def _annotation_kind(ann):
    """Annotation IRI -> its modality segment ("rgb" | "depth"), or None.

    The other direction of `annotation_iri`, and narrow on purpose: §8.2's
    `read_experiment` dict does not carry the modality, but §7.1's per-frame
    table asks for a per-modality `hw1:qualificationStatus` column. Reading the
    frozen last segment (§2) is the same one-line rule `_modalities_of` already
    applies; it is not a second implementation of the frame-index tail parse,
    which stays in `frame_index_from_iri`.
    """
    text = str(ann)
    for kind in _ANNOTATION_KINDS:
        if text.endswith("/" + kind):
            return kind
    return None


# ── view dispatch: by CONTENT, never by filename (§7.1) ──────────
def _classify(path):
    """What kind of thing this is: "batch" | "declaration" | "experiment".

    BY CONTENT, not by name. A capture directory (rgb/ + depth/) is a batch;
    a `hw1:Batch` subject is a (deprecated) batch.ttl; a `hw1:Experiment`
    subject is an experiment file, assessed iff the file carries
    `MACHINE_MARKER` (§3.1). Anything else is an error that NAMES what it
    found, because "unrecognised file" with no evidence is the least useful
    error a reader can get.
    """
    if _is_capture_dir(path):
        return "batch"
    text = _read_text(path)
    g = Graph()
    try:
        g.parse(data=text, format="turtle")
    except Exception as exc:
        raise ValueError(f"{path}: not valid Turtle: {exc}") from None

    batches = sorted(set(g.subjects(RDF.type, HW1.Batch)), key=str)
    exps = sorted(set(g.subjects(RDF.type, HW1.Experiment)), key=str)
    assessed = _marker_offset(text) is not None

    if batches and exps:
        # v5 assessed files intentionally embed the Batch snapshot so a single
        # Turtle file is self-contained for SPARQL. Legacy v4 batch sidecars and
        # experiment files remain mutually exclusive.
        version = g.value(exps[0], HW1.schemaVersion)
        has_factors = any(g.subjects(HW1.inExperiment, exps[0]))
        if (version is not None and str(version).startswith("5")) or has_factors:
            return "experiment" if assessed else "declaration"
        raise ValueError(
            f"{path}: this file declares BOTH a hw1:Batch ({batches[0]}) and a "
            f"hw1:Experiment ({exps[0]}). §3 keeps them in separate "
            f"files — structure in the capture directory, measurement in "
            f"hw1/experiments/<expname>.ttl — so `explore` cannot tell which view "
            f"you want.")
    if batches:
        return "batch"
    if exps:
        return "experiment" if assessed else "declaration"

    types = sorted({_local(o) for o in g.objects(None, RDF.type)})
    raise ValueError(
        f"{path}: neither a capture directory nor an experiment file. `explore` "
        f"needs a directory with rgb/ + depth/ (the batch view) or a subject typed "
        f"hw1:Experiment (the declaration / assessed views). This file declares "
        + (f"{len(g)} triple(s) with rdf:type " + ", ".join('hw1:' + t for t in types)
           if types else f"{len(g)} triple(s) and no rdf:type at all")
        + ".")


# ── the batch view (§7.1 view 1) ────────────────────────────────
def _generation_from_batch_graph(g, b, path):
    """[(param_local, value), ...] from hw1:hasGenerationSetting on a Batch node."""
    generation = []
    for s in sorted(g.objects(b, HW1.hasGenerationSetting), key=str):
        param = g.value(s, HW1.settingParameter)
        lit = g.value(s, HW1.settingValue)
        local = _local(param) if param is not None else "(no settingParameter)"
        value = (_setting_value_to_python(lit, path, local) if lit is not None
                 else None)
        generation.append((local, value))
    return generation


def _batch_facts_from_ttl(path):
    """Batch header/table facts from a (deprecated) batch.ttl sidecar."""
    g, b, name, stored_path, floor = _parse_batch_ttl(path)
    frames = []
    for f in g.objects(b, HW1.hasFrame):
        idx_lit = g.value(f, HW1.frameIndex)
        idx = int(idx_lit) if idx_lit is not None else frame_index_from_iri(f)
        rgb = g.value(f, HW1.hasRGBImage)
        depth = g.value(f, HW1.hasDepthImage)
        frames.append((idx,
                       g.value(rgb, SCHEMA.contentUrl) if rgb is not None else None,
                       g.value(depth, SCHEMA.contentUrl) if depth is not None else None))
    frames.sort(key=lambda row: row[0])
    stems = [idx for idx, _, _ in frames]
    gaps = [(i, j) for i, j in zip(stems, stems[1:]) if j - i != 1]
    return {"path": path,
            "iri": b,
            "name": name,
            "batch_path": stored_path,
            "floor": floor,
            "frames": frames,
            "gaps": gaps,
            "generation": _generation_from_batch_graph(g, b, path),
            "derived_from": g.value(b, PROV.wasDerivedFrom)}


def _batch_facts_from_capture(data_dir, declared_name=None):
    """Batch header/table facts from the rasters on disk.

    An optional `<data_dir>/batch.ttl` sidecar still supplies generation
    provenance; the frame list always comes from `_pair_frames`.
    """
    sidecar = _sidecar_batch_ttl(data_dir)
    generation, derived_from, floor, name, iri = [], None, None, None, None
    if sidecar is not None:
        try:
            sidecar_facts = _batch_facts_from_ttl(sidecar)
            generation = sidecar_facts["generation"]
            derived_from = sidecar_facts["derived_from"]
            floor = sidecar_facts["floor"]
            name = sidecar_facts["name"]
            iri = sidecar_facts["iri"]
        except Exception:                                # noqa: BLE001 — sidecar is optional
            pass
    frames = [(int(stem), rgb, depth) for stem, rgb, depth in _pair_frames(data_dir)]
    if declared_name is not None:
        name = declared_name
        iri = batch_iri(declared_name)
        try:
            floor = _floor_from_batch_name(declared_name)
        except ValueError:
            pass
    elif name is None:
        floor = 1 if floor is None else int(floor)
        name = batch_name(data_dir, floor)
        iri = batch_iri(name)
    stems = [idx for idx, _, _ in frames]
    gaps = [(i, j) for i, j in zip(stems, stems[1:]) if j - i != 1]
    return {"path": sidecar if sidecar is not None else data_dir,
            "iri": iri,
            "name": name,
            "batch_path": data_dir,
            "floor": floor,
            "frames": frames,
            "gaps": gaps,
            "generation": generation,
            "derived_from": derived_from}


def _batch_facts(path, declared_name=None):
    """Everything the batch header and frame table need.

    Accepts a capture directory (preferred) or a batch.ttl (deprecated sidecar).
    Shared by view 1 (the batch itself) and view 2 (the capture a declaration
    names in `hw1:batchFile`), so the two print the SAME header from the same
    code.
    """
    resolved = path if os.path.isabs(path) else os.path.abspath(path)
    if _is_capture_dir(resolved):
        return _batch_facts_from_capture(resolved, declared_name=declared_name)
    if os.path.isfile(resolved):
        return _batch_facts_from_ttl(resolved)
    raise ValueError(
        f"{path}: not a capture directory (rgb/ + depth/) and not a batch.ttl "
        f"({resolved!r}).")


def _print_batch_header(facts):
    """The batch header of §7.1: name, floor, path, frame count, generation settings."""
    _kv("batch", facts["name"])
    _kv("floor", facts["floor"])
    _kv("batchPath", facts["batch_path"])
    if os.path.isfile(str(facts["path"])):
        _kv("batch.ttl", facts["path"])
    stems = [idx for idx, _, _ in facts["frames"]]
    span = f", stems {stems[0]}..{stems[-1]}" if stems else ""
    _kv("frames", f"{len(stems)}{span}")
    if facts["derived_from"] is not None:
        _kv("derivedFrom", _fmt_term(facts["derived_from"]))
    if facts["generation"]:
        _kv("generation", ", ".join(f"{k}={_fmt_setting(v)}"
                                    for k, v in facts["generation"]))
    else:
        # Absence is a claim about nothing, not a claim of cleanliness (§5), and
        # it is also why the Generation verdict cannot fire on the shipped
        # corrupted captures (§12 O-F). Say so here rather than let a reader
        # infer "uncorrupted" from a blank line.
        _kv("generation", "none recorded — NOT ASSERTED, not 'uncorrupted' "
                          "(§5)")

    gaps = facts["gaps"]
    if gaps:
        shown = ", ".join(f"{i}->{j} ({j - i - 1} stem(s) missing)" for i, j in gaps)
        _kv("stem gaps", f"{len(gaps)}: {shown}")
        _note("every experiment over this batch pairs ACROSS those gaps, so those "
              "pairs span more than one capture step and their observables read as "
              "larger motion.", indent=_INDENT + " " * 20)
    else:
        _kv("stem gaps", "none — the paired stems are consecutive")


def _view_batch(path):
    """View 1: batch header + the frame table, with stem gaps called out."""
    facts = _batch_facts(path)
    _rule(f"BATCH  {facts['name']}")
    _print_batch_header(facts)

    _sub("frames")
    rows = [(idx, rgb, depth) for idx, rgb, depth in facts["frames"]]
    _print_table(("frame", "rgb", "depth"), rows, right=(0,))
    _note(f"{len(rows)} frame(s), all shown. A capture holds no measured value by "
          f"contract (§4.1); write a declaration naming it in hw1:batchFile and run "
          f"`api.py experiment` to measure one.")
    return 0


# ── the declaration view (§7.1 view 2) ──────────────────────────
def _selection_rows(selected, factors, settings):
    """One row per selected factor: what it measures, which way, against which number."""
    rows = []
    for f in selected:
        info = factors[f]
        threshold = settings.get(info["qualifiedBy"])
        rows.append((f, "hw1:" + info["over"],
                     "higher is better" if info["polarity"] == "higher"
                     else "lower is better",
                     info["qualifiedBy"],
                     "-" if threshold is None else _fmt_setting(threshold)))
    return rows


def _settings_rows(names, values, decls, source_of):
    """One row per recorded/required parameter: value, role, primary factor, source."""
    rows = []
    for param in sorted(names):
        d = decls.get(param)
        rows.append((param,
                     _fmt_setting(values[param]),
                     d["role"] if d else "(undeclared)",
                     d["primary"] if d else "-",
                     source_of(param)))
    return rows


def _view_declaration(path):
    """View 2: an unassessed declaration — batch header, selection, settings table."""
    decl = read_declaration(path)
    decls = load_parameter_declarations(_ONTOLOGY_TTL)
    factors = load_quality_factors(_ONTOLOGY_TTL)

    _rule(f"DECLARATION  {decl['exp_name']}  (not yet assessed)")
    _kv("file", path)
    _kv("experiment IRI", decl["exp_iri"])
    _kv("state", "no MACHINE SECTION marker — no values and no verdicts exist yet")
    if "PREDICTION" not in _read_text(path).upper():
        _note("Add a `# PREDICTION` comment before assessment: expected input "
              "Fails, baseline outcome, selected outcome, and the mechanism that "
              "would produce that differential. The declaration digest seals it "
              "with the treatment (§14).")

    _sub("batch (via hw1:batchFile)")
    _print_batch_header(_batch_facts(decl["batch_path"],
                                    declared_name=decl["batch_name"]))

    _sub(f"selection — {len(decl['selected'])} factor(s) of the "
         f"{len(_selectable_factors(factors))}-factor menu")
    _print_table(("factor", "observable", "polarity", "qualifiedBy", "threshold"),
                 _selection_rows(decl["selected"], factors, decl["required"]))

    _sub("settings — the required set of §4.2, total over THIS selection")
    given = decl["given"]
    _print_table(
        ("parameter", "value", "role", "primary factor", "source"),
        _settings_rows(decl["required"], decl["required"], decls,
                       lambda p: "declared" if p in given else "WILL BE DEFAULTED"),
        right=(1,))
    _note(f"{len(given)} declared, {len(decl['required']) - len(given)} will be "
          f"filled from hw1:paramDefault and recorded below the marker. Generation "
          f"parameters are never in this set: they live on the batch and are never "
          f"defaulted (§5).")
    print(f"{_INDENT}Next: `api.py experiment {path}` — once, ever (§3.1).")
    return 0


# ── the assessed view (§7.1 view 3) ─────────────────────────────
def _section_settings(path, exp_iri):
    """({param: value} above the marker, {param: value} below it). The §4.2 split.

    Given-versus-defaulted is recorded by WHICH SIDE OF THE MARKER a setting sits
    on (§4.2) — it is not derivable from the setting node itself, because a
    student's setting may be a blank node and a machine-minted one uses the §2
    scheme IRI, and neither shape is a promise about who wrote it. So the two
    halves are parsed separately and read by the SAME reader
    (`_experiment_settings`), which is also what keeps the "one parameter, one
    level" check applying to both.
    """
    student_text, machine_text = _split_sections(_read_text(path))
    sides = []
    for text in (student_text, machine_text or ""):
        g = Graph()
        g.parse(data=text, format="turtle")
        sides.append(_experiment_settings(g, exp_iri, path))
    return sides[0], sides[1]


def _observable_placement(g, exp, factors, selected):
    """{factor: "frame" | "pair"} — read off WHERE each observable actually landed.

    Generic by construction: the placement table of `build_machine_graph` is not
    consulted and no factor is named, so a menu factor added to the TBox and
    measured onto either node class appears in the right column group with no
    edit here. A selected factor whose observable is nowhere in the file falls
    back to the frame group and prints "-" in every row, which is visible rather
    than silent.
    """
    on_ann, on_pair = set(), set()
    for ann in g.objects(exp, HW1.producesAnnotation):
        on_ann.update(_local(p) for p in g.predicates(ann, None))
    for pair in g.objects(exp, HW1.producesPair):
        on_pair.update(_local(p) for p in g.predicates(pair, None))
    placement = {}
    for f in selected:
        over = factors[f]["over"]
        placement[f] = "pair" if (over in on_pair and over not in on_ann) else "frame"
    return placement


def _collect_frame_rows(g, exp, factors, selected, placement, path):
    """The per-frame table's data: annotations by frame+modality, pairs by source frame.

    Values are found by the factor's `hw1:overProperty` and statuses by its
    `hw1:statusProperty`, both read from the TBox — never by concatenating
    "Status" onto a name and never by branching on which factor it is.
    """
    ann_values, ann_aggregate, means, kinds = {}, {}, {}, []
    for ann in g.objects(exp, HW1.producesAnnotation):
        idx = _annotation_frame_index(g, ann, path)
        kind = _annotation_kind(ann)
        if kind is not None and kind not in kinds:
            kinds.append(kind)
        ann_aggregate[(idx, kind)] = g.value(ann, HW1.qualificationStatus)
        for f in selected:
            over = factors[f]["over"]
            value = g.value(ann, HW1[over])
            if value is not None:
                ann_values[(idx, f)] = (value, g.value(ann, HW1[factors[f]["status"]]))
        mean = g.value(ann, HW1.meanValue)
        if mean is not None:
            means[idx] = mean

    pairs = {}
    for pair in g.objects(exp, HW1.producesPair):
        i = frame_index_from_iri(g.value(pair, HW1.sourceFrame))
        j = frame_index_from_iri(g.value(pair, HW1.targetFrame))
        cells = {}
        for f in selected:
            if placement[f] != "pair":
                continue
            value = g.value(pair, HW1[factors[f]["over"]])
            cells[f] = (value, g.value(pair, HW1[factors[f]["status"]]))
        pairs[i] = (j, cells, g.value(pair, HW1.qualificationStatus))

    kinds = [k for k in _ANNOTATION_KINDS if k in kinds]
    return ann_values, ann_aggregate, means, kinds, pairs


def _print_frame_table(exp, g, factors, selected, placement, path):
    """§7.1's per-frame table. EVERY ROW, narrow columns — see the section header."""
    ann_values, ann_agg, means, kinds, pairs = _collect_frame_rows(
        g, exp["exp_iri"], factors, selected, placement, path)

    frame_factors = [f for f in selected if placement[f] == "frame"]
    pair_factors = [f for f in selected if placement[f] == "pair"]
    short = _abbrev([factors[f]["over"] for f in selected])

    print(f"{_INDENT}columns (value and status merged; P = hw1:Pass, F = hw1:Fail, "
          f"- = not measured on this node):")
    for f in selected:
        info = factors[f]
        op = "<=" if info["polarity"] == "lower" else ">="
        print(f"{_INDENT}  {short[info['over']]:<12} hw1:{info['over']} — {f}, "
              f"Pass iff value {op} {info['qualifiedBy']} "
              f"= {_fmt_setting(exp['settings'][info['qualifiedBy']])}")
    if means:
        # The statusless baseline of §4.3, printed next to the clip factors
        # precisely so "did a clip factor beat the mean?" is a lookup (§7.2).
        print(f"{_INDENT}  {'meanValue':<12} hw1:meanValue — the BASELINE: no factor, "
              f"no threshold, deliberately NO status")

    headers = ["frame"]
    right = [0]
    for f in frame_factors:
        headers.append(short[factors[f]["over"]])
    if means:
        headers.append("meanValue")
    for kind in kinds:
        headers.append(kind + "QS")
    if pairs:
        headers.append("->next")
        right.append(len(headers) - 1)
        for f in pair_factors:
            headers.append(short[factors[f]["over"]])
        headers.append("pairQS")

    def cell(entry):
        if entry is None:
            return "-"
        value, status = entry
        return f"{_fmt_num(value)} {_fmt_status(status, short=True)}"

    rows = []
    for idx in sorted(exp["frame_status"]):
        row = [idx]
        for f in frame_factors:
            row.append(cell(ann_values.get((idx, f))))
        if means:
            row.append(_fmt_num(means.get(idx)))
        for kind in kinds:
            row.append(_fmt_status(ann_agg.get((idx, kind)), short=True))
        if pairs:
            step = pairs.get(idx)
            if step is None:
                row.append("-")
                row.extend("-" for _ in pair_factors)
                row.append("-")
            else:
                j, cells, agg = step
                row.append(j)
                for f in pair_factors:
                    row.append(cell(cells.get(f)))
                row.append(_fmt_status(agg, short=True))
        rows.append(row)

    _print_table(headers, rows, right=tuple(right))
    _note(f"{len(rows)} frame(s), all shown — nothing elided. "
          f"`… | grep ' F'` finds every failing row.")


def _factor_counts(g, factors, selected):
    """Pass/total per selected factor, counted over the nodes that carry its value."""
    rows = []
    for f in sorted(selected):
        info = factors[f]
        total = passed = 0
        for node in g.subjects(HW1[info["over"]], None):
            total += 1
            if g.value(node, HW1[info["status"]]) == HW1.Pass:
                passed += 1
        rows.append((f, "hw1:" + info["over"], f"{passed}/{total}",
                     f"{total - passed}"))
    return rows


def _run_nodes(g, exp_iri):
    """Every hw1:ReconstructionRun of this experiment, in a stable mode order."""
    runs = []
    for run in g.objects(exp_iri, HW1.hasRun):
        mode = g.value(run, HW1.selectionMode)
        runs.append((_local(mode) if mode is not None else "?", run))
    runs.sort(key=lambda row: str(row[1]))
    return runs


def _run_factor_locals(g, factors, run):
    """The factors that actually landed on one run node — TBox-resolved, not listed."""
    return sorted(f for f, info in factors.items()
                  if g.value(run, HW1[info["over"]]) is not None)


def _print_summary(exp, g, factors, selected):
    """§7.1's summary: pass counts, usable links, segments, runs."""
    _print_table(("factor", "observable", "Pass", "Fail"),
                 _factor_counts(g, factors, selected), right=(2, 3))

    frames = exp["frame_status"]
    pair_status = exp["pair_status"]
    usable = exp["usable_links"]
    print()
    _kv("frames usable", f"{sum(1 for v in frames.values() if v)}/{len(frames)} "
                         f"(every annotation of the frame Pass; vacuously usable "
                         f"with no annotation)")
    _kv("pairs Pass", f"{sum(1 for v in pair_status.values() if v)}/"
                      f"{len(pair_status)}")
    _kv("usable links", f"{len(usable)}/{len(pair_status)} "
                        f"(pair Pass AND both endpoint frames usable — §4.5)")

    segments = cut_contiguous_segments(usable)
    kept = sum(len(s) for s in segments)
    _kv("segments", f"{len(segments)} maximal segment(s), "
                    f"{kept} frame(s) kept of {len(frames)}")
    if segments:
        shown = ", ".join(f"{s[0]}-{s[-1]}({len(s)})" for s in segments)
        print(f"{_INDENT}{'':<20}{shown}")

    _sub("runs")
    runs = _run_nodes(g, exp["exp_iri"])
    if not runs:
        _note("no hw1:ReconstructionRun yet. Run `reconstruct.py --data_root "
              "<capture> --experiment <this file>`; runs are the only mutation an "
              "assessed file accepts (§3.1).")
        return
    value_locals = sorted({v for _, run in runs
                           for v in _run_factor_locals(g, factors, run)})
    metadata_locals = ["gatedSteps", "spliceCount", "maxGapLength"]
    shown_metadata = [p for p in metadata_locals
                      if any(g.value(run, HW1[p]) is not None for _, run in runs)]
    headers = (["run", "selectionMode", "frames"] +
               [factors[f]["over"] for f in value_locals] + shown_metadata)
    rows = []
    for mode_local, run in runs:
        count = g.value(run, HW1.runFrameCount)
        row = [str(run).rsplit("/", 1)[-1], mode_local,
               "-" if count is None else int(count)]
        for f in value_locals:
            info = factors[f]
            value = g.value(run, HW1[info["over"]])
            status = g.value(run, HW1[info["status"]])
            row.append("-" if value is None
                       else f"{_fmt_num(value)} {_fmt_status(status)}")
        for prop in shown_metadata:
            value = g.value(run, HW1[prop])
            row.append("-" if value is None else int(value))
        rows.append(row)
    _print_table(headers, rows, right=(2,))


def _scoped_subjects(g, prop, value, this_run):
    """Subjects carrying `prop` (optionally == `value`), scoped to ONE run.

    Run nodes OTHER than the run being attributed are skipped, so a failing
    baseline does not turn up inside the selected run's attribution and vice
    versa. The test is on rdf:type, not on the factor, so it stays generic — and
    it is applied to the failing count and the total alike, so the two numbers
    count the same population.
    """
    nodes = []
    for node in g.subjects(prop, value):
        if node != this_run and (node, RDF.type, HW1.ReconstructionRun) in g:
            continue
        nodes.append(node)
    return nodes


def _failing_nodes(g, factors, factor_local, this_run):
    """Every node whose `hw1:statusProperty` for this factor reads hw1:Fail.

    THE GENERIC RESOLUTION, and the whole reason the TBox keeps the
    `statusProperty` wiring remains useful alongside SPARQL: this is
    `?factor hw1:statusProperty ?sp . ?node ?sp hw1:Fail .` in Python, with no
    predicate enumerated and no factor named. A menu factor added to the TBox is
    picked up here with no edit.
    """
    return _scoped_subjects(g, HW1[factors[factor_local]["status"]], HW1.Fail,
                            this_run)


def _culprit_rows(failing, recorded, decls, where_of):
    """Recorded settings whose parameter touches a failing factor. TBox traversal only.

    `hw1:settingForFactor` on the setting node is NOT consulted — it is
    single-valued on an experiment setting (§4.5) and still multi-valued on a
    batch-side generation setting, so trusting it would give two different
    answers to one question. The blame set is
    `settingParameter -> paramPrimaryFactor / paramAffectsFactor`, read from the
    TBox, which is what reaches `brightnessGain` for a ShadowClipping failure
    whose primary factor is HighlightClipping.
    """
    rows = []
    for param in sorted(recorded):
        d = decls.get(param)
        if d is None:
            continue
        touched = sorted(set((d["primary"],) + tuple(d["affects"])) & set(failing))
        if not touched:
            continue
        rows.append((param, _fmt_setting(recorded[param]), d["role"],
                     where_of(param), ", ".join(touched),
                     _ROLE_FIX.get(d["role"], "(no fix declared for this role)")))
    return rows


def _print_verdict(exp, g, factors, decls, given, path):
    """§7.1's VERDICT SECTION — the old failure_attribution.rq, computed in code.

    For every run carrying a Fail: the factors that failed anywhere in this
    experiment with their failing-node counts, then every setting whose
    parameter touches one of them, with the ROLE as the verdict and the role's
    fix spelled out. Nothing here branches on a factor or a parameter name.
    """
    _rule("VERDICT — what failed, and what to fix")
    runs = _run_nodes(g, exp["exp_iri"])
    if not runs:
        bad = sum(1 for f, info in factors.items()
                  if any(True for _ in g.subjects(HW1[info["status"]], HW1.Fail)))
        _note(f"No hw1:ReconstructionRun in this file, so there is nothing to "
              f"attribute yet: attribution starts from a FAILING RUN and walks back "
              f"to the input factors ({bad} factor(s) already carry at least one Fail "
              f"above). Run `reconstruct.py --experiment {path}` first.")
        return

    # Run modes form interventions. Say what the differential proves
    # before walking from input Fails to their setting roles; that older walk is
    # true about the pixels but cannot, by itself, explain a regression caused by
    # deleting temporal links.
    by_tail = {str(run).rsplit("/", 1)[-1]: run for _, run in runs}

    def run_verdict(run):
        statuses = [g.value(run, HW1[info["status"]]) for info in factors.values()
                    if g.value(run, HW1[info["over"]]) is not None]
        statuses = [s for s in statuses if s is not None]
        return None if not statuses else all(s == HW1.Pass for s in statuses)

    baseline = by_tail.get("baseline")
    selected_run = by_tail.get("selected")
    if baseline is not None and selected_run is not None:
        b_ok, s_ok = run_verdict(baseline), run_verdict(selected_run)
        splices = g.value(selected_run, HW1.spliceCount)
        gap = g.value(selected_run, HW1.maxGapLength)
        gated = g.value(selected_run, HW1.gatedSteps)
        mechanism = (f"{int(splices) if splices is not None else '?'} splice(s), "
                     f"max gap {int(gap) if gap is not None else '?'} frame(s), "
                     f"{int(gated) if gated is not None else '?'} gated step(s)")
        if b_ok is True and s_ok is False:
            _note("SELECTION EFFECT — the full-batch baseline passed and the "
                  f"selected probe failed ({mechanism}). Deletion/splicing is the "
                  "observed treatment difference, so deletion is not a repair. "
                  "The candidate table below scopes true INPUT Fails; it does not "
                  "explain this run regression. Act through the setting roles.")
        elif b_ok is False and s_ok is True:
            _note("SELECTION EFFECT — the full-batch baseline failed and the "
                  f"selected probe passed ({mechanism}). Under this experiment, "
                  "the rejected inputs are load-bearing for geometric ICP; use the "
                  "culprit setting roles below to design the next intervention.")
        elif b_ok is not None and s_ok is not None:
            b_l2 = g.value(baseline, HW1.mapMeanL2)
            s_l2 = g.value(selected_run, HW1.mapMeanL2)
            raw = ""
            if b_l2 is not None and s_l2 is not None:
                b_value, s_value = float(b_l2), float(s_l2)
                ratio = s_value / b_value if b_value else float("inf")
                raw = (f" Continuous mapMeanL2 still changed {b_value:.4f} -> "
                       f"{s_value:.4f} m ({ratio:.1f}x); the threshold is not the "
                       f"effect size.")
            _note("SELECTION EFFECT — baseline and selected have the same run verdict "
                  f"({mechanism}). Compare raw outcomes before claiming no effect."
                  + raw)

    attributed = 0
    for mode_local, run in runs:
        run_fails = [f for f, info in factors.items()
                     if g.value(run, HW1[info["status"]]) == HW1.Fail]
        if not run_fails:
            continue
        attributed += 1
        _sub(f"run {str(run).rsplit('/', 1)[-1]} ({mode_local}) FAILED")
        for f in sorted(run_fails):
            info = factors[f]
            print(f"{_INDENT}{f}: hw1:{info['over']} = "
                  f"{_fmt_num(g.value(run, HW1[info['over']]))} vs "
                  f"{info['qualifiedBy']} = "
                  f"{_fmt_setting(exp['settings'].get(info['qualifiedBy']))} -> Fail")

        # ── the failing factors, with their failing-node counts ───────────────
        failing, rows = [], []
        for f in sorted(factors):
            nodes = _failing_nodes(g, factors, f, run)
            if not nodes:
                continue
            failing.append(f)
            total = len(_scoped_subjects(g, HW1[factors[f]["over"]], None, run))
            rows.append((f, "hw1:" + factors[f]["status"], len(nodes),
                         total, "yes" if f in exp["selected"] else
                         ("run factor" if f not in _selectable_factors(factors)
                          else "no")))
        print()
        _note("factors carrying hw1:Fail (found through hw1:statusProperty — no "
              "predicate is enumerated, so a menu factor added to the TBox needs no "
              "edit here):")
        _print_table(("factor", "status property", "Fail", "of nodes", "selected"),
                     rows, right=(2, 3))

        # ── the culprit settings, role first ──────────────────────────────────
        recorded = dict(exp["settings"])
        culprits = _culprit_rows(
            failing, recorded, decls,
            lambda p: "declaration" if p in given else "defaulted")

        gen_rows, gen_note = [], None
        batch_file = g.value(exp["exp_iri"], HW1.batchFile)
        if batch_file is not None:
            resolved = _resolve_batch_file(str(batch_file))
            try:
                facts = _batch_facts(resolved)
                gen_rows = _culprit_rows(
                    failing, dict(facts["generation"]), decls,
                    lambda p: "batch (--gen)")
                if not facts["generation"]:
                    would = sorted(
                        n for n, d in decls.items()
                        if d["role"] == "GenerationSetting"
                        and set((d["primary"],) + tuple(d["affects"])) & set(failing))
                    if would:
                        gen_note = (
                            f"the batch records NO hw1:GenerationSetting, so the "
                            f"Generation verdict cannot fire here. "
                            f"{', '.join(would)} would be the culprit(s) — each "
                            f"touches a failing factor — but nothing recorded a "
                            f"level. NEVER INVENT ONE (§12 O-F): "
                            f"recover the --gen values the capture was made with, "
                            f"or regenerate it.")
            except (ValueError, OSError):
                gen_note = (f"hw1:batchFile {str(batch_file)!r} does not resolve "
                            f"({resolved!r}), so the batch's GenerationSettings could "
                            f"not be read from here.")

        print()
        _note("candidate settings — every recorded setting whose PARAMETER's "
              "hw1:paramPrimaryFactor or hw1:paramAffectsFactor is one of those "
              "factors (TBox traversal, §4.5). This graph walk establishes scope, "
              "not causal relevance to the run; use a matched-setting dataset "
              "contrast or a one-setting experiment to establish that:")
        _print_table(
            ("parameter", "value", "role", "recorded on", "failing factor(s)", "FIX"),
            culprits + gen_rows, right=(1,))
        if gen_note:
            _note(f"note: {gen_note}")
        _note("The ROLE is the verdict. Measurement and Qualification fixes both mean "
              "a NEW declaration under a NEW name: experiments are write-once (§3.1).")

    if attributed == 0:
        _note("Every run in this file passed every run-level factor, so there is "
              "nothing to attribute. (Input factors may still carry Fails — see the "
              "summary above; a selection that drops frames is a design choice, not a "
              "failure.)")


def _view_experiment(path):
    """Terminal view of one assessed experiment. Full strategy note: docs/triplestore.md."""
    # `read_experiment` verifies the §3.1 seal FIRST. A file edited above the
    # marker raises here, which is the entire point: printing a table of numbers
    # computed for a treatment the file no longer declares would be the most
    # misleading artefact this program could produce.
    exp = read_experiment(path)
    decls = load_parameter_declarations(_ONTOLOGY_TTL)
    factors = load_quality_factors(_ONTOLOGY_TTL)
    g = exp["graph"]
    selected = exp["selected"]

    _rule(f"EXPERIMENT  {exp['exp_name']}  (assessed)")
    _kv("file", path)
    _kv("experiment IRI", exp["exp_iri"])
    _kv("batch", exp["batch_name"])
    _kv("evaluates", f"{len(selected)} factor(s): {', '.join(selected)}")
    label = g.value(exp["exp_iri"], RDFS.label)
    if label is not None:
        _kv("label", str(label))
    _kv("seal", f"hw1:declarationDigest verified "
                f"({str(g.value(exp['exp_iri'], HW1.declarationDigest))[:12]}…)")

    _sub("settings — given vs defaulted, read from WHICH SIDE OF THE MARKER (§4.2)")
    given, defaulted = _section_settings(path, exp["exp_iri"])
    _print_table(
        ("parameter", "value", "role", "primary factor", "source"),
        _settings_rows(exp["settings"], exp["settings"], decls,
                       lambda p: ("declaration (above marker)" if p in given else
                                  "defaulted (below marker)" if p in defaulted else
                                  "(not in either section)")),
        right=(1,))
    print(f"{_INDENT}{len(given)} declared, {len(defaulted)} defaulted.")

    _sub("selection")
    _print_table(("factor", "observable", "polarity", "qualifiedBy", "threshold"),
                 _selection_rows(selected, factors, exp["settings"]))

    _sub("per frame")
    placement = _observable_placement(g, exp["exp_iri"], factors, selected)
    _print_frame_table(exp, g, factors, selected, placement, path)

    _sub("summary")
    _print_summary(exp, g, factors, selected)

    _sub("filter masks")
    if not exp["mask_files"]:
        _note("no exported factor masks in this experiment")
    else:
        rows = []
        for factor, scopes in sorted(exp["mask_files"].items()):
            sample = next(iter(scopes["frames"].values()), None)
            if sample is None:
                sample = next(iter(scopes["pairs"].values()), "-")
            rows.append((factor, len(scopes["frames"]), len(scopes["pairs"]), sample))
        _print_table(("factor", "frame masks", "pair masks", "example maskFile"),
                     rows, right=(1, 2))

    _print_verdict(exp, g, factors, decls, given, path)
    return 0


# ── the comparison view (§7.1 view 4) ───────────────────────────
def _compare_column(path):
    """One column of the comparison table, from whichever reader the file supports."""
    kind = _classify(path)
    if kind == "batch":
        # Named by hw1:batchName, not by the file name: every batch file in this
        # project is called `batch.ttl` (§3), so a basename would label two
        # columns identically.
        return {"path": path, "kind": "batch", "name": str(_batch_facts(path)["name"]),
                "why": "a batch file carries no selection, no settings and no runs "
                       "(§4.1), so there is nothing to compare it on"}
    if kind == "declaration":
        decl = read_declaration(path)
        return {"path": path, "kind": "declaration", "name": decl["exp_name"],
                "batch": decl["batch_name"], "selected": decl["selected"],
                "settings": decl["required"], "given": decl["given"],
                "runs": {}, "counts": None,
                "why": "declared but NOT ASSESSED — its settings are the required "
                       "set §4.2 would record, and it has no values, verdicts or "
                       "runs to compare"}
    exp = read_experiment(path)
    g = exp["graph"]
    factors = load_quality_factors(_ONTOLOGY_TTL)
    runs = {}
    for _mode_local, run in _run_nodes(g, exp["exp_iri"]):
        # Keyed by the run IRI's `<mode>` segment (§2) — "baseline" / "selected",
        # the words the CLI and the student use — not by the hw1:SelectionMode
        # individual, which is the same fact spelled for the ontology.
        mode = str(run).rsplit("/", 1)[-1]
        for f in _run_factor_locals(g, factors, run):
            info = factors[f]
            runs[(mode, info["over"])] = (
                f"{_fmt_num(g.value(run, HW1[info['over']]))} "
                f"{_fmt_status(g.value(run, HW1[info['status']]))}")
    given, _defaulted = _section_settings(path, exp["exp_iri"])
    return {"path": path, "kind": "experiment", "name": exp["exp_name"],
            "batch": exp["batch_name"], "selected": exp["selected"],
            "settings": exp["settings"], "given": given, "runs": runs,
            "factor_counts": {
                factor: (
                    sum(1 for node in g.subjects(HW1[info["over"]], None)
                        if g.value(node, HW1[info["status"]]) == HW1.Pass),
                    sum(1 for _ in g.subjects(HW1[info["over"]], None)))
                for factor, info in factors.items() if factor in exp["selected"]},
            "counts": (sum(1 for v in exp["frame_status"].values() if v),
                       len(exp["frame_status"]), len(exp["usable_links"]),
                       len(exp["pair_status"])),
            "why": None}


def _view_compare(paths):
    """View 4: several files side by side — selection, settings and runs."""
    columns = [_compare_column(p) for p in paths]
    usable = [c for c in columns if c["kind"] != "batch"]
    _rule(f"COMPARE  {len(columns)} file(s)")
    for c in columns:
        _kv(c["name"], f"{c['path']}  [{c['kind']}]"
                       + (f" — {c['why']}" if c["why"] else ""))
    if not usable:
        print(f"{_INDENT}Nothing comparable was given.")
        return 0

    names = [c["name"] for c in usable]
    _sub("identity")
    _print_table(["", *names],
                 [["batch", *[c["batch"] for c in usable]],
                  ["assessed", *["yes" if c["kind"] == "experiment" else "no"
                                 for c in usable]],
                  ["frames usable", *[f"{c['counts'][0]}/{c['counts'][1]}"
                                      if c["counts"] else "-" for c in usable]],
                  ["usable links", *[f"{c['counts'][2]}/{c['counts'][3]}"
                                     if c["counts"] else "-" for c in usable]]])

    _sub("selection (hw1:evaluatesFactor)")
    all_factors = sorted({f for c in usable for f in c["selected"]})
    rows = []
    for f in all_factors:
        marks = ["yes" if f in c["selected"] else "-" for c in usable]
        rows.append([("*" if len(set(marks)) > 1 else " ") + " " + f, *marks])
    _print_table(["  factor", *names], rows)

    _sub("input factor outcomes")
    rows = []
    for factor in all_factors:
        vals = []
        for column in usable:
            counts = column.get("factor_counts", {}).get(factor)
            vals.append("-" if counts is None else f"{counts[0]}/{counts[1]} Pass")
        rows.append([factor, *vals])
    _print_table(["  factor", *names], rows,
                 right=tuple(range(1, len(names) + 1)))

    _sub("settings")
    all_params = sorted({p for c in usable for p in c["settings"]})
    rows = []
    for p in all_params:
        vals = [_fmt_setting(c["settings"][p]) if p in c["settings"] else "-"
                for c in usable]
        annotated = [v + ("" if c["kind"] != "experiment" else
                          (" (d)" if p in c["given"] else ""))
                     for v, c in zip(vals, usable)]
        rows.append([("*" if len(set(vals)) > 1 else " ") + " " + p, *annotated])
    _print_table(["  parameter", *names], rows, right=tuple(range(1, len(names) + 1)))
    _note("* marks a row where the columns disagree — that is the treatment "
          "difference. (d) marks a level the DECLARATION set; everything else was "
          "filled from hw1:paramDefault (§4.2).")

    _sub("runs")
    all_runs = sorted({k for c in usable for k in c["runs"]})
    if not all_runs:
        print(f"{_INDENT}no hw1:ReconstructionRun in any of these files.")
    else:
        rows = []
        for mode_local, over in all_runs:
            vals = [c["runs"].get((mode_local, over), "-") for c in usable]
            rows.append([f"{mode_local} / hw1:{over}", *vals])
        _print_table(["  run / observable", *names], rows,
                     right=tuple(range(1, len(names) + 1)))
    return 0


def _query_rows(result):
    """Return a stable `(headers, rows)` projection for a SPARQL SELECT result."""
    headers = [str(var) for var in result.vars]
    rows = []
    for row in result:
        rows.append(["" if row[var] is None else str(row[var]) for var in result.vars])
    return headers, rows


def cmd_query(args):
    """Run a read-only SPARQL query against one .ttl file. Full strategy note: docs/triplestore.md."""
    if not os.path.isfile(args.path):
        raise ValueError(f"{args.path!r}: query needs a local Turtle file")
    query_text = args.query_text if args.query_text is not None else _read_text(args.query_file)
    graph = Graph()
    try:
        graph.parse(data=_read_text(args.path), format="turtle")
    except Exception as exc:
        raise ValueError(f"{args.path}: not valid Turtle: {exc}") from None
    result = query_graph(graph, query_text)

    if result.type == "SELECT":
        headers, rows = _query_rows(result)
        if args.format == "csv":
            writer = csv.writer(sys.stdout)
            writer.writerow(headers)
            writer.writerows(rows)
        elif args.format == "json":
            print(json.dumps([dict(zip(headers, row)) for row in rows], indent=2))
        else:
            _print_table(headers, rows)
        return 0
    if result.type == "ASK":
        answer = bool(result.askAnswer)
        print(json.dumps({"boolean": answer}) if args.format == "json"
              else str(answer).lower())
        return 0

    # CONSTRUCT and DESCRIBE results are graphs. Turtle is the useful default;
    # `--format json` selects a standards-friendly JSON-LD serialization.
    serialization = "json-ld" if args.format == "json" else "turtle"
    print(result.graph.serialize(format=serialization), end="")
    return 0


def cmd_explore(args):
    """Read-only terminal tables (writes/measures nothing). Full strategy note: docs/triplestore.md."""
    paths = list(args.paths)
    missing = [p for p in paths if not (os.path.isfile(p) or _is_capture_dir(p))]
    if missing:
        raise ValueError(
            f"no such capture directory or file: {', '.join(repr(p) for p in missing)}. "
            f"`explore` reads a capture directory, a declaration or an assessed "
            f"experiment — and several experiment files at once print the "
            f"comparison view (§7.1).")
    if len(paths) > 1:
        return _view_compare(paths)
    kind = _classify(paths[0])
    if kind == "batch":
        return _view_batch(paths[0])
    if kind == "declaration":
        return _view_declaration(paths[0])
    return _view_experiment(paths[0])


# =============================================================================
# Frame selection — the segment cutter
#   In v1 this was the body of a `select` subcommand that minted a derived
#   experiment. Both are deleted (§11): selection is not an artefact,
#   it is a step inside `reconstruct.py`, which calls this function on the
#   `usable_links` that `read_experiment` derived and then records the outcome as
#   `hw1:usedFrame` on the selected run. So this is a pure list-to-lists function
#   with no RDF in it at all — the grading happened upstream, in `status_for`.
# =============================================================================
def cut_contiguous_segments(usable_links):
    """Usable links -> maximal contiguous segments (no length floor). Full strategy note: docs/triplestore.md."""
    links = sorted(set((int(i), int(j)) for i, j in usable_links))
    segments = []
    current = []
    prev_j = None
    for i, j in links:
        if current and prev_j == i:
            current.append(j)                 # the chain continues: one more frame
        else:
            if current:
                segments.append(current)      # hole (or first link): start a segment
            current = [i, j]                  # ONE link is already TWO frames
        prev_j = j
    if current:
        segments.append(current)
    return segments


# =============================================================================
# CLI
# =============================================================================
def _build_parser():
    p = argparse.ArgumentParser(
        description="HW1 data-quality CLI (rdflib + local SPARQL). Scaffold a DECLARATION "
                    "over a capture directory (declare), assess it (experiment), "
                    "and read the result back as terminal tables (explore). "
                    "batch2ttl is a deprecated optional sidecar for generation "
                    "provenance. reconstruct.py completes the suite. SPARQL runs "
                    "locally over .ttl files; no server or daemon is required.",
        epilog="A measured value means nothing without the settings it was "
               "measured and judged under, so the settings live in the same file "
               "as the values — the student's own declaration, which is "
               "WRITE-ONCE.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    b2t = sub.add_parser(
        "batch2ttl",
        help="DEPRECATED. Optional sidecar: write generation provenance to "
             "<data_dir>/batch.ttl. declare/experiment/explore take the capture "
             "directory directly.")
    b2t.add_argument("--data-dir", required=True,
                     help="Directory containing rgb/ and depth/ subdirs of integer-stem .png "
                          "frames. A semantic/ subdir, if present, is tolerated but not "
                          "ingested. batch.ttl is written HERE as an optional sidecar "
                          "for --gen / --derived-from. declare and experiment do not "
                          "need this file.")
    b2t.add_argument("--floor", type=int, default=1,
                     help="Floor this capture is from (default 1). Stored on the Batch node AND "
                          "prefixed onto the batch name, so floor1_mixed_dev and floor2_mixed_dev "
                          "are distinct batches rather than one batch that overwrites itself.")
    b2t.add_argument("--gen", action="append", default=[], metavar="NAME=VALUE",
                     help="One GenerationSetting, repeatable: the corruption ALREADY BAKED "
                          "into these pixels, e.g. --gen brightnessGain=0.45. NAME must be "
                          "declared in ontology/hw1.ttl as a hw1:Parameter whose "
                          "hw1:paramRole is hw1:GenerationSetting; a Measurement or "
                          "Qualification name here is an error (those belong in the "
                          "declaration, as hw1:FactorSetting nodes), and an undeclared name is "
                          "an error with the declared list printed. Applied by NOTHING — "
                          "the capture already embodies it — and never filled from a "
                          "default, because a default would assert something false about "
                          "pixels this code never opened. Absence means NOT ASSERTED.")
    b2t.add_argument("--derived-from", default=None, metavar="BATCHNAME",
                     help="The hw1:batchName of the capture this one was derived from, e.g. "
                          "floor1_baseline for a corrupted copy of it; emits one "
                          "prov:wasDerivedFrom. The only surviving prov: term in the "
                          "project (§6) and the only one that spans two "
                          "batches — there are no derived EXPERIMENTS.")
    b2t.add_argument("--out", default=None,
                     help="Output Turtle path (default <data_dir>/batch.ttl).")
    b2t.set_defaults(func=cmd_batch2ttl)

    dec = sub.add_parser(
        "declare",
        help="Scaffold a declaration Turtle (prefixes, Experiment node, batch join, "
             "factor selection, PREDICTION TODOs) so nobody starts from a blank "
             "page. Writes the STUDENT section only; never assesses, never "
             "overwrites.")
    dec.add_argument("--name", required=True,
                     help="Experiment name: the file stem AND the tail of the "
                          "Experiment IRI, which must stay equal (§2). "
                          "[A-Za-z0-9_-]+ — naming your experimental conditions is "
                          "part of designing them.")
    dec.add_argument("--data-dir", default=None,
                     help="Capture directory containing rgb/ and depth/ subdirs of "
                          "integer-stem .png frames. Written into hw1:batchFile "
                          "verbatim and resolved against the CWD; hw1:onBatch is "
                          "derived from --floor + the directory basename.")
    dec.add_argument("--floor", type=int, default=1,
                     help="Floor this capture is from (default 1). Prefixed onto the "
                          "batch name, so floor1_mixed_dev and floor2_mixed_dev are "
                          "distinct batches.")
    dec.add_argument("--batch-file", default=None,
                     help="DEPRECATED. Path to a capture directory or a legacy "
                          "batch.ttl. Prefer --data-dir; the scaffold writes the "
                          "capture directory into hw1:batchFile either way.")
    dec.add_argument("--factor", action="append", default=[], metavar="FACTOR",
                     help="One menu factor to select, repeatable (with or without the "
                          "hw1: prefix). Omitted entirely: the scaffold selects the "
                          "FULL menu and tells you to trim it — the selection is part "
                          "of the design, so the default is deliberately everything "
                          "rather than a guess.")
    dec.add_argument("--out", default=None,
                     help=f"Output path (default {_EXPERIMENT_DIR}/<name>.ttl). The "
                          f"file stem must equal --name. An existing file is a hard "
                          f"error: the notebook is append-only.")
    dec.set_defaults(func=cmd_declare)

    exp = sub.add_parser(
        "experiment",
        help="Assess ONE student-authored declaration: measure what it selects and "
             "append the machine section to that same file. Once, ever.")
    exp.add_argument("declaration",
                     help=f"Path to the declaration Turtle you wrote (convention: "
                          f"{_EXPERIMENT_DIR}/<expname>.ttl, and the file STEM must equal "
                          f"the tail of the hw1:Experiment IRI inside it). It states the "
                          f"batch (hw1:batchFile = capture directory, hw1:onBatch), "
                          f"the factor selection "
                          f"(hw1:evaluatesFactor, 1..8 from the menu) and any threshold or "
                          f"measurement override (hw1:hasFactorSetting); every value, "
                          f"status and run is COMPUTED and appended below the marker. THE "
                          f"ONLY ARGUMENT: --batch-dir, --floor, --set, --exp-id, --label, "
                          f"--no-pairs and --out are deleted (§7) — the "
                          f"design lives in the RDF now. Running this on an "
                          f"already-assessed file is a HARD ERROR: experiments are "
                          f"write-once, and every tuning is a new declaration under a new "
                          f"name (§3.1).")
    exp.set_defaults(func=cmd_experiment)

    exl = sub.add_parser(
        "explore",
        help="Read-only terminal tables: a batch, a declaration, an assessed "
             "experiment (values, verdicts, runs, attribution) or a comparison.")
    exl.add_argument("paths", nargs="+", metavar="PATH",
                     help="One capture directory, one declaration, one assessed "
                          "experiment — or SEVERAL experiment files, which prints "
                          "the comparison view (§7.1). Writes nothing "
                          "and measures nothing.")
    exl.set_defaults(func=cmd_explore)

    qry = sub.add_parser(
        "query",
        help="Run a read-only SPARQL query over one local Turtle file. Use a .rq "
             "file for reusable student queries; no triplestore is required.")
    qry.add_argument("path", help="Declaration, assessed experiment, or batch Turtle file.")
    source = qry.add_mutually_exclusive_group(required=True)
    source.add_argument("--query-file", metavar="FILE",
                        help="UTF-8 .rq file containing a SPARQL SELECT, ASK, CONSTRUCT, or DESCRIBE query.")
    source.add_argument("--query-text", metavar="SPARQL",
                        help="Inline SPARQL query (quote it in the shell).")
    qry.add_argument("--format", choices=("table", "csv", "json"), default="table",
                     help="SELECT: table (default), csv, or json; graph results: Turtle or JSON-LD.")
    qry.set_defaults(func=cmd_query)

    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
