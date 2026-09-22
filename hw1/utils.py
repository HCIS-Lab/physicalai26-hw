import json
import numpy as np
import open3d as o3d
import os
import cv2
from typing import Any, Dict, Tuple, Union
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

DEPTH_SCALE  = 1000.0     # uint16 depth PNGs store millimetres
# Image axes (+X right, +Y down, +Z forward) to Habitat's sensor axes
# (+X right, +Y up, +Z backward), used by the scorer.
CAM_TO_GT_AXES = np.diag([1.0, -1.0, -1.0])


_IMMERKAER_NORM = 6.0
_MAD_TO_SIGMA = 0.6745

DepthIntrinsics = Union[Dict[str, Any], tuple, list, np.ndarray]


def frame_mean_value(rgb_path: str) -> float:
    """Mean of the value channel over one RGB frame.

    What it does: reads one RGB frame and returns the mean of V = max(R, G, B)
    per pixel. Deliberate weak baseline, not a quality factor: no threshold, no
    polarity, no status.

    Inputs:
        rgb_path: path to an 8-bit RGB PNG.

    Outputs:
        Mean of V in [0, 255].

    Technical foundations:
        V (value channel of HSV without hue/saturation) is the per-pixel
        brightness ceiling. A single mean cannot separate under- from
        over-exposure and is confounded with scene content, which is why the
        clip factors below outperform it.
    """
    # TODO: Implement frame_mean_value.


def frame_clip_hi_fraction(rgb_path: str, tau_hi: float) -> float:
    """Fraction of blown-out pixels in one RGB frame (HighlightClipping).

    What it does: counts pixels with V >= tau_hi and divides by the pixel
    count. Lower is better; passes iff at or below maxClipHiFraction.

    Inputs:
        rgb_path: path to an 8-bit RGB PNG.
        tau_hi: highlight threshold in [0, 255]; has no default because a
            fraction measured at one tau is a different quantity than at
            another (recorded as hw1:tauHi).

    Outputs:
        Blown-out pixel fraction in [0, 1].

    Technical foundations:
        V >= tau_hi holds iff at least one channel is saturated, so one
        railed channel already destroys the pixel's colour. Adapted from the
        unsaturated-region mask of Shin et al., "Camera Exposure Control for
        Robust Robot Vision with Noise-Aware Image Quality Assessment",
        IROS 2019, eq. 7.
    """
    # TODO: Implement frame_clip_hi_fraction.


def frame_clip_lo_fraction(rgb_path: str, tau_lo: float) -> float:
    """Fraction of crushed pixels in one RGB frame (ShadowClipping).

    What it does: counts pixels with V <= tau_lo and divides by the pixel
    count. Lower is better; passes iff at or below maxClipLoFraction.

    Inputs:
        rgb_path: path to an 8-bit RGB PNG.
        tau_lo: shadow threshold in [0, 255]; has no default (recorded as
            hw1:tauLo).

    Outputs:
        Crushed pixel fraction in [0, 1].

    Technical foundations:
        V <= tau_lo holds iff every channel is crushed, so e.g. pure red
        (255, 0, 0) correctly does not count as crushed, unlike a min(R, G, B)
        test. Same Shin et al. (IROS 2019, eq. 7) source as the highlight
        factor; split into two because one number cannot tell a crushed frame
        from a blown-out one.
    """
    # TODO: Implement frame_clip_lo_fraction.


def frame_high_frequency_depth_residual(
        depth_path: str, residual_mask_k: float = 5.0) -> Tuple[float, np.ndarray]:
    """Robust high-frequency depth residual of one depth frame.

    What it does: estimates sensor noise as a robust residual in metres and
    builds a drop mask flagging pixels whose response exceeds residual_mask_k
    times the frame median. Fail-closed: returns inf with an empty mask when
    no fully-valid 3x3 window exists.

    Inputs:
        depth_path: path to a uint16-millimetre depth PNG (0 = invalid).
        residual_mask_k: mask threshold multiplier, finite and >= 0.

    Outputs:
        (value, mask): residual in metres (lower is better); uint8 HxW mask
        with 255 where the pixel should be dropped.

    Technical foundations:
        Immerkaer (1996) estimator: a 3x3 mask (difference of two discrete
        Laplacians) annihilates locally-linear surfaces, leaving the
        high-frequency residual; the median absolute response is scaled by the
        mask norm 6.0 and the MAD-to-sigma constant 0.6745. A 1 mm-response
        floor keeps quantised planar input from turning step edges into noise.
    """
    # TODO: Implement frame_high_frequency_depth_residual.


def frame_flying_pixel_ratio(
        depth_path: str, flying_pixel_window: int = 5,
        flying_pixel_planarity_tol: float = 0.03) -> Tuple[float, np.ndarray]:
    """Fraction of mixed-boundary (flying) pixels in one depth frame.

    What it does: flags a pixel only if its neighbourhood splits into two
    locally planar populations at different depths while the centre belongs
    to neither, then returns the flagged fraction with a drop mask.
    Fail-closed: returns inf with an empty mask when no pixel is valid.

    Inputs:
        depth_path: path to a uint16-millimetre depth PNG (0 = invalid).
        flying_pixel_window: neighbourhood size, odd integer >= 3.
        flying_pixel_planarity_tol: plane tolerance in metres, finite and > 0.

    Outputs:
        (value, mask): flying-pixel fraction in [0, 1] (lower is better);
        uint8 HxW mask with 255 where the pixel should be dropped.

    Technical foundations:
        Window min/max pre-filter selects candidates whose local depth range
        exceeds twice the tolerance; the neighbourhood depth distribution is
        split at its largest gap and each side is fit to a local plane by
        least squares. A genuine step edge fits both planes with the centre
        on one of them; a mixed pixel lies strictly between the planes.
    """
    # TODO: Implement frame_flying_pixel_ratio.


def frame_valid_tile_coverage(
        depth_path: str, tile_size: int = 64,
        tile_valid_floor: float = 0.5) -> Tuple[float, np.ndarray]:
    """Fraction of depth tiles with enough valid returns (ValidTileCoverage).

    What it does: partitions the frame into tile_size squares and counts a
    tile as supported when its valid-return share meets tile_valid_floor; the
    mask covers every unsupported tile. Higher is better.

    Inputs:
        depth_path: path to a uint16-millimetre depth PNG (0 = invalid).
        tile_size: tile edge in pixels, positive integer.
        tile_valid_floor: minimum valid share per tile, in [0, 1].

    Outputs:
        (value, mask): supported-tile fraction in [0, 1]; uint8 HxW mask with
        255 over every unsupported tile.

    Technical foundations:
        Tile-level coverage approximates the spatial support available to
        frame-to-frame ICP: isolated valid pixels contribute little to
        alignment, while contiguous supported tiles anchor correspondences.
        Edge tiles may be smaller than tile_size and are graded by the same
        share rule.
    """
    # TODO: Implement frame_valid_tile_coverage.


def pair_identity_median_depth_change(
        d0_path: str, d1_path: str,
        change_mask_k: float = 3.0) -> Tuple[float, np.ndarray, int]:
    """Median depth change between two frames at identity pixel alignment.

    What it does: computes |D0 - D1| over jointly valid pixels, returns the
    median in metres, and masks changes beyond median + change_mask_k * 1.4826
    * MAD. Fail-closed: shape mismatch or empty joint support yields inf with
    an empty mask and count 0.

    Inputs:
        d0_path: path to the source uint16-millimetre depth PNG.
        d1_path: path to the target uint16-millimetre depth PNG.
        change_mask_k: outlier threshold multiplier, finite and >= 0.

    Outputs:
        (value, mask, count): median change in metres (lower is better);
        uint8 HxW mask with 255 at outlier pixels; jointly-valid pixel count.

    Technical foundations:
        Median plus scaled MAD is a robust location/scale pair: 1.4826 maps
        MAD to sigma under normality, so the threshold adapts to the pair's
        own spread. A one-quantisation-step floor above the median keeps
        zero-MAD (coherently moving) frames from flagging ordinary motion.
    """
    # TODO: Implement pair_identity_median_depth_change.


def pair_joint_valid_depth_ratio(
        d0_path: str, d1_path: str) -> Tuple[float, np.ndarray, int]:
    """Fraction of pixels valid in both frames (JointValidDepthRatio).

    What it does: intersects the two valid masks (raw != 0 per the ICP
    consumer) and returns the joint share with a mask over every
    not-jointly-valid pixel. Higher is better. Fail-closed: shape mismatch
    yields 0.0 with a full mask and count 0.

    Inputs:
        d0_path: path to the source uint16-millimetre depth PNG.
        d1_path: path to the target uint16-millimetre depth PNG.

    Outputs:
        (value, mask, count): jointly-valid fraction in [0, 1]; uint8 HxW
        mask with 255 where a pixel is not valid in both frames;
        jointly-valid pixel count.

    Technical foundations:
        Pairwise ICP can only use points observed in both frames, so joint
        validity upper-bounds the correspondence pool (overlap driver D1 of
        pairwise ICP failure). The mask convention is inverted relative to
        the other factors: kept pixels are 0, dropped pixels are 255.
    """
    # TODO: Implement pair_joint_valid_depth_ratio.


def pair_prior_warp_depth_residual(
        d0_path: str, d1_path: str, prior_T: np.ndarray,
        intrinsics: DepthIntrinsics,
        prior_warp_depth_gate: float = 0.10) -> Tuple[float, np.ndarray, int]:
    """Median depth residual after warping frame 0 into frame 1.

    What it does: unprojects valid frame-0 pixels, transforms them by the
    pre-ICP registration prior, reprojects into frame 1, and returns the median
    absolute depth residual over visible (non-occluded) support with a mask
    flagging residuals above the gate. Residuals are stored at source-image
    coordinates so the mask applies to frame 0's cloud. Fail-closed: shape
    mismatch or empty support yields inf with an empty mask and count 0.

    Inputs:
        d0_path: path to the source uint16-millimetre depth PNG.
        d1_path: path to the target uint16-millimetre depth PNG.
        prior_T: finite 4x4 rigid transform mapping frame-0 points toward
            frame 1 (callers pass the inverse of the current-to-previous
            registration prior).
        intrinsics: capture camera as a {"width", "height", "hfov"} dict, a
            (width, height, hfov) tuple, or a 3x3 matrix.
        prior_warp_depth_gate: residual gate in metres, finite and > 0; also
            the occlusion margin and the mask threshold.

    Outputs:
        (value, mask, count): median residual in metres (lower is better);
        uint8 HxW mask with 255 where the residual exceeds the gate; visible
        support pixel count.

    Technical foundations:
        Pinhole unprojection with square pixels and centre principal point
        (focal length from width and hfov), followed by rigid warping and
        nearest-pixel association. Points warped substantially behind the
        observed surface read as occluded rather than residual evidence;
        points in front stay as removable outliers. The only pair factor that
        receives consumer state, while still comparing two rasters.
    """
    # TODO: Implement pair_prior_warp_depth_residual.


def depth_image_to_point_cloud(rgb, depth_m, width, height, hfov, keep_mask=None):
    """Back-project one RGB-D frame into a colored 3-D point cloud.

    `rgb` is HxWx3 uint8 in BGR order and `depth_m` the same frame's HxW
    depths in metres; `width`/`height`/`hfov` are that capture's own camera
    parameters (read from its `intrinsics.json` by the caller — never
    hardcoded, so one capture is never unprojected through another's
    camera). Geometry is a pinhole with square pixels, image-centre
    principal point, and focal length derived from `width` and `hfov`.

    Returns points in metres in the frame's camera frame (+X right, +Y
    down, +Z forward) with colors as RGB floats in [0, 1]. A pixel yields
    a point iff `depth_m > 0` (zero means no return and is dropped, never
    placed at the origin), additionally ANDed with `keep_mask` when given.
    Points come out in row-major pixel order; an all-invalid frame yields
    an empty cloud. Inputs are not modified.
    """
    # TODO: Implement depth_image_to_point_cloud.


def preprocess_point_cloud(pcd, voxel_size, target_down=None,
                           target_fpfh=None, random_seed=None):
    """Downsample a cloud and prepare it for feature matching and ICP.

    Voxel-downsamples at `voxel_size`, estimates normals (radius
    `voxel_size * 2`, up to 30 neighbours) for point-to-plane ICP, and
    computes FPFH descriptors (radius `voxel_size * 5`, up to 100
    neighbours) for RANSAC matching. Returns the downsampled cloud and
    its feature set. When target_down and target_fpfh are supplied, also
    computes coarse RANSAC registration from this cloud to that target and
    returns it as a third value; otherwise the third value is None.
    """
    # TODO: Implement preprocess_point_cloud.


def local_icp_algorithm(source_down, target_down, trans_init, threshold):
    """Refine an initial alignment with point-to-plane ICP.

    Estimates normals on either cloud if missing (radius `threshold * 2`,
    up to 30 neighbours), then runs up to 100 ICP iterations at the single
    correspondence `threshold`. Returns the RegistrationResult; its
    `.transformation` is the refined 4x4.
    """
    # TODO: Implement local_icp_algorithm.


def my_local_icp_algorithm(source_down, target_down, trans_init, voxel_size):
    """From-scratch point-to-point ICP (the optional Bonus Track backend).

    Each iteration transforms the source by the current estimate, finds
    nearest neighbours in the target via cKDTree, keeps pairs closer than
    `voxel_size * 1.5` (stops below 10 survivors), and solves the optimal
    rigid update with the Kabsch SVD including a reflection fix. Composes
    each delta onto the running transform for up to 60 iterations, stopping
    when the mean inlier error changes by less than 1e-6. Returns a small
    object exposing the final 4x4 as `.transformation`.
    """
    # TODO: Implement my_local_icp_algorithm.


# ══════════════════════════════════════════════════════════════════════════════
#  Reconstruction (headless) + metric
# ══════════════════════════════════════════════════════════════════════════════
def reconstruct(data_root, version="open3d", voxel_size=0.05, verbose=True,
                build_cloud=True, frames=None, mask_root=None,
                down_voxel=None, registration_seed=0,
                prior_transform_callback=None):
    """Reconstruct a camera trajectory and optionally merge its point clouds.

    Steps for each selected frame: build and preprocess an RGB-D cloud; register
    it to the previous frame with RANSAC initialization and ICP refinement; chain
    the transform and, when requested, add the cloud to the map.

    ``prior_transform_callback``, when supplied, is called immediately before
    ICP for each successfully loaded adjacent pair as
    ``(previous_stem, current_stem, previous_depth_path, current_depth_path,
    previous_to_current_transform)``. RANSAC estimates current-to-previous,
    so the callback receives its inverse. The callback does not alter
    registration.

    Frame selection:
      frames=None -> whole batch: every stem present under BOTH rgb/ and
        depth/, ascending numeric order.
      frames=[...] -> only those integer stems, IN THE GIVEN ORDER;
        "consecutive" means consecutive in the subset.

    Returns (global_pcd, pred_cam_pos):
      global_pcd   : every frame's cloud in the FRAME-0 CAMERA frame
                     (+X right, +Y down, +Z forward). Empty when
                     build_cloud=False (trajectory identical either way).
      pred_cam_pos : (N,3) float64 camera centres, frame-0 anchored.
                     GT stays on disk; visualize_and_evaluate loads and
                     subsets it by the same stems when scoring.
    """
    # Camera intrinsics belong to this capture, not to a global default.
    intrinsics_path = os.path.join(data_root, 'intrinsics.json')
    if not os.path.isfile(intrinsics_path):
        raise FileNotFoundError(
            f"missing camera intrinsics: {intrinsics_path!r}\n"
            "Every capture directory must ship intrinsics.json "
            '("width", "height", "hfov") alongside rgb/ and depth/.')
    with open(intrinsics_path) as f:
        intrinsics = json.load(f)
    missing = [k for k in ("width", "height", "hfov") if k not in intrinsics]
    if missing:
        raise ValueError(
            f"{intrinsics_path!r} is missing required key(s) {missing}; it must hold "
            '{"width": int, "height": int, "hfov": float (DEGREES)}')
    width, height, hfov = (int(intrinsics["width"]), int(intrinsics["height"]),
                           float(intrinsics["hfov"]))

    # Build the frame list in numeric order, or preserve the requested subset order.
    rgb_dir = os.path.join(data_root, 'rgb')
    depth_dir = os.path.join(data_root, 'depth')
    if frames is None:
        rgb_files = sorted(
            (f for f in os.listdir(rgb_dir) if f.endswith('.png')),
            key=lambda f: int(os.path.splitext(f)[0]))
        depth_files = sorted(
            (f for f in os.listdir(depth_dir) if f.endswith('.png')),
            key=lambda f: int(os.path.splitext(f)[0]))
        n = min(len(rgb_files), len(depth_files))
        pairs = [(os.path.join(rgb_dir, rgb_files[i]),
                  os.path.join(depth_dir, depth_files[i])) for i in range(n)]
    else:
        pairs = [(os.path.join(rgb_dir, f"{int(s)}.png"),
                  os.path.join(depth_dir, f"{int(s)}.png")) for s in frames]

    # Index exported masks by every frame stem they affect.
    mask_index = {}
    if mask_root is not None:
        if not os.path.isdir(mask_root):
            raise FileNotFoundError(f"mask directory does not exist: {mask_root!r}")
        for name in os.listdir(mask_root):
            if not name.lower().endswith(".png"):
                continue
            parts = os.path.splitext(name)[0].split("_")
            try:
                incident = ([int(parts[0])] if len(parts) == 1
                            else [int(parts[0]), int(parts[1])])
            except (ValueError, IndexError):
                continue
            for frame in incident:
                mask_index.setdefault(frame, []).append(
                    os.path.join(mask_root, name))

    if verbose:
        print(f"[reconstruct] {data_root}: {len(pairs)} frames | version={version}")

    T_global = np.eye(4)
    camera_poses = []
    global_pcd = o3d.geometry.PointCloud()
    prev_down = prev_fpfh = None
    prev_depth_path = None
    prev_stem = None
    prev_pair_index = None
    icp_threshold = voxel_size * 1.5

    for i, (rgb_path, depth_path) in enumerate(pairs):
        rgb = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is not None:
            if depth.ndim == 3:
                depth = depth[:, :, 0]
            if depth.dtype == np.uint16:
                depth_m = depth.astype(np.float64) / DEPTH_SCALE
            else:
                depth_m = depth.astype(np.float64) / 255.0 * 10.0
        else:
            depth_m = None
        if rgb is None or depth_m is None:
            if verbose:
                print(f"  Warning: could not load frame {i}, skipping.")
            continue

        stem = int(os.path.splitext(os.path.basename(depth_path))[0])
        keep_mask = None
        drop_paths = mask_index.get(stem, ())
        if drop_paths:
            drop = np.zeros(depth_m.shape, dtype=bool)
            for mask_path in drop_paths:
                image = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise ValueError(f"could not read exported mask {mask_path!r}")
                if image.ndim == 3:
                    image = image[:, :, 0]
                if image.shape != depth_m.shape:
                    raise ValueError(
                        f"mask {mask_path!r} shape {image.shape} does not match "
                        f"depth {depth_m.shape}")
                drop |= image != 0
            keep_mask = ~drop

        # 1. Convert RGB-D to PointCloud (Task 1)
        # TODO: Create this frame's colored point cloud using its camera intrinsics
        # and optional keep mask.

        # 2. Preprocess (Voxel/FPFH/Normals)
        # TODO: Prepare the current cloud for registration, including the first
        # frame so it can become the reference for the next pair.

        if prev_down is not None:
            # 3. Execute Global Registration (RANSAC)
            # TODO: Estimate an initial transform from the current cloud to previous.

            # The callback receives previous-to-current motion immediately before
            # refinement. Keep this measurement hook independent of registration.
            if (prior_transform_callback is not None and
                    prev_pair_index == i - 1):
                prior_transform_callback(
                    prev_stem, stem, prev_depth_path, depth_path,
                    np.linalg.inv(np.asarray(trans_init, dtype=np.float64)))

            # 4. Execute Local Registration (ICP - Task 2)
            # TODO: Refine the initial transform with the selected ICP backend.

        # 5. Update camera_poses and accumulate points
        # TODO: Chain the relative transform, record this camera centre, and add
        # the transformed cloud to global_pcd when build_cloud is enabled.

        prev_down, prev_fpfh = cur_down, cur_fpfh
        prev_depth_path = depth_path
        prev_stem = stem
        prev_pair_index = i

        if verbose and (i % 25 == 0 or i == len(pairs) - 1):
            print(f"  frame {i:>4d}/{len(pairs)-1}")

    if not camera_poses:
        return o3d.geometry.PointCloud(), np.zeros((0, 3))
    return global_pcd, np.array(camera_poses, dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation + visualisation (used by the thin reconstruct.py CLI)
# ══════════════════════════════════════════════════════════════════════════════
def visualize_and_evaluate(reconstructed_pcd, predicted_cam_poses, data_root,
                           frames=None, title="reconstruction", show=True):
    """Score the trajectory and visualise map + trajectories.

    GT is loaded here from data_root/GT_pose.npy as (M,7)
    [x,y,z,qw,qx,qy,qz], or None when absent. GT rows are in capture order
    while image stems are identifiers, so when `frames` is given each stem
    is mapped to its ordinal in the sorted common RGB/depth frame list
    before indexing (stems absent from the capture or beyond GT are skipped,
    requested order preserved) — the score stays meaningful on subsets.

    1. Mean L2 over camera centres: reconcile frames first (pred is frame-0
       camera on image axes → flip Y and Z; GT world poses → R0^T (t_i - t_0)
       via pose 0), then mean over i of ||pred[i] - gt[i]||_2 with
       L2 = sqrt(dx^2 + dy^2 + dz^2). No scale/rotation/offset fitting TO the
       GT. inf when either side is missing/empty. Prints it and returns it.
    2. Estimated trajectory LineSet (red), GT trajectory LineSet (black).
    3. Open3D window with the ceiling-cropped cloud plus both trajectories
       (skipped when show=False, e.g. headless scoring). The ceiling sits at
       minimum y in the camera frame (y points down); points within margin
       1.0 of it are dropped from a copy before rotating onto the scoring
       axes, leaving the input cloud unmodified.
    """
    # TODO: Implement visualize_and_evaluate.
