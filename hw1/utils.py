"""Geometry-only ICP SLAM utilities (utils.py).

    Full module guide migrated to docs/utils.md. See utils.reconstruct
    docstring for the pipeline contract; this file stays the headless
    implementation imported by reconstruct.py.
"""

import json
import numpy as np
import open3d as o3d
import os
import time
import cv2
import copy
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

# ──────────────────────────────────────────────────────────────────────────────
# Camera intrinsics — PER CAPTURE, never hardcoded here.
#
#   Every capture directory ships its own `intrinsics.json` next to rgb/, depth/
#   and GT_pose.npy, carrying exactly three keys:
#
#       {"width": <px>, "height": <px>, "hfov": <degrees>}
#
#   Read them from the capture you are reconstructing and pass them in. Two
#   different floors, resolutions or fields of view must not be able to end up
#   unprojected through the same baked-in constants — that failure is silent and
#   produces a plausible-looking, wrong reconstruction.
#
#   Depth ENCODING is a separate thing and is not per capture: it is fixed by the
#   on-disk format and handled by load_depth_meters.
# ──────────────────────────────────────────────────────────────────────────────
DEPTH_SCALE  = 1000.0     # uint16 depth PNGs store millimetres

# Image axes (+X right, +Y down, +Z forward — what depth_image_to_point_cloud and
# reconstruct produce) -> the GT sensor's OpenGL axes (+X right, +Y up, +Z back).
# A 180 deg rotation about X: proper (det = +1), so distances and handedness hold.
CAM_TO_GT_AXES = np.diag([1.0, -1.0, -1.0])


def load_depth_meters(depth_path):
    # TODO: Load the depth image and convert it to metres.
    raise NotImplementedError("Implement load_depth_meters")


def depth_image_to_point_cloud(rgb, depth_m, width, height, hfov, keep_mask=None):
    # TODO: Back-project the RGB-D frame into a colored point cloud.
    raise NotImplementedError("Implement depth_image_to_point_cloud")


def preprocess_point_cloud(pcd, voxel_size):
    """
    Voxel-downsample a point cloud and compute FPFH descriptors for
    feature-based global registration.

    Returns:
        pcd_down : downsampled PointCloud with normals
        fpfh     : o3d.pipelines.registration.Feature

    SPEC:
        1. Voxel-downsample `pcd` at `voxel_size`.
        2. Estimate normals on the downsample with a hybrid KD-tree search of
           radius = voxel_size * 2.0, max_nn = 30 (needed for point-to-plane ICP
           and for FPFH).
        3. Compute the FPFH feature on the downsample with a hybrid KD-tree search
           of radius = voxel_size * 5.0, max_nn = 100.
        Return (pcd_down, fpfh). The feature radius must exceed the normal radius.
        WIKI: https://en.wikipedia.org/wiki/Point_Feature_Histograms

    Implement this function.
    """
    # TODO: Downsample the cloud, estimate normals, and compute FPFH features.
    raise NotImplementedError("Implement preprocess_point_cloud")


def global_registration(source_down, target_down, source_fpfh,
                        target_fpfh, voxel_size):
    """
    Estimate an initial rigid transform between two downsampled point clouds
    using RANSAC with FPFH feature matching.

    Returns:
        o3d.pipelines.registration.RegistrationResult

    SPEC:
        Run feature-based RANSAC (registration_ransac_based_on_feature_matching)
        aligning `source_down` -> `target_down` from their FPFH features.
        - Correspondence distance threshold: dist_thr = voxel_size * 1.5.
        - mutual_filter = True; estimation = point-to-point (no scaling); ransac_n = 3.
        - Pruning checkers: edge-length ratio 0.9 and distance <= dist_thr.
        - Convergence: RANSACConvergenceCriteria(max_iteration=100000,
          confidence=0.999).
        Return the RegistrationResult; `.transformation` is the coarse 4*4 init
        handed to ICP. (This is the fragile, non-deterministic path — see the
        reconstruct() `robust` note.)
        WIKI: https://en.wikipedia.org/wiki/Random_sample_consensus

    Implement this function.
    """
    # TODO: Estimate a coarse rigid transform with feature-based registration.
    raise NotImplementedError("Implement global_registration")


def local_icp_algorithm(source_down, target_down, trans_init, threshold):
    """
    Refine alignment with Open3D's Point-to-Plane ICP.

    Args:
        source_down : source PointCloud (with normals)
        target_down : target PointCloud (with normals)
        trans_init  : 4*4 initial transform (from RANSAC)
        threshold   : max correspondence distance

    Returns:
        o3d.pipelines.registration.RegistrationResult

    SPEC:
        Refine `trans_init` with a single-threshold point-to-plane ICP.
        - Ensure both clouds have normals; if missing, estimate with a hybrid
          KD-tree of radius = threshold * 2, max_nn = 30.
        - Run registration_icp(source_down, target_down, threshold, trans_init,
          estimation = TransformationEstimationPointToPlane,
          criteria = ICPConvergenceCriteria(max_iteration=100)).
        Return the RegistrationResult (`.transformation` is the refined 4*4).
        WIKI: https://en.wikipedia.org/wiki/Iterative_closest_point

    Implement this function.
    """
    # TODO: Refine the initial transform with local ICP.
    raise NotImplementedError("Implement local_icp_algorithm")


def multiscale_icp(source_down, target_down, trans_init,
                   thresholds=(0.4, 0.2, 0.1, 0.05), max_iter=60):
    """Coarse-to-fine point-to-plane ICP: refine `trans_init` through decreasing
    correspondence thresholds. The coarse passes give ICP a wide capture range so
    it reaches the true alignment from a constant-velocity init (a single tight
    threshold under-converges at turns and drifts). Geometry-only.

    SPEC:
        Ensure both clouds have normals (hybrid KD-tree radius=0.1, max_nn=30 if
        missing). Start T = trans_init. For each threshold in `thresholds` (coarse
        to fine, default 0.4, 0.2, 0.1, 0.05), run point-to-plane registration_icp
        with ICPConvergenceCriteria(max_iteration=max_iter) and feed the resulting
        transform forward as the init for the next (finer) threshold. Return a
        duck-typed object exposing `.transformation` (final 4*4 np.ndarray) — the
        same attribute an Open3D RegistrationResult exposes, so callers are uniform.
        WIKI: https://en.wikipedia.org/wiki/Iterative_closest_point

    Implement this function.
    """
    # TODO: Refine the initial transform through a coarse-to-fine ICP schedule.
    raise NotImplementedError("Implement multiscale_icp")


def my_local_icp_algorithm(source_down, target_down, trans_init, voxel_size):
    """
    Custom point-to-point ICP (SVD per iteration, cKDTree correspondences).

    Returns a duck-typed result with a `.transformation` (4*4 np.ndarray).

    SPEC:
        Implement point-to-point ICP from scratch (no Open3D registration calls).
        Setup: threshold = voxel_size * 1.5, max_iter = 60, tolerance = 1e-6.
        Take source/target XYZ as float64 arrays; build a cKDTree on the target;
        start T = trans_init (copied). Each iteration:
          1. Transform source by the current T.
          2. Nearest-neighbour correspondences (k=1) into the target via the tree.
          3. Keep pairs with distance < threshold; if fewer than 10 survive, stop.
          4. Solve the optimal rigid transform via the Kabsch/Umeyama SVD:
             centre both point sets, H = Pc^T Qc, U S V^T = svd(H),
             R = V U^T with a det(R) < 0 reflection fix (negate V's last row),
             t = q_bar - R p_bar.
          5. Compose the delta onto T (T = T_delta @ T).
          6. Converge when |prev_mean_err - mean_err| < tolerance.
        Return a duck-typed object with `.transformation` = final 4*4 T.
        WIKI: https://en.wikipedia.org/wiki/Kabsch_algorithm

    Implement this function.
    """
    # TODO: Implement point-to-point ICP without Open3D registration calls.
    raise NotImplementedError("Implement my_local_icp_algorithm")


# ══════════════════════════════════════════════════════════════════════════════
#  Reconstruction (headless) + metric
# ══════════════════════════════════════════════════════════════════════════════
def _sorted_frames(directory):
    """List `directory`'s .png files sorted by integer stem (numeric frame order)."""
    files = [f for f in os.listdir(directory) if f.endswith('.png')]
    return sorted(files, key=lambda f: int(os.path.splitext(f)[0]))


def _rot_angle_deg(R):
    """Geodesic rotation angle (deg) of a 3x3 rotation matrix."""
    c = (np.trace(R[:3, :3]) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _read_intrinsics(capture_root):
    """Return (width, height, hfov_deg) from <capture_root>/intrinsics.json.

    Every capture ships its own camera parameters next to GT_pose.npy, so a
    capture is always unprojected through the camera that recorded it. There is
    deliberately NO fallback default: a silently wrong camera produces a
    plausible-looking, wrong reconstruction, which is worse than a crash.
    """
    path = os.path.join(capture_root, 'intrinsics.json')
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"missing camera intrinsics: {path!r}\n"
            "Every capture directory must ship intrinsics.json "
            '({"width", "height", "hfov"}) alongside rgb/ and depth/. It is NOT '
            "defaulted — a guessed camera yields a wrong reconstruction.")
    with open(path) as f:
        data = json.load(f)
    missing = [k for k in ("width", "height", "hfov") if k not in data]
    if missing:
        raise ValueError(
            f"{path!r} is missing required key(s) {missing}; it must hold exactly "
            '{"width": int, "height": int, "hfov": float (DEGREES)}')
    return int(data["width"]), int(data["height"]), float(data["hfov"])


def _frame_paths(data_root, frames):
    """[(rgb_path, depth_path), ...] in the order the pipeline will consume them.

    frames=None  -> every stem present under BOTH rgb/ and depth/, ascending
                    (n = min(#rgb, #depth), the whole-batch default).
    frames=[...] -> exactly those stems, IN THE GIVEN ORDER — "consecutive"
                    downstream means consecutive in the SUBSET (see SUBSETTING).
    """
    rgb_dir = os.path.join(data_root, 'rgb')
    depth_dir = os.path.join(data_root, 'depth')
    if frames is None:
        rgb_files = _sorted_frames(rgb_dir)
        depth_files = _sorted_frames(depth_dir)
        n = min(len(rgb_files), len(depth_files))
        return [(os.path.join(rgb_dir, rgb_files[i]),
                 os.path.join(depth_dir, depth_files[i])) for i in range(n)]
    return [(os.path.join(rgb_dir, f"{int(s)}.png"),
             os.path.join(depth_dir, f"{int(s)}.png")) for s in frames]


def _mask_index(mask_root):
    """Index one exported factor-mask directory by incident frame stem."""
    if mask_root is None:
        return {}
    if not os.path.isdir(mask_root):
        raise FileNotFoundError(f"mask directory does not exist: {mask_root!r}")
    index = {}
    for name in os.listdir(mask_root):
        if not name.lower().endswith(".png"):
            continue
        stem = os.path.splitext(name)[0]
        parts = stem.split("_")
        try:
            incident = [int(parts[0])] if len(parts) == 1 else [int(parts[0]), int(parts[1])]
        except (ValueError, IndexError):
            continue
        path = os.path.join(mask_root, name)
        for frame in incident:
            index.setdefault(frame, []).append(path)
    return index


def _keep_mask_for_frame(mask_index, stem, shape):
    """Conjoin all frame/pair drop masks incident on ``stem`` into one keep mask."""
    paths = mask_index.get(int(stem), ())
    if not paths:
        return None
    drop = np.zeros(shape, dtype=bool)
    for path in paths:
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"could not read exported mask {path!r}")
        if image.ndim == 3:
            image = image[:, :, 0]
        if image.shape != shape:
            raise ValueError(
                f"mask {path!r} shape {image.shape} does not match depth {shape}")
        drop |= image != 0
    return ~drop


def reconstruct(data_root, version="open3d", voxel_size=0.05, verbose=True,
                build_cloud=True, robust=True, gate_trans=0.5, gate_rot=30.0,
                down_voxel=None, frames=None, return_diagnostics=False,
                mask_root=None, prior_warp_depth_gate=0.10,
                collect_prior_warp=False):
    """
    Geometry-only ICP SLAM over the frames under `data_root`: estimate the camera
    trajectory (and optionally a global map) by chaining pairwise registrations of
    consecutive frames.

    CONTRACT

    Dataset layout
        data_root/rgb/<stem>.png     8-bit colour, one per frame
        data_root/depth/<stem>.png   depth, encoding handled by load_depth_meters
        data_root/GT_pose.npy        optional (M,7) ground-truth pose array
        data_root/intrinsics.json    {"width", "height", "hfov"} for THIS capture
        `<stem>` is an integer. An rgb file and a depth file with the same stem are
        the same frame. Default frame ORDER is ascending integer stem.

        The intrinsics belong to the capture, not to this file: read them from
        `data_root/intrinsics.json` here and pass them into
        depth_image_to_point_cloud. Never hardcode them, and never take them from a
        config — a capture must be reconstructable from its own directory alone.

    Inputs
        data_root  : str — the directory above.
        version    : str — which pairwise-registration backend to use. "open3d"
                     (default) selects the Open3D point-to-plane wrapper that ships
                     working in this file; it is the reference baseline any other
                     backend is compared against. The choice of backend does not
                     change anything else in this contract.
        voxel_size : float, METRES — the resolution clouds are reduced to before
                     registration. Affects accuracy and runtime, not the contract.
        verbose    : bool — progress printing only. Must NOT change the return value.
        build_cloud: bool — when True (default) the accumulated global map is built
                     and returned; when False the map is skipped and an EMPTY
                     PointCloud is returned instead. `pred_cam_pos` and `gt_poses`
                     are identical either way. The evaluator passes False because
                     only the trajectory is scored and the full map reaches ~1e8
                     points.
        robust     : bool, default True. Two guarantees are attached to True, and
                     they are the reason it is the default:
                       (1) DETERMINISM — two runs over the same data return the
                           same `pred_cam_pos`. No uncovered RNG anywhere in the
                           path. A score that moves between runs is not a score.
                       (2) The per-step gate below is enforced.
                     robust=False selects the plain feature-matching-then-ICP chain
                     and carries neither guarantee; it exists so the two can be
                     compared.
        gate_trans : float, METRES  — per-step plausibility bound on translation.
        gate_rot   : float, DEGREES — per-step plausibility bound on the geodesic
                     rotation angle. Both apply only when robust=True, where the
                     invariant is: EVERY relative transform actually applied to the
                     trajectory satisfies ||t|| <= gate_trans and angle <= gate_rot.
                     A single implausible pair must not be allowed to derail the
                     whole trajectory. Defaults (0.5 m / 30 deg) are generous
                     headroom over the ~0.09 m / ~6 deg per-step motion of the
                     provided captures.
        down_voxel : float or None, METRES — when set AND build_cloud, each frame's
                     cloud is reduced to this resolution before being merged into
                     the global map, so the map stays a workable size. Affects the
                     returned map only, never `pred_cam_pos`.
        frames     : optional list of integer frame stems — see SUBSETTING below.
        return_diagnostics : bool — preserve the ordinary three-value return by
                     default.  When True, append a dict containing `gated_steps`
                     and `gated_frames`.  This is run evidence for the experiment
                     inspector; it does not change registration or scoring.

    Returns
        (global_pcd, pred_cam_pos, gt_poses)
          global_pcd   : o3d.geometry.PointCloud — every frame's cloud expressed in
                         the FRAME-0 CAMERA frame and merged. Empty when
                         build_cloud=False.
          pred_cam_pos : (N,3) float64, METRES — estimated camera centres, also in
                         the FRAME-0 CAMERA frame (+X right, +Y down, +Z forward).
                         RAW: do not rotate, scale or align to the GT here —
                         mean_l2 owns the frame reconciliation, and doing it twice
                         is the classic way to produce a wrong score that looks
                         plausible.
          gt_poses     : (M,7) float — [x, y, z, qw, qx, qy, qz] as stored on disk,
                         or None when GT_pose.npy is absent (a capture with no GT
                         must still reconstruct).

    Invariants
        * Frame 0 anchors the world: pred_cam_pos[0] == (0, 0, 0) exactly.
        * len(pred_cam_pos) == the number of frames actually used. A frame whose
          rgb or depth fails to load is skipped, not faked.
        * No frames loadable  -> (empty PointCloud, zeros((0,3)), gt_poses). Not an
          exception.
        * GEOMETRY ONLY (D1). Colour must never enter registration — it is carried
          on the clouds for visualisation and nothing else. This is what makes the
          whole assignment legible: lighting reaches the score ONLY through the
          depth sensor's light coupling, so if colour leaked into registration the
          causal chain being measured would no longer be the one being claimed.
        * `data_root` is read-only; nothing is written.

    SUBSETTING (the `frames` argument / the --experiment path)
        hw1/reconstruct.py --experiment cuts contiguous USABLE-LINK segments from
        the Pass/Fail statuses baked into an experiment Turtle (a link is usable
        iff the pair passed and both its endpoint frames passed — §4.5),
        turns each frame IRI into its integer stem and passes those stems
        here as `frames`. Its baseline run passes frames=None, i.e. whole batch.
          * frames=None (default): use every frame present under rgb/ and depth/,
            in ascending stem order — n = min(#rgb, #depth).
          * frames=[...]: use ONLY those stems, IN THE GIVEN ORDER, resolved as
            data_root/rgb/<stem>.png and data_root/depth/<stem>.png. The entire
            pipeline runs over that reduced sequence — "consecutive" means
            consecutive IN THE SUBSET, not in the original capture.
          * GT is subset by the SAME stems, so gt_poses[i] corresponds to
            pred_cam_pos[i] and mean_l2 stays meaningful. (_load_gt does this.)
          * Expected consequence, not a bug: dropping interior frames WIDENS the
            motion between the frames that remain, while the per-step gate above is
            sized for consecutive frames. A sparse selection therefore scores worse
            for reasons that have nothing to do with the quality of the frames it
            kept. This trade-off is the point of phase 1 — a filter that throws away
            too much is as wrong as one that keeps bad frames. It is also exactly
            why hw1/reconstruct.py cuts CONTIGUOUS segments instead of filtering
            frames one by one: a contiguous segment leaves every surviving pair
            consecutive, so the gate keeps the size it was designed for.

    Reference: https://en.wikipedia.org/wiki/Simultaneous_localization_and_mapping

    Smoke fixture: hw1/tests/fixtures/ ships five synthetic frames with exactly
    known per-step transforms and a GT trajectory whose perfect mean_l2 is 0.0.
    Run this function on that directory before running it on real data — it tells
    "my ICP is wrong" apart from "my loop is wrong". See the README there.
    """
    # Camera parameters come from the capture being reconstructed, never from a
    # config and never baked in — two floors/resolutions must not share one model.
    width, height, hfov = _read_intrinsics(data_root)
    pairs = _frame_paths(data_root, frames)
    mask_index = _mask_index(mask_root)

    if verbose:
        print(f"[reconstruct] {data_root}: {len(pairs)} frames | version={version} "
              f"| robust={robust}")

    T_global = np.eye(4)          # frame i -> frame-0 camera frame
    T_rel_prev = np.eye(4)        # constant-velocity prior (robust path)
    pred_cam_pos = []
    global_pcd = o3d.geometry.PointCloud()
    prev_down = prev_fpfh = None
    icp_thr = voxel_size * 1.5
    n_gated = 0
    gated_frames = []
    link_diagnostics = []
    prior_warp_measurements = []
    previous_stem = None
    previous_depth_path = None

    # Streamed one frame at a time: only the PREVIOUS frame's downsample has to
    # stay resident, so a 400-frame capture does not hold 400 full clouds in RAM.
    for i, (rgb_path, depth_path) in enumerate(pairs):
        t0 = time.time()
        rgb = cv2.imread(rgb_path)
        depth_m = load_depth_meters(depth_path)
        if rgb is None or depth_m is None:
            if verbose:
                print(f"  Warning: could not load frame {i}, skipping.")
            continue

        stem = int(os.path.splitext(os.path.basename(depth_path))[0])
        keep_mask = _keep_mask_for_frame(mask_index, stem, depth_m.shape)
        pcd = depth_image_to_point_cloud(
            rgb, depth_m, width, height, hfov, keep_mask=keep_mask)
        cur_down, cur_fpfh = preprocess_point_cloud(pcd, voxel_size)

        if prev_down is not None:
            if robust:
                trans_init = T_rel_prev                  # constant-velocity init
            else:
                trans_init = global_registration(
                    cur_down, prev_down, cur_fpfh, prev_fpfh,
                    voxel_size).transformation

            if collect_prior_warp:
                try:
                    try:
                        from . import api as factor_api
                    except ImportError:
                        import api as factor_api
                    # The registration prior maps current -> previous.  The
                    # factor's ordered pair is previous -> current.
                    value, drop_mask, count = factor_api._prior_warp_depth_residual(
                        previous_depth_path, depth_path, np.linalg.inv(trans_init),
                        (width, height, hfov), prior_warp_depth_gate)
                except Exception:
                    value, count = float("inf"), 0
                    drop_mask = np.zeros(depth_m.shape, dtype=np.uint8)
                prior_warp_measurements.append({
                    "source": int(previous_stem), "target": int(stem),
                    "value": float(value), "count": int(count),
                    "mask": drop_mask})

            if version == 'open3d':
                # Coarse-to-fine in the robust path (wide capture range from the
                # CV prior); single tight-threshold ICP in the raw path.
                if robust:
                    result_icp = multiscale_icp(cur_down, prev_down, trans_init)
                else:
                    result_icp = local_icp_algorithm(
                        cur_down, prev_down, trans_init, icp_thr)
            else:   # my_icp
                result_icp = my_local_icp_algorithm(
                    cur_down, prev_down, trans_init, voxel_size)
            T_rel_pre_gate = np.asarray(result_icp.transformation, dtype=np.float64)
            T_rel = T_rel_pre_gate.copy()
            proposed_translation = float(np.linalg.norm(T_rel[:3, 3]))
            proposed_rotation = _rot_angle_deg(T_rel)
            gate_fired = False

            # Physical gate: an implausible one-step jump is rejected and the
            # trajectory coasts on the prior instead. One bad pair must not be
            # able to derail everything downstream of it.
            if robust and (np.linalg.norm(T_rel[:3, 3]) > gate_trans
                           or _rot_angle_deg(T_rel) > gate_rot):
                T_rel = T_rel_prev
                gate_fired = True
                n_gated += 1
                gated_frames.append(int(os.path.splitext(os.path.basename(rgb_path))[0]))

            src_points = np.asarray(cur_down.points, dtype=np.float64)
            tgt_points = np.asarray(prev_down.points, dtype=np.float64)
            correspondence_count = 0
            fitness = 0.0
            inlier_rmse = float("inf")
            if len(src_points) and len(tgt_points):
                transformed = ((T_rel_pre_gate[:3, :3] @ src_points.T).T +
                               T_rel_pre_gate[:3, 3])
                distances, _nearest = cKDTree(tgt_points).query(
                    transformed, k=1, workers=1)
                inliers = distances < icp_thr
                correspondence_count = int(np.count_nonzero(inliers))
                fitness = float(correspondence_count) / float(len(src_points))
                if correspondence_count:
                    inlier_rmse = float(np.sqrt(np.mean(distances[inliers] ** 2)))

            link_diagnostics.append({
                "source": int(previous_stem), "target": int(stem),
                "prior": np.asarray(trans_init).tolist(),
                "pre_gate_transform": T_rel_pre_gate.tolist(),
                "applied_transform": T_rel.tolist(),
                "gate_fired": bool(gate_fired),
                "proposed_translation_m": proposed_translation,
                "proposed_rotation_deg": proposed_rotation,
                "fitness": fitness,
                "inlier_rmse_m": inlier_rmse,
                "correspondence_count": correspondence_count,
                "rpe_translation_m": None,
                "rpe_rotation_deg": None,
                "drift_increment_m": None})

            T_rel_prev = T_rel
            T_global = T_global @ T_rel

        pred_cam_pos.append(T_global[:3, 3].copy())

        if build_cloud:
            # down_voxel reduces the MAP only; the trajectory above is untouched.
            merged = pcd if down_voxel is None else pcd.voxel_down_sample(down_voxel)
            merged.transform(T_global)
            global_pcd += merged

        prev_down, prev_fpfh = cur_down, cur_fpfh
        previous_stem = stem
        previous_depth_path = depth_path

        if verbose and (i % 25 == 0 or i == len(pairs) - 1):
            print(f"  frame {i:>4d}/{len(pairs)-1}  dt={time.time()-t0:.2f}s  "
                  f"gated={n_gated}")

    gt_for_run = _load_gt(data_root, frames)
    if gt_for_run is not None and link_diagnostics:
        n_links = min(len(link_diagnostics), max(0, len(gt_for_run) - 1))
        for k in range(n_links):
            prev_pose, cur_pose = gt_for_run[k], gt_for_run[k + 1]
            world_prev = np.eye(4)
            world_cur = np.eye(4)
            world_prev[:3, :3] = Rotation.from_quat(
                [prev_pose[4], prev_pose[5], prev_pose[6], prev_pose[3]]).as_matrix()
            world_cur[:3, :3] = Rotation.from_quat(
                [cur_pose[4], cur_pose[5], cur_pose[6], cur_pose[3]]).as_matrix()
            world_prev[:3, 3] = prev_pose[:3]
            world_cur[:3, 3] = cur_pose[:3]
            gt_rel = np.linalg.inv(world_prev) @ world_cur
            applied = np.asarray(link_diagnostics[k]["applied_transform"])
            error = np.linalg.inv(gt_rel) @ applied
            link_diagnostics[k]["rpe_translation_m"] = float(
                np.linalg.norm(error[:3, 3]))
            link_diagnostics[k]["rpe_rotation_deg"] = _rot_angle_deg(error)

        pred_arr = np.asarray(pred_cam_pos, dtype=np.float64)
        pred_eval = pred_positions_frame0(pred_arr)
        gt_eval = gt_positions_frame0(gt_for_run)
        n_eval = min(len(pred_eval), len(gt_eval))
        if n_eval > 1:
            errors = np.linalg.norm(pred_eval[:n_eval] - gt_eval[:n_eval], axis=1)
            for k in range(min(len(link_diagnostics), n_eval - 1)):
                link_diagnostics[k]["drift_increment_m"] = float(
                    errors[k + 1] - errors[k])

    diagnostics = {"schema_version": 1,
                   "gated_steps": n_gated,
                   "gated_frames": gated_frames,
                   "links": link_diagnostics,
                   "prior_warp_measurements": prior_warp_measurements}
    if not pred_cam_pos:
        result = (o3d.geometry.PointCloud(), np.zeros((0, 3)),
                  gt_for_run)
        return (*result, diagnostics) if return_diagnostics else result

    result = (global_pcd,
              np.array(pred_cam_pos, dtype=np.float64),
              gt_for_run)
    return (*result, diagnostics) if return_diagnostics else result


def _load_gt(data_root, frames=None):
    """Load data_root/GT_pose.npy (the (M,7) GT pose array), or None if absent.

    GT rows are in capture order; image stems are identifiers, not guaranteed row
    numbers.  When `frames` is given, map each requested stem to its ordinal in the
    sorted common RGB/depth frame list before indexing GT.  This handles both the
    conventional 0..N-1 capture and datasets such as eval/first_floor whose PNGs
    are named 1..N.  Stems absent from the capture or beyond GT are skipped while
    preserving requested order.
    """
    gt_path = os.path.join(data_root, 'GT_pose.npy')
    if not os.path.exists(gt_path):
        return None
    gt = np.load(gt_path)
    if frames is None:
        return gt
    ordered_paths = _frame_paths(data_root, None)
    stem_to_row = {
        int(os.path.splitext(os.path.basename(rgb_path))[0]): row
        for row, (rgb_path, _depth_path) in enumerate(ordered_paths)
    }
    rows = [stem_to_row[int(stem)] for stem in frames
            if int(stem) in stem_to_row and stem_to_row[int(stem)] < len(gt)]
    return gt[rows]


def pred_positions_frame0(pred_cam_pos):
    """Predicted camera centres re-expressed on the GT sensor's axes (metres).

    `reconstruct` returns centres on IMAGE axes (+X right, +Y down, +Z forward),
    while Habitat's recorded poses use the OpenGL sensor axes (+X right, +Y up,
    +Z backward). The two differ by a 180 deg rotation about X — a PROPER rotation
    (det = +1), not a mirror — so the conversion is a sign flip on Y and Z.

    Both frames are anchored at the frame-0 camera, so the origin is untouched.

    SHIPS WORKING — the code below is the reference implementation. Not a stub;
    you do not write this one.
    """
    return np.asarray(pred_cam_pos, dtype=np.float64) @ CAM_TO_GT_AXES


def gt_positions_frame0(gt_poses):
    """GT camera centres expressed in the FRAME-0 GT CAMERA frame (metres).

    Takes each stored pose to a 4x4 world transform and left-multiplies by
    inv(pose[0]), i.e. the full rigid change of frame the reference scorer does:
    the returned row i is R0^T (t_i - t_0), so the GT trajectory starts at the
    origin AND is expressed on frame 0's own axes.

    The rotation is what a translation-only alignment misses: if the capture's
    first pose is rotated relative to the world, dropping R0^T scores a PERFECT
    reconstruction as several metres wrong. Every capture shipped with this
    assignment happens to start at identity, which hides the difference — the
    rotation is applied anyway so a capture that does not cannot score nonsense.

    SHIPS WORKING — the code below is the reference implementation. Not a stub;
    you do not write this one.
    """
    gt = np.asarray(gt_poses, dtype=np.float64)
    t = gt[:, :3]
    qw, qx, qy, qz = gt[0, 3], gt[0, 4], gt[0, 5], gt[0, 6]
    R0 = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()   # scipy order: x,y,z,w
    return (t - t[0]) @ R0                                  # rows: R0^T (t_i - t_0)


def mean_l2(pred_cam_pos, gt_poses):
    """
    Mean L2 distance between predicted and GT camera centres (metres).

    Reconciles the two coordinate frames before scoring:
      * pred is in the frame-0 CAMERA frame on IMAGE axes → flip Y and Z to reach
        the GT sensor's axes (pred_positions_frame0).
      * GT is stored as world poses [x, y, z, qw, qx, qy, qz] → transform by
        inv(pose_0), giving R0^T (t_i - t_0) (gt_positions_frame0).
    Both trajectories then start at the origin on the same axes, so they are
    compared directly — no further alignment, and in particular no fitting of a
    scale, rotation or offset TO the GT.
    Returns +inf if either trajectory is missing/empty.

    SPEC:
        Compute the mean per-frame Euclidean distance (metres) between predicted and
        GT camera centres over the first n = min(len(pred), len(gt)) frames.
        - Guard: return float("inf") if either input is None or empty.
        - Reconcile frames before comparing:
            pred (frame-0 camera, image axes): negate the Y and Z columns.
            gt   (world poses):  R0 = rotation matrix of pose 0's quaternion
                 (stored qw first, scipy wants x,y,z,w); GT centre i becomes
                 R0^T (t_i - t_0).
        - Return mean over i of ||pred[i] - gt_c[i]||_2.

    SHIPS WORKING — the code below is the reference implementation. Not a stub;
    you do not write this one.
    """
    if gt_poses is None or len(gt_poses) == 0 or pred_cam_pos is None \
            or len(pred_cam_pos) == 0:
        return float("inf")

    pred = pred_positions_frame0(pred_cam_pos)
    gt_c = gt_positions_frame0(gt_poses)

    n = min(len(pred), len(gt_c))
    return float(np.mean(np.linalg.norm(pred[:n] - gt_c[:n], axis=1)))


# ══════════════════════════════════════════════════════════════════════════════
#  Visualisation helpers (used by the thin reconstruct.py CLI)
# ══════════════════════════════════════════════════════════════════════════════
def make_trajectory(positions, color):
    """Create an Open3D LineSet from a sequence of XYZ camera positions.

    SPEC:
        Given positions (N*3) and an RGB `color` triple, build an o3d LineSet whose
        vertices are the positions and whose edges connect each consecutive pair
        [i, i+1] for i in 0..N-2 (a polyline through the trajectory), with every
        line painted `color`. Return the LineSet.

    SHIPS WORKING — the code below is the reference implementation. Not a stub;
    you do not write this one.
    """
    positions = np.asarray(positions)
    lines = [[i, i + 1] for i in range(len(positions) - 1)]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(positions)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


def remove_ceiling(pcd, margin=0.3):
    """Crop the ceiling out of a point cloud for a cleaner top-down view.

    SPEC:
        In the camera frame the ceiling sits at the MINIMUM y (y points down).
        Let y_min = min of the y coordinates. Keep only points with
        y > y_min + margin, carrying their matching colors across. Return a new
        PointCloud of the kept points/colors (input left unmodified).

    SHIPS WORKING — the code below is the reference implementation. Not a stub;
    you do not write this one.
    """
    pts  = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    y_min = pts[:, 1].min()                    # ceiling is at min in camera frame
    mask  = pts[:, 1] > (y_min + margin)
    out   = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts[mask])
    out.colors = o3d.utility.Vector3dVector(cols[mask])
    return out
