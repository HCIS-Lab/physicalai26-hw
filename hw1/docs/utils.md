# `utils.py` — geometry-only ICP SLAM

Headless reconstruction pipeline shared by the evaluator and the interactive
CLI. `reconstruct.py` is the thin visual entry point; the heavy lifting lives
here so scoring can run with no Open3D window.

## The one rule: geometry only

No colour enters registration, anywhere. RGB is carried on the clouds for
visualisation and nothing else. That keeps the causal chain legible:
lighting reaches the score only through the depth sensor's ambient-light
coupling (noise, dropout, range loss) — if colour leaked into registration,
every downstream conclusion would be unsupported. Keep it that way.

## Depth handling

`load_depth_meters` auto-detects the on-disk encoding:

| Encoding | Conversion to metres |
|---|---|
| uint16 PNG | value / 1000 (millimetres — the eval path) |
| anything else (uint8 vis) | value / 255 × 10 (Habitat 8-bit visualisation) |

The collector writes 16-bit millimetres so injected coupling noise survives
to the reconstructor instead of being swamped by 8-bit quantisation. A pixel
contributes a point iff `depth > 0` — zero means "no return" and must never
become a point at the origin.

`depth_image_to_point_cloud` back-projects one RGB-D frame with an explicit
pinhole model (square pixels, no distortion, centre principal point; focal
length from `width` + `hfov`). Open3D's projection helpers are off-limits
there — the mapping is the thing being learned. Camera parameters always come
from the capture's own `intrinsics.json`, never from a config and never
baked in; there is deliberately no fallback default, because a silently wrong
camera yields a plausible-looking, wrong reconstruction.

## Pipeline stages

| Function | Role |
|---|---|
| `preprocess_point_cloud` | Voxel-downsample, estimate normals, compute FPFH descriptors |
| `global_registration` | RANSAC feature matching → coarse initial transform (fragile, non-deterministic; the raw path) |
| `local_icp_algorithm` | Single-threshold point-to-plane ICP refinement |
| `multiscale_icp` | Coarse-to-fine point-to-plane ICP through shrinking thresholds — the wide capture range the constant-velocity init needs |
| `my_local_icp_algorithm` | From-scratch point-to-point ICP (SVD per iteration, cKDTree correspondences, Kabsch/Umeyama) — student implementation |
| `reconstruct` | Chain pairwise registrations over consecutive frames into a trajectory plus optional global map |
| `mean_l2` | Trajectory score: mean per-frame distance after reconciling the image-axes prediction and world-frame GT into the frame-0 camera frame — no fitting, no alignment to hide drift |
| `pred_positions_frame0` / `gt_positions_frame0` | The two frame reconciliations `mean_l2` scores in, exposed so the visualiser draws exactly what is scored |
| `make_trajectory` / `remove_ceiling` | Visualisation helpers for the CLI |

## `reconstruct` contract, briefly

- **Input.** A capture directory (`rgb/`, `depth/`, `GT_pose.npy`,
  `intrinsics.json`). `frames=None` uses every stem present under both image
  dirs; a stem list uses exactly those, in the given order — "consecutive"
  then means consecutive *in the subset*.
- **Robust mode (default).** Deterministic across runs, with a per-step
  plausibility gate (0.5 m / 30°): an implausible pair is rejected and the
  trajectory coasts on the constant-velocity prior, so one bad pair cannot
  derail everything downstream.
- **Output.** `(global_pcd, pred_cam_pos, gt_poses)` — the merged map in the
  frame-0 camera frame, raw camera centres (never pre-aligned to GT;
  `mean_l2` owns reconciliation), and GT poses or `None`. Frame 0 anchors the
  world (`pred_cam_pos[0] == (0,0,0)`). With `build_cloud=False` the map is an
  empty cloud while trajectory and GT are identical — the evaluator uses this
  because only the trajectory is scored. `return_diagnostics=True` appends
  gate counts and per-link evidence.
- **Subsetting caveat.** Dropping interior frames widens inter-frame motion
  while gate and prior are sized for consecutive frames, so a sparse
  selection scores worse for reasons unrelated to frame quality. That is why
  selection cuts *contiguous segments* (see `docs/reconstruct.md`).
