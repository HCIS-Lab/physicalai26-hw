# Homework 1 — 3D Scene Reconstruction from RGB-D Observations

The assignment has two phases:

1. Reconstruct the provided first-floor capture of apartment_0.
2. Collect a second-floor capture with Habitat-Sim, reconstruct it with the
   same pipeline, and compare the two results.

## Learning objectives

You should be able to:

- back-project valid depth pixels with the pinhole camera model;
- voxel-downsample point clouds and perform Open3D global registration with RANSAC;
- refine the RANSAC transform with point-to-point or point-to-plane ICP;
- accumulate a 3D map and camera trajectory without using GT poses as reconstruction input;
- inspect RGB-D data quality and relate measurements to alignment behavior;
- report Mean L2 Distance, evaluated frame count, units, runtime, and hyperparameters.

## Environment

The Pixi workspace for this checkout is the directory containing this file and
pixi.toml:

~~~bash
pixi install -e habitat
pixi run -e habitat python -c "import open3d, rdflib; print('ready')"
~~~

The supported workflow is 64-bit Linux. Clone with submodules and run from
physicalai26-hw/hw1 in this checkout:

~~~bash
git clone --recurse-submodules https://github.com/HCIS-Lab/physicalai26-hw.git
cd physicalai26-hw/hw1
pixi install -e habitat
~~~

## Capture format

Each phase uses one capture directory:

~~~text
<capture-dir>/
|-- rgb/<integer-stem>.png       # 8-bit RGB image
|-- depth/<same-stem>.png        # uint16 depth in millimetres; 0 = invalid
|-- GT_pose.npy                  # [x,y,z,qw,qx,qy,qz], shape (N, 7)
`-- intrinsics.json              # {"width", "height", "hfov"}
~~~

RGB and depth files with the same integer stem form one Frame. Preserve matching
stems, resolution, depth encoding, and camera metadata. Always use the
capture's own intrinsics.json. Document preprocessing or working-copy changes.

### Phase 1 — provided first-floor capture

Download and unpack the supplied archive under eval/ without extra nesting or
renamed files:

https://drive.google.com/file/d/1GKa5nNexuRSCDBXQII_K2Q50Ky6rydx3/view?usp=sharing

Inspect it first:

~~~bash
pixi run -e habitat python api.py explore <capture-dir>
~~~

Inspect representative RGB/depth images, metadata, invalid depth, and changes
between consecutive frames. Missing or noisy depth is a possible failure
condition; do not assume every observation is perfect.

### Phase 2 — student-collected second-floor capture

~~~bash
pixi run -e habitat python load.py --config configs/second_floor.yaml --output-root eval/_data/second_floor/<student-id>
~~~

Use w/s to move, a/d to turn, c or Space to capture, and q or Escape to finish.
The collector writes synchronized RGB, depth, GT poses, and intrinsics.json.
Record the output directory, route, frame count, frame spacing, and collection
settings in the submitted README.md. Do not submit Replica environment assets.
Use --clean or uncertainties.enabled: false for a clean collection.

## Required Standard Track

Both phases use the same pipeline:

1. **Depth unprojection:** for valid Z=D(u,v)/s, compute
   X=(u-cx)Z/fx and Y=(v-cy)Z/fy. Zero depth is invalid and must not become a
   point at the origin. Open3D projection helpers are not used for this step.
2. **Voxelization:** downsample point clouds while retaining scene structure.
3. **Global registration:** compute FPFH features and estimate the initial
   transform with Open3D feature-based RANSAC.
4. **Local registration:** refine the RANSAC result with Open3D ICP. The supplied
   path uses point-to-plane ICP; point-to-point ICP is also allowed.
5. **Map and trajectory:** transform and accumulate aligned clouds, compose the
   estimated trajectory, visualize it, and record runtime and settings. The
   reference CLI uses a 0.05 m voxel and Open3D RANSAC seed 0 unless overridden;
   both are written to the run diagnostics sidecar.

GT poses are reference data for evaluation only, not registration input. Show the
estimated trajectory in red and GT in black. Remove the ceiling when it blocks
inspection.

The Standard Track is the required base implementation. The optional Bonus Track
is my_local_icp_algorithm; it must not replace or break the Open3D path.

## `utils.py` implementation checklist

`utils.py` currently contains **19 TODO markers across 15 functions**. Implement
the following functions:

| Area | Functions |
|---|---|
| RGB frame measurements | `frame_mean_value`, `frame_clip_hi_fraction`, `frame_clip_lo_fraction` |
| Depth frame measurements | `frame_high_frequency_depth_residual`, `frame_flying_pixel_ratio`, `frame_valid_tile_coverage` |
| Depth-pair measurements | `pair_identity_median_depth_change`, `pair_joint_valid_depth_ratio`, `pair_prior_warp_depth_residual` |
| Depth and point-cloud preparation | `depth_image_to_point_cloud`, `preprocess_point_cloud` |
| Registration backends | `local_icp_algorithm`, `my_local_icp_algorithm` (optional Bonus Track) |
| Reconstruction pipeline | `reconstruct` (five TODO sections: cloud creation, preprocessing, initial registration, refinement, and pose/map update) |
| Scoring and visualization | `visualize_and_evaluate` |

The checklist count includes every `# TODO:` comment in the file. `reconstruct`
has five TODO sections: cloud creation, preprocessing, initial registration,
refinement, and pose/map update. Every other listed function has one TODO
marker. `my_local_icp_algorithm` is optional and belongs to the Bonus Track;
the other functions are part of the required implementation.

## Mean L2 Distance

Report this required metric in metres for every reconstruction:

~~~text
MeanL2 = (1/N) * sum_i || estimated_position_i - GT_position_i ||_2
~~~

The first camera pose defines the reference frame. Transform estimated and GT
positions into that common frame before comparison. Do not substitute RMSE, MAE,
or a separately fitted alignment.

The RDF support layer stores this metric under `hw1:mapMeanL2`, the implementation
predicate for the trajectory Mean L2 Distance. It is a trajectory metric, not a
point-cloud map metric. Reports and logs use the specification name, Mean L2
Distance.

## Data-verification vocabulary

Use these terms consistently:

| Term | Meaning |
|---|---|
| Batch | One capture directory and shared camera information |
| Frame | One indexed RGB-D observation and pose, when available |
| RGBImage / DepthImage | The color/depth raster for a frame |
| RGBPair / DepthPair | Specification terms for two ordered observations inspected together |
| FramePair (RDF) | Repository representation of one ordered adjacent-frame pair; depth-pair measurements link its two DepthImages |
| Experiment | One named reconstruction/verification attempt on a batch |
| FactorMeasurement | One numeric observation linked to its factor and source image or pair |
| single-image factor | Measurement from one RGB or depth image |
| depth-pair factor | Measurement from two ordered depth observations |

The specification terms are mapped to the RDF implementation as follows:

- `FactorMeasurement` is an experiment-scoped `hw1:Factor` occurrence linked
  through `hw1:hasDefinition` to a reusable `hw1:QualityFactor` definition.
- `RGBPair` and `DepthPair` are conceptual scopes. The RDF graph uses one
  ordered `hw1:FramePair` with `hw1:sourceFrame` and `hw1:targetFrame`; a
  depth-pair `hw1:Factor` occurrence links the ordered DepthImages through
  `hw1:hasPrevious` and `hw1:hasCurrentFrame`.
- A single-image factor is typed as `hw1:SingleImageFactor`; a depth-pair
  factor is typed as `hw1:DepthPairFactor`.

No separate completion-frame class or pair alias is emitted.

Supported factors:

| Scope | Factor |
|---|---|
| RGB frame | HighlightClipping, ShadowClipping |
| depth frame | HighFrequencyDepthResidual, FlyingPixelRatio, ValidTileCoverage |
| depth pair | IdentityMedianDepthChange, JointValidDepthRatio, PriorWarpDepthResidual |

Record each value's scope, units or range, settings, source links, limitations,
and failure cases. These measurements support interpretation; they do not
alone prove causation.

## Optional RDF evidence workflow

api.py provides a local Turtle/rdflib evidence workflow; no server, triple store,
named graph, or OWL reasoner is required:

~~~bash
pixi run -e habitat python api.py explore <capture-dir>
pixi run -e habitat python api.py declare --name phase1_test --data-dir <capture-dir> --floor 1
pixi run -e habitat python api.py experiment experiments/phase1_test.ttl
pixi run -e habitat python reconstruct.py --data_root <capture-dir> --experiment experiments/phase1_test.ttl --no-vis
pixi run -e habitat python api.py explore experiments/phase1_test.ttl
~~~

Assessed experiment files are write-once. Completion is derived from expected
measurement occurrences and evaluationState; no completion RDF class is emitted.
No separate completion-frame vocabulary is emitted.

## Report and submission

Submit exactly <student-id>_hw1.zip with no extra top-level directory:

~~~text
<student-id>_hw1.zip
|-- load.py
|-- reconstruct.py
|-- utils.py
|-- api.py
|-- ontology/
|   `-- hw1.ttl
|-- report.pdf
`-- README.md
~~~

The English PDF must include the complete pipeline and verification workflow,
implementation issues and justifications, screenshots for both phases with
red/black trajectories and ceiling removed when needed, and a comparison table
with Mean L2 Distance, evaluated frame count, unit, total runtime, and tested
hyperparameters. Include quality measurements, settings, source links,
limitations, failure cases, their reconstruction relationship, and answers to:

1. What happens when ICP runs without global RANSAC, and why?
2. Which choices improved ICP stability?
3. How did data quality and collection choices affect reconstruction and Mean L2?

Include a custom-ICP comparison only if the optional Bonus Track is attempted.
Do not include Replica assets, build artifacts, or an extra enclosing directory.

## Verification

~~~bash
python -m py_compile api.py reconstruct.py load.py completeness.py utils.py packages/simulator/simulator/*.py
git diff --check
pixi run -e habitat pytest test_e2e.py -v
env -u PYTHONPATH pixi run -e habitat python -m pytest packages/simulator/tests/ -v
~~~

The Habitat environment must be installed with the repository submodule before
the Pixi commands can run successfully.
