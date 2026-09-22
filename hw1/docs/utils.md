# `utils.py` — reconstruction and measurement helpers

This module contains the numerical work used by the command-line tools. Each
function has a docstring describing its role, inputs, outputs, and constraints.
Implement the TODOs in small steps and use the provided checks to validate each
stage before connecting the full pipeline.

## Suggested order

1. Implement depth loading and RGB-D back-projection. Confirm that invalid depth
   samples do not become points and that camera parameters come from the capture.
2. Implement point-cloud preparation, coarse registration, and local refinement.
   Keep colour out of all registration decisions; colour is for displaying the
   result.
3. Implement trajectory reconstruction. Express camera positions consistently
   relative to the first frame, and use recorded poses only when evaluating the
   estimate.
4. Implement the frame and frame-pair measurements. Return the documented value
   and any required mask or support count, and handle empty or malformed input
   according to each function's contract.
5. Add visualization and scoring once the headless numerical path works.

## Shared expectations

- Work in metres for geometry and keep image dimensions and camera metadata
  consistent with the input capture.
- Preserve frame ordering and pair direction. A frame's RGB and depth images
  share the same integer filename stem.
- Treat absent depth returns as invalid data. Do not silently fabricate points
  or poses when inputs are missing.
- Keep the required Open3D registration path as the default. The custom ICP
  function is an optional extension described by its own docstring.
- Keep functions deterministic for identical inputs where their contracts
  require reproducibility.

## Pipeline overview

The capture loader supplies RGB images, depth images, and camera intrinsics.
Depth back-projection creates colored point clouds; preprocessing prepares each
cloud for registration; registration estimates relative motion between adjacent
frames. Chaining relative motions produces the trajectory, while optional map
accumulation combines the transformed clouds. Evaluation compares the predicted
trajectory with available ground truth after accounting for their coordinate
frames.

Frame and pair measurements summarize image and depth conditions that may
influence reconstruction quality. Their docstrings describe the quantity,
parameters, and return shape expected by the rest of the assignment. The
experiment tools connect those measurements to their records and reports.
