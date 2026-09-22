# `reconstruct.py` — reconstruction command line

Use this command to run the geometry pipeline on a capture, inspect its estimated
trajectory, and evaluate it when ground-truth poses are available. The
numerical work is in `utils.py`; this script handles command-line options,
reporting, and visualization.

## Getting started

Run the command's help option first, then try it on the supplied capture with
visualization disabled if working on a remote or headless machine. Confirm that
the reported frame count and input directory match your intended capture before
comparing scores.

```bash
pixi run -e habitat python reconstruct.py --help
pixi run -e habitat python reconstruct.py --data_root <capture-directory>
```

The capture directory contains matching RGB and depth images plus its camera
metadata. Use the camera metadata stored with that capture. Ground truth is
used for evaluation and visualization, not as an input to the estimated motion.

## Implementation checkpoints

- Verify one-frame depth loading and point-cloud generation first.
- Check a pairwise registration on adjacent frames before chaining a sequence.
- Confirm that the first camera defines the trajectory origin and that all
  reported positions use a consistent coordinate frame.
- Compare runs using the same capture and clearly reported settings.
- If using a frame selection or mask, record what was selected and explain how
  it affects the sequence being reconstructed.

Consult `utils.py` and the CLI help for supported options. Include the metric,
units, number of evaluated frames, runtime, and relevant settings in your
report. Explain failures and limitations rather than reporting a score alone.
