# `completeness.py` — coverage-aware map score

Trajectory error alone rewards doing less: 5 frames of one corner can post a
tiny error while a 380-frame whole-apartment run scores worse. This module
couples *accuracy* with *coverage* by scoring the reconstructed cloud against
a whole-floor ground-truth map:

| Metric | Meaning at threshold τ (default 5 / 10 / 20 cm) |
|---|---|
| accuracy(τ) | fraction of **predicted** points within τ of the GT map |
| completeness(τ) | fraction of **GT-map** points within τ of the prediction |
| F(τ) | `2·A·C / (A+C)` — one number folding both together |

Accuracy alone is gameable by a well-placed sliver; completeness collapses
for a sliver because most of the floor is uncovered. The primary score is
F at 10 cm (`coverageF` in run nodes).

## Alignment without fitting

The reconstruction lives in the first camera's frame, and Habitat supplies
that camera's world pose — so one known anchor matrix lifts every predicted
point into the world frame:

```
T_anchor = Twc0 @ F        Twc0 = [ R(quat0) | t0 ]   (from GT_pose[0])
                               F    = diag(1, -1, -1)
```

No Umeyama, no ICP, no RANSAC: nothing to overfit or diverge. The residual
distance stays equal to the real reconstruction drift, where an alignment fit
would hide it.

## The GT reference

`build_gt_reference` unprojects every 4th frame of a **clean baseline**
capture and places each with its ground-truth pose. It refuses a `mixed/`
(uncertainty-corrupted) directory outright — a corrupted reference would
poison every score measured against it. The map is fixed and independent of
what anyone collected, so a 5-frame submission is scored against the entire
apartment.

## The grader trusts no student code

This module imports nothing from `utils.py`, deliberately duplicating a few
dozen lines of frame listing, depth loading, and unprojection in frozen
private helpers. Two reasons: `utils.py` ships as a `#TODO` stub, which
would make the grader uncallable; worse, the reference map itself is built
here, so student unprojection would corrupt the yardstick rather than the
thing being measured. Do not "clean up" the duplication — a change here
changes everyone's score.
