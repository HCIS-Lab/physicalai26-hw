# `reconstruct.py` — reconstruction CLI and run orchestrator

Reconstructs one captured run (`rgb/`, `depth/`, `GT_pose.npy`) with
geometry-only ICP SLAM, prints trajectory error against ground truth, writes
the number back into the experiment that describes it, and opens an Open3D
window with the cloud plus estimated (red) and GT (black) trajectories.

All registration lives in `utils.py` (headless); all RDF in `api.py`. This
file orchestrates and visualises — it never parses Turtle and never
re-derives the IRI scheme.

```bash
pixi run -e habitat python hw1/reconstruct.py --data_root eval/_data/first_floor/baseline/
pixi run -e habitat python hw1/reconstruct.py --data_root eval/_data/first_floor/baseline \
  --experiment hw1/experiments/strict_clip.ttl --no-vis
```

## Experiment in, two runs out

With `--experiment <path.ttl>`, one experiment file is both the selection
input and the result sink, so a run is self-describing: which frames, under
which settings, produced which error. Selection comes in through
`api.read_experiment`'s usable-link query; results go out through
`api.write_run`.

- **Baseline** (`FullBatch`): the whole batch. This is the convergence
  outcome.
- **Selected** (`GoodSegments`): the usable-link segments. This is a
  **falsification probe**, not a repair promise — it tests whether failed
  inputs are load-bearing while recording the splice, gap, and gate evidence
  for how deletion itself can hurt. Read the selected run together with its
  `spliceCount`/`maxGapLength`/`gatedSteps`, not just its error.

Both runs execute by default, because their comparison is the deliverable.
`--baseline-only` / `--selected-only` restrict; `--no-write` prints without
touching the file. Without `--experiment` this is a plain whole-batch visual
run (no selection, no write-back).

## Why segments, not a frame filter

Selection cuts usable links into maximal **contiguous** segments and
concatenates them, so inside a segment every surviving pair is still a
consecutive capture pair — exactly what the constant-velocity prior and the
per-step gate are sized for. Only the handful of segment boundaries are
splices. Filtering frames one by one would widen every surviving step and
score worse for reasons unrelated to frame quality. If no usable link exists
at all, no selected run is written: an `INF` run would conflate "nothing to
reconstruct" with measurement failure.

A personal policy can replace the default status-driven selection via
`--selection-query <file.rq>` (a local SPARQL `SELECT` binding `?frame` or
`?frameIndex`). Results are validated against the experiment's frames and
cut into contiguous segments, so custom queries cannot smuggle in hidden
temporal jumps.

## Flags worth knowing

| Flag | Meaning |
|---|---|
| `--data_root` | Capture directory. Stays explicit even though the experiment names it: a batch-mismatch guard warns loudly rather than scoring capture A into capture B's experiment. |
| `--version` | `open3d` or `my_icp`. The experiment's recorded `icpBackend` is authoritative — disagreeing with it is a hard error, not an override; with neither, the default is `open3d`. |
| `--reference-root` | Clean capture used to build the whole-scene GT map, adding the `coverageF` map metric at 10 cm alongside the trajectory `mapMeanL2`. |
| `--mask-dir` / `--mask-factor` | One exported factor-mask directory (255 = drop); adds a `MaskFiltered` run. |
| `--no-vis` / `--no-write` | Skip the Open3D window / print without writing. |

## Write-back

Each run goes through `api.write_run` with `mapMeanL2` (a trajectory metric
despite the legacy name), `gatedSteps`, and — for selected runs — splice/gap
metadata plus one `hw1:usedFrame` per consumed frame, so provenance is
answerable from the experiment alone. Missing GT scores `inf`, written as
`"INF"^^xsd:double`, which grades FAILED — fail-closed, like every measurer.
Per-link diagnostics (prior, applied transform, gate firings, fitness, RPE,
drift increments) land in a JSON sidecar next to the experiment.
