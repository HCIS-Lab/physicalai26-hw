# `api.py` — data-quality triplestore CLI

`api.py` turns a directory of captured RGB-D frames into local RDF data,
measures the quality factors a student selects, and stores each result together
with its Pass/Fail verdict — all in the student's own experiment file.

There is no server. Every artefact is a self-contained `.ttl` file on disk,
and queries run locally against it with `rdflib`.

## The workflow

1. **Declare** — `declare` scaffolds a starting experiment file under
   `hw1/experiments/`: prefixes, the Experiment node, the link to the capture
   directory, a factor selection, and a prediction block left as TODO comments
   for the student to write. It never measures and never overwrites.
2. **Assess** — `experiment` takes one student-written declaration, measures the
   rasters in the capture directory, and appends the machine section to that
   same file: one annotation per frame per measured modality, one node per
   consecutive frame pair, every value with its baked status, defaulted
   settings, and the `hw1:declarationDigest` seal. This happens exactly once —
   a file that already has a machine section is rejected.
3. **Inspect** — `explore` prints read-only terminal tables over a capture, a
   declaration, an assessed experiment, or several experiments side by side.
   It writes and measures nothing. `query` runs a student-supplied read-only
   SPARQL query against one local file.
4. **Reconstruct** — `reconstruct.py` (a separate file, the fifth command of the
   suite) appends `hw1:ReconstructionRun` nodes below the marker via `write_run`.

## Commands

| Command | What it does |
|---|---|
| `declare` | Scaffold a declaration Turtle (boilerplate only; the student authors the rest) |
| `experiment` | Measure the capture and append the machine section — once, ever |
| `explore` | Read-only tables over captures, declarations, and assessed experiments |
| `query` | Run a read-only SPARQL query against one local `.ttl` file |
| `batch2ttl` | Optional: write generation provenance (`--gen`, `--derived-from`) into `<data_dir>/batch.ttl`. No longer a required step. |

## Core ideas (details in `triplestore.md`)

- **Write-once files.** An experiment file is a student declaration plus a
  machine section. Nothing is ever overwritten, so re-measuring at a new
  threshold means a new file — `hw1/experiments/` is an append-only lab
  notebook.
- **Baked verdicts.** Every measured value is stored next to its Pass/Fail
  status, computed from the thresholds recorded on the same experiment. The
  raw value is always kept too, so re-grading stays possible.
- **Parameters with roles.** Every settable number is a `hw1:Parameter`
  declared in `ontology/hw1.ttl`. Its role tells you where it lives and what
  to fix: *generation* (how the pixels were produced → regenerate the data),
  *measurement* (how pixels are scored → new declaration), or
  *qualification* (the Pass/Fail threshold → new declaration).
- **Shared frames, scoped results.** All experiments over one capture point at
  the same frame IRIs; everything each experiment mints carries its own name,
  so experiments never collide.

## What students implement

Only the eight qualification-factor measurers (see `factors.md`); never the
RDF machinery:

| Function | Factor |
|---|---|
| `frame_clip_hi_fraction` | HighlightClipping |
| `frame_clip_lo_fraction` | ShadowClipping |
| `frame_high_frequency_depth_residual` (+ mask) | HighFrequencyDepthResidual |
| `frame_flying_pixel_ratio` (+ mask) | FlyingPixelRatio |
| `frame_valid_tile_coverage` (+ mask) | ValidTileCoverage |
| `pair_identity_median_depth_change` (+ mask) | IdentityMedianDepthChange |
| `pair_joint_valid_depth_ratio` (+ mask) | JointValidDepthRatio |
| `pair_prior_warp_depth_residual` (+ mask) | PriorWarpDepthResidual |

Each ships with a reference implementation so the pipeline runs end to end;
in the student distribution each body is a `#TODO` stub whose docstring
contract is the assignment. The CLI, validation, status computation, ontology
wiring, and marker/seal machinery ship as-is and are off-limits.

## Technical notes

- **Depth format.** Depth PNGs are uint16 millimetres (`metres = raw / 1000`).
  A pixel is valid iff `raw != 0` — exactly what
  `utils.depth_image_to_point_cloud` consumes.
- **Dependencies.** Standard library + numpy + Pillow + rdflib. No scipy, no
  OpenCV, no Open3D — every factor is computed directly from the raw PNGs, so
  the factors grade the input data rather than a reconstruction pipeline that
  may itself be unimplemented. Nothing here imports `utils.py`, so an
  unfinished ICP cannot block measurement and vice versa.
- **See also.** `ontology/hw1.ttl` (the TBox these triples must satisfy),
  `docs/triplestore.md` (recording strategy), `docs/factors.md` (factor
  contracts).
