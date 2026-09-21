# `load.py` — Habitat data collector

Interactive Habitat-Sim collector for hw1. A thin driver over the
`simulator` package: it parses the CLI, wires the pieces together, and runs
the event loop. All simulation, pixel-pipeline, replay, and viewer logic
lives in `packages/simulator`.

## Running it

Interactive collection (pygame window, keyboard-driven):

```bash
pixi run -e habitat python hw1/load.py
```

Replay a saved pose trajectory through the same frame-saving pipeline:

```bash
pixi run -e habitat python hw1/load.py --trajectory trajectories/secondfloor.npy
```

The config defaults to `hw1/configs/second_floor.yaml` (`--config` to
override); `--output-root` overrides `output.root`; `--fps` paces the preview
loop.

## Controls

| Keys | Action |
|---|---|
| `w` / `s` | Move forward / backward |
| `a` / `d` | Turn left / right |
| `c` / `SPACE` | Capture the current frame |
| `q` / `ESC` | Finish and quit (aborts a replay too) |

## Uncertainty zones are spatial

The config's `uncertainties` block defines hard-edged circular flicker zones
(`center: [x, z]`, `radius`). A frame is degraded iff the **agent stands
inside a zone** — where it is, not when it got there. Zone membership is
therefore reproducible; the flicker phase and depth noise are not (both key
on time `t`, which is wall-clock while driving and `frame_index / fps` in
replay). `--clean` (or `uncertainties.enabled: false`) switches the zones off
for uncorrupted collection. After a run, a per-zone frame count is printed —
a zone never entered contributed nothing, silently degenerating toward a
clean baseline.

## Outputs (under `output.root`)

- `rgb/<n>.png`, `depth/<n>.png`, optionally `semantic/<n>.png`
- `GT_pose.npy`: `(N, 7)` poses `[x, y, z, qw, qx, qy, qz]`
- `intrinsics.json`: `{"width", "height", "hfov"}` — the capture's own camera
  parameters. Reconstruction reads them from the capture being
  reconstructed, never from a config.

## Performance notes

The preview holds 30 fps by engineering out the expensive parts: the raw
readout is cached while the agent stands still, only consumed sensors are
attached (semantic and bird's-eye views are optional), lighting is a lookup
table, and the viewer repaints only changed panels. If it still stutters:
`display.show_birdseye: false`, `display.scale: 0.5` (matters most over
remote desktop, which re-encodes every changed pixel), or `--fps 20`.

## Import ordering (do not reorder)

On Linux, habitat-sim and pygame fight over the GL context on the same X
display. So the Engine (habitat, offscreen EGL) is constructed **first**, the
viewer (SDL software rendering) is imported **lazily** after that, and this
file never imports pygame directly. Both workarounds are no-ops on macOS.
