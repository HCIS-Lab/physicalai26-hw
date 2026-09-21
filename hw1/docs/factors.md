# Quality-factor definitions

The eight measurable quality factors (plus one baseline) behind the factor
menu. A student selects 1–8 input factors per experiment; each produces a
scalar, a baked Pass/Fail status, and — for every depth factor — a drop mask.
Students implement the eight measurers (README.md §5.2); each body ships
as a reference implementation so the pipeline runs end to end, and is a
`#TODO` stub in the student distribution whose contract here is the
assignment.

## Shared conventions

- **Value channel.** `V = max(R,G,B)` per pixel, in `[0, 255]`. No
  colorimetric weighting anywhere.
- **Depth validity.** Depth PNGs are uint16 millimetres; a pixel is valid iff
  `raw != 0` — exactly what the ICP consumer
  (`utils.depth_image_to_point_cloud`) consumes.
- **Drop masks.** Every depth factor also emits a uint8 mask: 255 means drop.
- **Purity.** Every measurer is pure and deterministic: frame path(s) plus
  parameters in, one documented scalar out. No I/O beyond reading the frame,
  no global state, no randomness — so a stored value is reproducible from the
  file it was measured from.
- **Fail-closed.** A measurement that cannot be made returns `inf` (or `0.0`
  for higher-is-better fractions) rather than raising, and therefore grades
  Fail. See `docs/triplestore.md` for how non-finite values are stored.

## At a glance

| Scope | Factor | What it measures | Better | Threshold | Tuning knobs |
|---|---|---|---|---|---|
| RGB frame | HighlightClipping | blown-out pixel fraction | lower | `maxClipHiFraction` | `tauHi` |
| RGB frame | ShadowClipping | crushed pixel fraction | lower | `maxClipLoFraction` | `tauLo` |
| depth frame | HighFrequencyDepthResidual | sensor-noise residual (m) | lower | `maxResidualM` | `residualMaskK` |
| depth frame | FlyingPixelRatio | mixed-boundary pixel fraction | lower | `maxFlyingPixelRatio` | `flyingPixelWindow`, `flyingPixelPlanarityTol` |
| depth frame | ValidTileCoverage | supported-tile fraction | higher | `minValidTileCoverage` | `tileSize`, `tileValidFloor` |
| depth pair | IdentityMedianDepthChange | unwarped median depth change (m) | lower | `maxMedianDepthChange` | `changeMaskK` |
| depth pair | JointValidDepthRatio | jointly-valid pixel fraction | higher | `minJointValidRatio` | — |
| depth pair | PriorWarpDepthResidual | constant-velocity prior residual (m) | lower | `maxPriorWarpResidual` | `priorWarpDepthGate` |

## RGB frame factors

Geometry-only ICP never reads RGB, so these two are negative controls for
this consumer: accurate indicators of a capture condition that should be
irrelevant to the geometry. Telling "true" apart from "load-bearing" is part
of the report.

### HighlightClipping — `frame_clip_hi_fraction`

Fraction of blown-out pixels: `|{ V >= tau_hi }| / N`. Lower is better; Pass
iff at or below `maxClipHiFraction`.

`tau_hi` has no default and must never be hardcoded: it is a measurement
parameter you derive and justify, and a fraction measured at one tau is not
the same quantity as at another. It is recorded on the Batch node.

### ShadowClipping — `frame_clip_lo_fraction`

Fraction of crushed pixels: `|{ V <= tau_lo }| / N`. Lower is better; Pass
iff at or below `maxClipLoFraction`. Same no-default rule for `tau_lo`.

### A deliberate asymmetry worth knowing

Both clip factors use `max(R,G,B)`, but they mean different quantifiers —
and that is correct, not a bug:

- `V >= tau_hi` ⟺ **at least one** channel is saturated (exists). One railed
  channel has already destroyed the pixel's colour.
- `V <= tau_lo` ⟺ **every** channel is crushed (for all). A pixel is only
  truly black when nothing is left in any channel — e.g. pure red
  `(255,0,0)` must not count as crushed, which a `min(R,G,B)` test would get
  wrong.

Source for both: Shin et al., "Camera Exposure Control for Robust Robot
Vision with Noise-Aware Image Quality Assessment", IROS 2019, eq. 7
(unsaturated-region mask). One observable is split into two because the gate
does not need the direction but the diagnosis does: a single number cannot
tell a crushed frame from a blown-out one.

### `mean_value` — the baseline to beat (not a factor)

Mean of `V` over one frame, in `[0, 255]`. It ships implemented, has no
threshold, no polarity, and deliberately no status — it takes no part in
qualification. Its job is to be a weak baseline your clip factors must
outperform, and it is weak on purpose: a single mean cannot represent two
failure modes (one threshold admits both under- and over-exposure), it is
confounded with scene content (a bright window moves it as much as the
effect you measure), and it is blind to distribution shape (a half-crushed,
half-blown frame averages a perfectly ordinary 127.5 while not one pixel is
recoverable). Tail and order statistics do not fail this way.

## Depth frame factors

### HighFrequencyDepthResidual — `frame_high_frequency_depth_residual`

Robust high-frequency depth residual in metres (Immerkaer 1996 estimator: 3×3
mask annihilating locally-linear surfaces, median-absolute-response scaled by
the mask norm 6.0 and the MAD-to-sigma constant 0.6745). Lower is better.
The mask flags pixels whose response exceeds `residualMaskK` times the frame
median (with a one-millimetre-response floor so quantised planar input does
not turn every step edge into noise). These kernel constants are fixed — they
are the published estimator, not parameters.

### FlyingPixelRatio — `frame_flying_pixel_ratio`

Fraction of mixed-boundary ("flying") pixels, discriminated by local plane
fits: a candidate pixel is flagged only if its neighbourhood splits into two
locally planar populations at different depths while the centre belongs to
neither. Lower is better. Tuned by `flyingPixelWindow` (odd, ≥ 3) and
`flyingPixelPlanarityTol` (metres).

### ValidTileCoverage — `frame_valid_tile_coverage`

Fraction of depth tiles (default 64×64) whose valid-return share meets
`tileValidFloor`. Higher is better. The mask covers every unsupported tile.

## Depth pair factors

Frame factors cannot see what happens *between* two frames — and a pair of
individually perfect frames taken a metre apart registers no better than a
pair of bad ones. The pair factors cover overlap (D1) and inter-frame motion
(D2) drivers of pairwise ICP failure.

### IdentityMedianDepthChange — `pair_identity_median_depth_change`

Median `|D0 − D1|` at identity pixel alignment over jointly valid pixels, in
metres. Lower is better. The mask flags changes beyond
`median + changeMaskK · 1.4826 · MAD` (with a one-quantisation-step floor for
zero-MAD frames). Shape mismatches or empty joint support fail closed.

### JointValidDepthRatio — `pair_joint_valid_depth_ratio`

Fraction of pixels valid in *both* frames, in `[0, 1]`. Higher is better.
The mask is simply every not-jointly-valid pixel.

### PriorWarpDepthResidual — `pair_prior_warp_depth_residual`

Median depth residual after warping frame 0 into frame 1 under the
constant-velocity prior transform, in metres. Lower is better. Residuals are
stored at source-image coordinates so the mask applies directly to frame 0's
cloud; points substantially behind the observed surface count as occluded
rather than as residual evidence. The mask flags residuals above
`priorWarpDepthGate`. Measured from the real pre-fit prior state and written
back by `write_pair_measurements` — the one pair factor that receives
consumer state, while still comparing two rasters.
