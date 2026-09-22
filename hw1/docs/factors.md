# Image and depth quality measurements

The assignment asks you to measure conditions in individual RGB or depth frames
and conditions shared by adjacent depth frames. These measurements help explain
when reconstruction succeeds or struggles. Read the matching function
signatures and docstrings in `utils.py` for the required inputs and outputs.

## Shared implementation guidance

- Keep each measurement focused on its stated image or frame-pair property.
- Use the capture's stored depth encoding and treat missing returns as invalid.
- Make output units and direction of improvement clear in your report.
- Depth measurements that produce masks should use the documented mask format
  and shape. A mask is an aid to inspecting or filtering the measurement's
  evidence; it does not change the source capture.
- Handle empty support, mismatched dimensions, and invalid settings deliberately
  as described by each function's contract.
- Keep measurements reproducible from the same source files and settings.

## Frame measurements

The RGB measurements summarize exposure clipping. Depth-frame measurements
summarize sensor residuals, boundary artifacts, and spatial support. Consider
how each property could affect the points available to registration, and explain
that connection in your report. Do not use RGB appearance to estimate geometric
alignment.

## Pair measurements

Pair measurements describe overlap and depth consistency between two ordered
frames. Preserve the source/target order throughout your implementation. Where a
measurement uses a motion prior, document how that prior is supplied and which
frame its output mask refers to.

## Experiments and interpretation

A measurement value is meaningful only with its settings, units, and source
frame or frame pair. Record enough provenance to reproduce it. Discuss whether
an observed quality change corresponds to an alignment change; correlation
alone does not show that a factor caused the change.
