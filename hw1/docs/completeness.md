# Optional map coverage evaluation

The main assignment score evaluates the estimated camera trajectory. A map
coverage measure can provide additional context by considering both how well
the reconstruction agrees with a reference and how much of that reference it
covers.

If you use the supplementary tooling, follow the function contracts in
`completeness.py` and report the distance tolerance, reference capture, units,
and sampling choices. Build the reference from the designated clean data and
keep it fixed while comparing runs. Do not use ground-truth alignment fitting
to hide trajectory or map drift.

This evaluation is supplementary. It does not replace the required trajectory
metric or justify omitting frames that are difficult to reconstruct.
