# `api.py` — experiment and inspection commands

The local command-line tools help organize capture measurements and experiment
records. Work with local files; there is no service to configure.

## Typical workflow

1. Inspect the capture directory and confirm that its RGB, depth, and camera
   metadata are present and consistent.
2. Create an experiment declaration using the provided command scaffold.
3. Complete the declaration with the factors, settings, and predictions required
   by the assignment.
4. Run the measurement command and inspect its report for missing or invalid
   evidence.
5. Reconstruct the capture and record the evaluation results when requested.
6. Use the read-only inspection and query commands to check your records.

Use `python api.py --help` and each subcommand's help output for the current
arguments. Keep experiment inputs reproducible, and create a new declaration
when changing settings so results remain interpretable.

## Data model guidance

The ontology in `ontology/hw1.ttl` defines the vocabulary for batches, frames,
measurements, settings, and reconstruction runs. Read the vocabulary and the
assignment handout before adding data. Link each result to its source capture
and relevant settings, keep ordered frame pairs ordered, and retain measurement
units and evaluation status. Use the provided validation tools to catch missing
or inconsistent information.

## Quality measurements

The measurement functions and their contracts are summarized in
`docs/factors.md`. The CLI should record the values and evidence returned by
those functions without changing their meaning. See `docs/triplestore.md` for
student guidance on structuring and checking experiment records.
