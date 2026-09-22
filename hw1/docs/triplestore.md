# Recording experiment evidence with RDF

The assignment uses Turtle and the vocabulary in `ontology/hw1.ttl` to make
captures, measurements, settings, and reconstruction runs inspectable. This
guide gives a workflow for using that model; the ontology and function
signatures are the authority for exact property names and required fields.

## Work with the supplied vocabulary

- Inspect the ontology before writing triples. Reuse its classes, properties,
  and controlled values instead of inventing near-duplicate terms.
- Keep capture-level information distinct from experiment-specific
  measurements and settings.
- Link each measurement to the frame or ordered frame pair it describes.
- Include units and the settings needed to interpret a numeric value.
- Preserve provenance from input capture through analysis and reconstruction.

## Build and validate a declaration

Start from the declaration scaffold produced by the CLI. Complete the required
selection, settings, and prediction fields using the assignment specification.
Run the validator before measurement, then inspect the generated evidence after
measurement. Resolve missing fields and invalid links at their source rather
than editing generated output by hand.

## Query and compare

Use the read-only query command to answer questions about your own records.
Check that queried measurements refer to the intended capture and settings,
and compare experiments only when their inputs and measurement choices are
understood. Report the query purpose and summarize what the returned records
show.

## Debugging checklist

- Are all resource identifiers tied to the correct capture and experiment?
- Does every result point to its source frame or ordered pair?
- Are units and parameter values present where the ontology requires them?
- Do the recorded values agree with the measurement outputs?
- Does validation identify any missing or malformed information?
