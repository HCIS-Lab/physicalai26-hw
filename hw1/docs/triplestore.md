# Triple-store recording strategy

How `api.py` records experiments as RDF: the IRI scheme, the write-once file
lifecycle, the TBox as authority, baked Pass/Fail statuses, and how results
are read back. This is reference material — most readers only need the first
three sections; the rest is here when a test or an error message points at it.

## Big picture

Everything lives in one RDF graph per file, with no named graphs and no
server. Isolation between experiments comes from naming, not from storage:

- **Shared structure** names pixels on disk. Every experiment over one capture
  points at the *same* frame IRIs.
- **Experiment-scoped nodes** carry the experiment's name. Two experiments over
  one batch produce disjoint annotation, pair, and run sets over identical
  frames — nothing collides, nothing is overwritten.

That split is the one rule never to relax.

## Namespaces

The vocabulary IRI is `http://taica.course/hw1/ontology#` (`hw1:`); instance
data uses `http://taica.course/hw1/data/`. The namespace list must match
`ontology/hw1.ttl` exactly:

| Prefix | Namespace | Why it is there |
|---|---|---|
| `hw1:` | ontology + data (see above) | the assignment vocabulary and its instances |
| `schema:` | `https://schema.org/` | `schema:contentUrl` locates the PNG behind an image node |
| `qudt:` / `unit:` | qudt.org schema / vocab | units, so a value is never a bare number nobody can check |
| `skos:` | W3C SKOS core | marks the four closed vocabularies (Status, Polarity, SettingRole, SelectionMode) as vocabularies |
| `prov:` | W3C PROV | `prov:wasDerivedFrom` links a corrupted capture to its source — between batches only |

There is deliberately no `dqv:` prefix: the v1 Result/Metric shape it carried
is gone.

## IRI scheme (frozen)

`<ns>` below is the data namespace `http://taica.course/hw1/data/` for v5
(`schemaVersion "5.x"`) experiments — the form `api.py declare` scaffolds.
Legacy v4 artifacts use the ontology namespace
`http://taica.course/hw1/ontology#` in the same shapes; readers accept both.

Shared structure — identical for every experiment over one capture:

| Node | Shape |
|---|---|
| batch | `<ns>batch/<name>` |
| frame | `<ns>batch/<name>/frame/<n>` |
| rgb / depth image | `<ns>batch/<name>/frame/<n>/rgb` (or `/depth`) |
| generation setting | `<ns>batch/<name>/setting/<param>` |

Experiment-scoped — minted by assessing one declaration:

| Node | Shape |
|---|---|
| experiment | `<ns>experiment/<expname>` |
| factor setting (default) | `<ns>experiment/<expname>/setting/<factor>/<param>` |
| frame annotation | `<ns>experiment/<expname>/annotation/<n>/<kind>` (`rgb` \| `depth`) |
| frame pair | `<ns>experiment/<expname>/pair/<i>_<j>` |
| run | `<ns>experiment/<expname>/run/<mode>` (`baseline` \| `selected` \| `masked`) |
| factor occurrence | `<ns>experiment/<expname>/factor/<local>` |

Naming rules:

- `<name>` is `floor<floor>_<capture-dir-basename>` (e.g. `floor1_baseline`).
  The floor qualifier exists because both floors ship same-named captures.
- `<n>` is the integer file stem, decimal and **unpadded**: `frame/7` and
  `frame/007` would be different IRIs, so there is exactly one spelling.
- `<expname>` is the declaration file's stem, limited to `[A-Za-z0-9_-]+` —
  it must survive both a filesystem and an IRI, and it is also the
  `hw1:batchName`-style identity of the experiment.
- Pairs are **ordered** (`sourceFrame` → `targetFrame`) and named by **stems,
  not ordinals**: `41_43` over a capture gap stays visibly gapped. Ordering
  for queries uses the integer `hw1:pairIndex`, never the IRI string.
- Settings carry the **primary-factor** segment so the IRI reads as a
  sentence ("under HighlightClipping, tauHi was 250"), even though parameter
  names are already globally unique.
- Annotations are **per modality**: an experiment that measures only depth
  says nothing about RGB, so no node is minted for it. A missing property is
  therefore a bug, never a state.

### Parse IRIs in exactly one place

`reconstruct.py` must map a frame IRI back to an integer stem (the
`hw1:frameIndex` lives in the batch graph, which reconstruct never reads).
That tail parse exists **once**, in `frame_index_from_iri`, with
`batch_name_from_frame_iri` as its companion. No other module may split an
IRI string. The parse is strict on purpose: component IRIs
(`.../frame/7/depth`) and non-decimal tails raise instead of returning a
plausible-looking number, because a silently short frame list shows up as a
slightly worse score rather than an error.

## File lifecycle: write-once, sealed

An experiment file is `<student declaration>` + `MACHINE_MARKER` +
`<machine section>`:

1. The student authors the declaration (batch, factor selection, thresholds,
   prediction TODOs).
2. `experiment` appends the machine section **once**. A file that already has
   the marker is a hard error.
3. `write_run` is the only accepted mutation afterwards. It re-serializes
   only the machine half (comments and prefix order there may change; the
   declaration above the marker is preserved byte for byte).

A `hw1:declarationDigest` (SHA-256 of the declaration half) seals the file.
Both writers verify it before touching anything, so editing the declaration
after assessment is refused rather than silently scored.

Machine-section contents: one `FrameAnnotation` per frame per measured
modality (values plus baked statuses, plus the aggregate
`hw1:qualificationStatus` — Pass iff every selected factor of that modality
passed, vacuously Pass when none was selected), one `FramePair` per
consecutive pair (same treatment for pair factors), the defaulted
`FactorSettings` completing the required set, and any exported drop-mask
artefacts (`hw1:maskFile`, PNGs where 255 means drop).

## Parameters and the TBox as authority

Parameter names, factor names, polarities, and thresholds are **not** string
constants in the code. They are read from `ontology/hw1.ttl` at run time, so
the ontology is obeyed rather than merely described. A misspelled parameter
is a hard error with the declared list printed — never a silent extra
setting.

Three pieces implement this:

- `load_parameter_declarations` — what may be set and how each value parses.
- `load_quality_factors` — what is measured, which direction is better, and
  which parameter is its threshold.
- `status_for` — the single implementation of the Pass rule: compare the
  (stored-precision) value against the threshold per the factor's polarity;
  non-finite values fail closed.

Each parameter has a `hw1:paramRole` that says where its setting lives and
what to fix when attribution names it:

| Role | Lives on | Meaning when blamed |
|---|---|---|
| GenerationSetting | the Batch (optional `batch2ttl --gen` sidecar) | how the pixels were produced → regenerate the data |
| MeasurementSetting | the Experiment (declaration) | how pixels are scored → write a new declaration |
| QualificationSetting | the Experiment (declaration) | the Pass/Fail threshold → write a new declaration |

The declaration must supply a **selection-scoped required set**: every
parameter of every selected factor, plus the run parameters
(`icpBackend`, `maxMapMeanL2`, …). Missing ones are filled with recorded
defaults below the marker so the file always states the full treatment.

## Numbers: stored at graded precision

`rdflib`'s Turtle serializer re-derives an `xsd:double`'s lexical form from
its value at about seven significant digits. A value graded in memory could
therefore disagree with the value read back from the file — and since the
verdict is baked next to the value, the file would contradict itself
(observed in practice: depth rasters are millimetre-quantised, so medians
land exactly on round thresholds often).

Two helpers close the gap, and every double goes through them:

- `_storable` rounds to the precision the file can hold, so the stored number
  *is* the graded number. Non-finite values pass through untouched.
- `_double_literal` is the one constructor for `xsd:double`. `INF`, `-INF`,
  and `NaN` are spelled by hand in XSD-valid form: Python's `'inf'` is not
  valid `xsd:double`, and an ill-typed literal makes every SPARQL comparison
  on it silently false — exactly the fail-closed rows would vanish from
  results instead of erroring.

## Reading: `read_experiment`

The canonical reader returns the declaration (experiment name, batch,
selection, settings), the values with statuses, and three derived maps:

- `frame_status` — per-frame verdicts as the **conjunction** over that
  frame's annotation statuses. The per-frame verdict is never stored. A frame
  that appears only as a pair endpoint (no annotation) is vacuously usable:
  with no claim about it, there is nothing to fail.
- `pair_status` — per-pair verdicts in capture order.
- `usable_links` — the conjunction over all selected factors, computed by the
  **shared local-SPARQL usable-link query** that `reconstruct.py` also uses:
  a link is usable iff the pair passed and both endpoint frames are usable.

Malformed input fails loudly (missing statuses, missing pair endpoints or
indices) instead of silently dropping rows from a query result.

## Selection: usable links into contiguous segments

`cut_contiguous_segments` chains usable links into maximal contiguous
segments. Every maximal chain is kept, whatever its length — there is no
minimum-segment floor, since that would encode a property of the ICP backend
rather than of the data. An empty selection means no selected run is written
at all: an `INF` run would conflate "nothing to reconstruct" with
measurement failure in the same triple.

A student may override the default selection with `--selection-query`, a
local SPARQL `SELECT` binding `?frame` (frame IRI) or `?frameIndex`
(integer). Results are validated against the experiment's frames (foreign
frames and `?frame`/`?frameIndex` disagreements rejected) and cut into
contiguous segments, so a personal policy cannot smuggle in hidden temporal
jumps.

## Runs: `write_run` is the one writer

`write_run` is the sole writer of `hw1:ReconstructionRun` triples. It is
idempotent per key: re-running replaces the run's own values instead of
accumulating duplicates, and a later `coverageF` never erases `mapMeanL2`.
Run modes pin to selection-mode individuals (`baseline`→FullBatch,
`selected`→GoodSegments, `masked`→MaskFiltered) through one shared dict so
the mapping cannot drift between callers.

Each run records `mapMeanL2` (a trajectory metric, despite the legacy name),
`gatedSteps`, and — for selected runs — `spliceCount`/`maxGapLength` plus
one `hw1:usedFrame` per consumed frame and `hw1:runFrameCount`, so frame
provenance is answerable from the experiment alone. `write_run` also grades
`hw1:mapMeanL2Status` from the experiment's own threshold; callers never
compare values to thresholds. A missing-GT `inf` is written as
`"INF"^^xsd:double` and therefore grades FAILED — fail-closed, like every
measurer. Run metadata beyond a small allow-list (gate/splice/gap
diagnostics) is rejected so a typo cannot mint a predicate no reader knows.

## The batch graph and `declare`

`build_batch_graph` emits the structural graph for a capture: batch node,
frame nodes with integer `frameIndex`, and image nodes carrying
`schema:contentUrl` **and nothing else** — observables live on
experiment-scoped annotations, because two experiments cannot both own one
image node. Non-consecutive stems are warned about (pairs spanning a gap are
wider motions, and the pair factors will correctly read them as such).
Generation settings and `prov:wasDerivedFrom` may be recorded here as
provenance; they are applied by nothing, and derived-from edges exist only
between batches — never between experiments.

`cmd_declare` scaffolds the declaration (prefixes, Experiment node, batch
join, factor menu, prediction TODOs) and `read_declaration` validates a
hand-written one: known batch and factors, the required setting set, and an
honest prediction block.

## CLI reference

`api.py` subcommands: `declare`, `experiment`, `explore`, `query`,
`batch2ttl` (see `docs/api.md` for the workflow). `explore` renders terminal
tables over captures, declarations, assessed experiments, and side-by-side
comparisons, including the computed verdict that maps a failing run back to
the culprit setting and its role — i.e. what to fix.

## Function index

| Area | Functions |
|---|---|
| IRI minting | `batch_iri`, `frame_iri`, `component_iri`, `generation_setting_iri`, `experiment_iri`, `setting_iri`, `annotation_iri`, `pair_iri`, `run_iri`, `factor_iri`, `data_*_iri` |
| IRI parsing | `frame_index_from_iri`, `batch_name_from_frame_iri`, `_batch_name_from_batch_iri`, `_experiment_name_from_iri`, `batch_name` |
| TBox + grading | `load_parameter_declarations`, `load_quality_factors`, `status_for`, `parse_setting_arg` |
| Numbers | `_storable`, `_double_literal`, `_setting_value_literal`, `_setting_value_to_python` |
| Marker + seal | `_marker_offset`, `_split_sections`, `_declaration_digest`, `_verify_seal`, `_sole_experiment` |
| Machine section | `build_machine_graph`, `_build_semantic_machine_graph`, `_write_machine_section`, `_write_observable`, `_measured`, `_measured_with_mask`, `_write_mask_artifact`, `_fail_closed_value`, `_status_property`, `_require_settings` |
| Reading + selection | `read_experiment`, `query_graph`, `usable_links_from_query`, `cut_contiguous_segments`, `_experiment_settings`, `_annotation_frame_index` |
| Runs | `write_run`, `write_pair_measurements`, `_SELECTION_MODE`, `_RUN_METADATA` |
| Declaration + batch | `cmd_declare`, `read_declaration`, `build_batch_graph`, `cmd_experiment`, `cmd_batch2ttl`, `cmd_explore`, `cmd_query`, `_check_batch_file`, `_resolve_*`, `_warn_stem_gaps`, `_pair_frames` |
