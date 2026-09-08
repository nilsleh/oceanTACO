# Train/eval QuerySet splits

OceanTACO can make draws reproducible; it cannot infer an experiment's
leakage boundary. A `QuerySet` stores a position/date population and coverage
evidence, while a `QueryFilter` selects a subset of that population. Decide
the split before drawing, retain both experiment records, and ensure that a
row (or its scientific dependency) does not appear on both sides.

Published QuerySets are named `<patch-size-km>-<kind>` (for example
`256-eval`), and `QuerySet.from_hub` fetches one by name. The kind is a release
label, not a proof that any downstream model has no leakage.

## Date-held-out split

Use non-overlapping anchor-date windows. Context and forecast lead must also
remain inside each QuerySet's canonical date domain.

```python
from ocean_taco import QueryFilter, QuerySet, draw_queryset

train = QuerySet.from_hub(256, "training")
evaluation = QuerySet.from_hub(256, "eval")

train_draw = draw_queryset(
    train,
    requested_row_count=8,
    seed=1,
    record_path="runs/train.json",
    query_filter=QueryFilter(date_end="2024-10-31"),
)
eval_draw = draw_queryset(
    evaluation,
    requested_row_count=8,
    seed=2,
    record_path="runs/eval.json",
    query_filter=QueryFilter(date_start="2024-11-01"),
)
```

If context windows cross the boundary, leave a temporal buffer at least as
wide as the maximum context offset and forecast lead. Source products may also
have temporal interpolation or assimilation dependencies that are wider than
the loader's explicit context window.

## Box-held-out split

Use disjoint `GeoBox` filters when the generalisation target is geographic.
Antimeridian boxes must state `wraps_antimeridian=True`.

```python
from ocean_taco import GeoBox, QueryFilter

train_filter = QueryFilter(box=GeoBox(-80.0, -30.0, 10.0, 45.0))
eval_filter = QueryFilter(box=GeoBox(140.0, -150.0, 10.0, 45.0, wraps_antimeridian=True))
```

Nearby patches can overlap in their physical footprint even when their centre
coordinates lie on different sides of a box boundary. Add a guard band of at
least the patch radius (and more when the task has spatial autocorrelation),
or use distinct published position populations.

## Published-kind split

The most convenient default is one published `*-training` QuerySet for
training and the matching `*-eval` QuerySet for evaluation. Check their
headers and records in version control:

```python
assert train.header["kind"] == "training"
assert evaluation.header["kind"] == "eval"
assert train.queryset_id != evaluation.queryset_id
```

This protects the release-level split only. It does not prevent leakage from
preprocessing fitted on all rows, duplicate external labels, a shared
normalisation statistic, or source products whose construction uses data over
a wider window. Fit transforms on the train draw only and record the exact
QuerySet IDs, filters, seeds, renderer settings, and source revision.
