# QuerySet sampling

## Core value objects

Geography and time are named objects rather than bare tuples, so longitude
order and temporal selection are explicit at every call site.

```{eval-rst}
.. autoclass:: ocean_taco.GeoBox
   :members:
.. autoclass:: ocean_taco.TimeRange
   :members:
.. autoclass:: ocean_taco.Query
   :members:
.. autoclass:: ocean_taco.PatchSize
   :members:
.. autoclass:: ocean_taco.PatchSpec
   :members:
.. autoclass:: ocean_taco.QuerySet
   :members:
```

## Selection and replay

```{eval-rst}
.. autoclass:: ocean_taco.QueryFilter
   :members:
.. autoclass:: ocean_taco.CoverageRequirement
   :members:
.. autoclass:: ocean_taco.QueryDraw
   :members:
.. autofunction:: ocean_taco.draw_queryset
.. autofunction:: ocean_taco.select_queryset
.. autofunction:: ocean_taco.replay_experiment
```

A QuerySet is a published population. `draw_queryset` records an exact uniform
draw into a `QueryDraw`; `replay_experiment` verifies that record and
reconstructs the same rows.
