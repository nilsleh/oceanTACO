# QuerySet sampling

## Core value objects

```{eval-rst}
.. autoclass:: ocean_taco.Query
.. autoclass:: ocean_taco.PatchSize
.. autoclass:: ocean_taco.PatchSpec
.. autoclass:: ocean_taco.QuerySet
```

## Selection and replay

```{eval-rst}
.. autoclass:: ocean_taco.QueryFilter
.. autoclass:: ocean_taco.CoverageRequirement
.. autofunction:: ocean_taco.draw_queryset
.. autofunction:: ocean_taco.select_queryset
.. autofunction:: ocean_taco.replay_experiment
```

A QuerySet is a published population. `draw_queryset` records an exact uniform
draw; `replay_experiment` verifies and reconstructs it. There is no shipped
ad-hoc `QueryGenerator` API.
