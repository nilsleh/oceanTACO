# Renderers and plotting

Renderers define how a native retrieved source becomes a model input. They do
not normalise values or silently turn missing values into zero.

```{eval-rst}
.. autoclass:: ocean_taco.render.Resample
   :members:
.. autoclass:: ocean_taco.render.Native
   :members:
.. autoclass:: ocean_taco.render.Points
   :members:
.. autoclass:: ocean_taco.render.VectorPair
   :members:
.. autofunction:: ocean_taco.plot.plot_ocean_sample
```

`plot_ocean_sample` requires the `viz` extra (`pip install -e ".[viz]"`). It
plots a dense source, a selected vector component, or ragged points from one
unbatched dataset sample.
