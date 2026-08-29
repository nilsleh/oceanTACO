# PyTorch dataset

## OceanTACODataset

```{eval-rst}
.. autoclass:: ocean_taco.torch.OceanTACODataset
   :members:
```

## CoreSourceLoader

```{eval-rst}
.. autoclass:: ocean_taco.torch.CoreSourceLoader
   :members:

.. autoclass:: ocean_taco.torch.loader.PlannedSourceLoader
   :members:
```

## Native-grid batching

```{eval-rst}
.. autoclass:: ocean_taco.torch.ShapeBucketSampler
   :members:

.. autofunction:: ocean_taco.torch.native_shapes
```

## Collation and worker setup

```{eval-rst}
.. autofunction:: ocean_taco.torch.collate_ocean_samples
.. autofunction:: ocean_taco.torch.native_pad_collate
.. autofunction:: ocean_taco.torch.seed_ocean_taco_worker
```
