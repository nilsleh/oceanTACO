"""Grouped geodetic vector rendering for paired GLORYS components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..geobox import GeoBox
from ..registry import get_modality
from .native import Native, _times, crop_dense
from .resample import Resample


@dataclass(frozen=True, slots=True)
class VectorPair:
    """Render one registered east/north vector pair with shared availability.

    The two components are selected and resampled as a unit.  ``valid_mask``
    therefore describes cells for which both components have support, while
    ``pair_available`` is the corresponding sample-level Boolean.
    """

    renderer: Native | Resample
    components: tuple[str, str] = ("glorys_uo", "glorys_vo")

    def __post_init__(self) -> None:
        if len(self.components) != 2 or self.components[0] == self.components[1]:
            raise ValueError("VectorPair.components must name two distinct registered tokens.")
        first, second = (get_modality(token) for token in self.components)
        if first.render_class != "vector_pair" or second.render_class != "vector_pair":
            raise ValueError("VectorPair components must be registry-declared vector_pair sources.")
        if first.vector_pair_tokens != self.components or second.vector_pair_tokens != self.components:
            raise ValueError("VectorPair components must belong to the same ordered registry pair.")

    def render(self, first, second, box: GeoBox, *, ocean_mask=None, token: str | None = None, **_: Any) -> dict[str, Any]:
        """Render paired components with a common coordinate/support contract."""
        if isinstance(self.renderer, Resample):
            payload = self.renderer.render_pair(first, second, box, ocean_mask=ocean_mask, token=token)
        else:
            first_cropped, second_cropped = crop_dense(first, box), crop_dense(second, box)
            for coordinate in ("time", "lat", "lon"):
                if not np.array_equal(first_cropped[coordinate].values, second_cropped[coordinate].values):
                    raise ValueError(f"Vector components disagree on their {coordinate!r} coordinates.")
            values = np.stack((np.asarray(first_cropped.values), np.asarray(second_cropped.values)), axis=1)
            source_valid = np.isfinite(values).all(axis=1)
            payload = {
                "data": values,
                "source_valid": source_valid,
                "support_mask": np.ones_like(source_valid, dtype=bool),
                "valid_mask": source_valid,
                "lat": np.asarray(first_cropped["lat"].values),
                "lon": np.asarray(first_cropped["lon"].values),
                "times": _times(first_cropped),
                "native_shape": values.shape[-2:],
                "pair_available": bool(source_valid.any()),
            }
            if ocean_mask is not None:
                mask, in_mask_domain = ocean_mask.nearest_with_domain(payload["lat"], payload["lon"])
                payload["ocean_mask"] = mask
                payload["in_mask_domain"] = in_mask_domain
                payload["valid_mask"] = payload["valid_mask"] & mask[None, :, :]
                payload["pair_available"] = bool(payload["valid_mask"].any())
        payload["components"] = self.components
        return payload

    def empty(self) -> dict[str, Any]:
        """Return a structurally valid missing vector pair."""
        if isinstance(self.renderer, Resample):
            payload = self.renderer.empty_pair()
        else:
            payload = self.renderer.empty()
            payload["data"] = np.stack((payload["data"], payload["data"]), axis=1)
            payload["pair_available"] = False
        payload["components"] = self.components
        return payload
