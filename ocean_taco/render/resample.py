"""Mask-weighted dense-grid resampling with an explicit support threshold."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..geobox import GeoBox
from .native import _mask_payload, crop_dense

_WARNED_UPSAMPLING_TOKENS: set[str] = set()


def _resampled_axis(axis: np.ndarray, output_size: int, *, method: str) -> np.ndarray:
    """Return coordinates of the actual resampling kernel centres.

    PyTorch's bilinear interpolation with ``align_corners=False`` samples at
    half-pixel locations, rather than at a linspace through the first and last
    input centres.  Adaptive pooling has its own integer-bin centres.  Mapping
    those source indices through the native coordinate axis also preserves the
    documented small coordinate jitter found in some Core products.
    """
    values = np.asarray(axis, dtype=np.float64)
    input_size = values.size
    if input_size == 0:
        return np.full((output_size,), np.nan, dtype=np.float32)
    if method == "adaptive_pool":
        starts = np.floor(np.arange(output_size) * input_size / output_size)
        ends = np.ceil((np.arange(output_size) + 1) * input_size / output_size)
        source_index = (starts + ends - 1.0) / 2.0
    else:
        source_index = (np.arange(output_size) + 0.5) * input_size / output_size - 0.5
        # ``interpolate`` uses edge-value padding for positions just beyond an
        # input edge during upsampling.
        source_index = np.clip(source_index, 0.0, input_size - 1.0)
    return np.interp(source_index, np.arange(input_size), values).astype(np.float32)


@dataclass(frozen=True, slots=True)
class Resample:
    """Render one dense source onto a fixed ``(height, width)`` grid.

    ``support_threshold`` has no default by design: support is a scientific
    choice, recorded with materialised outputs, not a hidden interpolation
    detail.
    """

    shape: tuple[int, int]
    support_threshold: float
    method: Literal["auto", "adaptive_pool", "bilinear", "nearest"] = "auto"

    def __post_init__(self) -> None:
        if len(self.shape) != 2 or any(not isinstance(size, int) or size <= 0 for size in self.shape):
            raise ValueError("Resample shape must be a positive integer (H, W) tuple.")
        if not 0.0 <= self.support_threshold <= 1.0:
            raise ValueError("support_threshold must lie in [0, 1].")
        if self.method not in {"auto", "adaptive_pool", "bilinear", "nearest"}:
            raise ValueError("Resample method must be 'auto', 'adaptive_pool', 'bilinear', or 'nearest'.")

    def _resolved_method(self, native_shape: tuple[int, int], *, render_class: str | None = None) -> str:
        """Choose the registered direction/type-aware interpolation operator."""
        if self.method != "auto":
            return self.method
        if render_class == "categorical":
            return "nearest"
        return "adaptive_pool" if all(target <= native for target, native in zip(self.shape, native_shape)) and self.shape != native_shape else "bilinear"

    def render(self, data, box: GeoBox, *, ocean_mask=None, token: str | None = None, source=None, **_: Any) -> dict[str, Any]:
        """Resample values and validity separately, returning data and support."""
        import torch
        import torch.nn.functional as functional

        cropped = crop_dense(data, box)
        values = np.asarray(cropped.values, dtype=np.float32)
        if values.shape[-2] == 0 or values.shape[-1] == 0:
            return self.empty()
        native_shape = values.shape[-2:]
        method = self._resolved_method(native_shape, render_class=getattr(source, "render_class", None))
        if method == "adaptive_pool" and any(target > native for target, native in zip(self.shape, native_shape)):
            raise ValueError("adaptive_pool is only defined when both output axes do not exceed the native shape.")
        valid = np.isfinite(values)
        filled = np.where(valid, values, 0.0)
        input_tensor = torch.from_numpy(filled).unsqueeze(1)
        mask_tensor = torch.from_numpy(valid.astype(np.float32)).unsqueeze(1)
        keyword = {"size": self.shape, "mode": method}
        if method == "bilinear":
            keyword["align_corners"] = False
        if method == "adaptive_pool":
            scaled_data = functional.adaptive_avg_pool2d(input_tensor, self.shape).squeeze(1).numpy()
            support = functional.adaptive_avg_pool2d(mask_tensor, self.shape).squeeze(1).numpy()
        else:
            scaled_data = functional.interpolate(input_tensor, **keyword).squeeze(1).numpy()
            support = functional.interpolate(mask_tensor, **keyword).squeeze(1).numpy()
        # A threshold of zero permits any positive support; it must not turn a
        # destination with no observations into a "valid" NaN cell.
        source_valid = support > 0.0
        support_mask = support >= self.support_threshold
        output_valid = source_valid & support_mask
        output = np.divide(scaled_data, support, out=np.full_like(scaled_data, np.nan), where=support > 0)
        output[~output_valid] = np.nan
        ratio = max(self.shape[0] / native_shape[0], self.shape[1] / native_shape[1])
        if ratio > 2.0 and token is not None and token not in _WARNED_UPSAMPLING_TOKENS:
            warnings.warn(
                f"{token} is being upsampled {ratio:.1f}x from native shape {native_shape}; native shape is recorded in sample metadata.",
                UserWarning,
                stacklevel=2,
            )
            _WARNED_UPSAMPLING_TOKENS.add(token)
        lat = _resampled_axis(cropped["lat"].values, self.shape[0], method=method)
        lon = _resampled_axis(cropped["lon"].values, self.shape[1], method=method)
        payload = _mask_payload(cropped, ocean_mask=None)
        payload.update(
            {
                "data": output.astype(np.float32),
                "source_valid": source_valid,
                "support_mask": support_mask,
                "valid_mask": output_valid,
                "support": support.astype(np.float32),
                "lat": lat,
                "lon": lon,
                "native_shape": native_shape,
            }
        )
        if ocean_mask is not None:
            mask, in_mask_domain = ocean_mask.nearest_with_domain(lat, lon)
            payload["ocean_mask"] = mask
            payload["in_mask_domain"] = in_mask_domain
            payload["valid_mask"] = payload["valid_mask"] & mask[None, :, :]
        return payload

    def empty(self) -> dict[str, Any]:
        """Return a fixed-shape missing grid source."""
        shape = (0, *self.shape)
        return {
            "data": np.full(shape, np.nan, dtype=np.float32),
            "source_valid": np.zeros(shape, dtype=bool),
            "support_mask": np.zeros(shape, dtype=bool),
            "valid_mask": np.zeros(shape, dtype=bool),
            "support": np.zeros(shape, dtype=np.float32),
            "lat": np.full((self.shape[0],), np.nan, dtype=np.float32),
            "lon": np.full((self.shape[1],), np.nan, dtype=np.float32),
            "times": [],
        }

    def render_pair(self, first, second, box: GeoBox, *, ocean_mask=None, token: str | None = None) -> dict[str, Any]:
        """Resample a geodetic vector pair with one shared support footprint."""
        import torch
        import torch.nn.functional as functional

        first_cropped, second_cropped = crop_dense(first, box), crop_dense(second, box)
        for coordinate in ("time", "lat", "lon"):
            if not np.array_equal(first_cropped[coordinate].values, second_cropped[coordinate].values):
                raise ValueError(f"Vector components disagree on their {coordinate!r} coordinates.")
        values = np.stack(
            (np.asarray(first_cropped.values, dtype=np.float32), np.asarray(second_cropped.values, dtype=np.float32)),
            axis=1,
        )
        if values.shape[-2] == 0 or values.shape[-1] == 0:
            return self.empty_pair()
        native_shape = values.shape[-2:]
        method = self._resolved_method(native_shape, render_class="vector_pair")
        if method == "adaptive_pool" and any(target > native for target, native in zip(self.shape, native_shape)):
            raise ValueError("adaptive_pool is only defined when both output axes do not exceed the native shape.")
        source_valid = np.isfinite(values).all(axis=1)
        filled = np.where(source_valid[:, None, :, :], values, 0.0)
        input_tensor = torch.from_numpy(filled)
        mask_tensor = torch.from_numpy(source_valid.astype(np.float32)).unsqueeze(1)
        if method == "adaptive_pool":
            scaled_data = functional.adaptive_avg_pool2d(input_tensor, self.shape).numpy()
            support = functional.adaptive_avg_pool2d(mask_tensor, self.shape).squeeze(1).numpy()
        else:
            keyword = {"size": self.shape, "mode": method}
            if method == "bilinear":
                keyword["align_corners"] = False
            scaled_data = functional.interpolate(input_tensor, **keyword).numpy()
            support = functional.interpolate(mask_tensor, **keyword).squeeze(1).numpy()
        support_mask = support >= self.support_threshold
        output_valid = (support > 0.0) & support_mask
        output = np.divide(
            scaled_data,
            support[:, None, :, :],
            out=np.full_like(scaled_data, np.nan),
            where=support[:, None, :, :] > 0.0,
        )
        output = np.where(output_valid[:, None, :, :], output, np.nan)
        lat = _resampled_axis(first_cropped["lat"].values, self.shape[0], method=method)
        lon = _resampled_axis(first_cropped["lon"].values, self.shape[1], method=method)
        payload: dict[str, Any] = {
            "data": output.astype(np.float32),
            "source_valid": source_valid,
            "support_mask": support_mask,
            "valid_mask": output_valid,
            "support": support.astype(np.float32),
            "lat": lat,
            "lon": lon,
            "times": [str(value) for value in first_cropped["time"].values],
            "native_shape": native_shape,
            "pair_available": bool(output_valid.any()),
        }
        if ocean_mask is not None:
            mask, in_mask_domain = ocean_mask.nearest_with_domain(lat, lon)
            payload["ocean_mask"] = mask
            payload["in_mask_domain"] = in_mask_domain
            payload["valid_mask"] = payload["valid_mask"] & mask[None, :, :]
            payload["pair_available"] = bool(payload["valid_mask"].any())
        if token is not None:
            ratio = max(self.shape[0] / native_shape[0], self.shape[1] / native_shape[1])
            if ratio > 2.0 and token not in _WARNED_UPSAMPLING_TOKENS:
                warnings.warn(
                    f"{token} is being upsampled {ratio:.1f}x from native shape {native_shape}; native shape is recorded in sample metadata.",
                    UserWarning,
                    stacklevel=2,
                )
                _WARNED_UPSAMPLING_TOKENS.add(token)
        return payload

    def empty_pair(self) -> dict[str, Any]:
        """Return a fixed-shape missing vector pair."""
        payload = self.empty()
        payload["data"] = np.stack((payload["data"], payload["data"]), axis=1)
        payload["pair_available"] = False
        return payload
