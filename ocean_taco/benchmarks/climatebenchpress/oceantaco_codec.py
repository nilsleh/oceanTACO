"""OceanTACO's shipped archival codec as a ClimateBenchPress ``Compressor``.

Importing this module registers ``OceanTaco`` (``name = "oceantaco"``) into the
ClimateBenchPress ``Compressor.registry`` so it is evaluated head-to-head with the
upstream WASM codecs, under the same metrics and error bounds.

The codec is a *fixed, faithful* replay of the int16 packing OceanTACO ships in
``generate_dataset/new_format_ssh_data.py::get_variable_encoding``: per variable a
``FixedScaleOffset(scale_factor, add_offset) -> int16`` packing (NaN mapped to the
``_FillValue = -32767`` sentinel) followed by ``Zlib(level=5)``. It is *not* adaptive:
``abs_bound_codec`` ignores the requested error bound and always emits the real shipped
scheme, because OceanTACO ships one fixed scale per variable. The reconstruction matches
``xarray``'s CF packing bit-for-bit (verified in the benchmark tests) when the encode
arithmetic is done in float32, exactly as ``xarray``/netCDF does it.

This module must be importable in conda env ``testpy312`` (the ClimateBenchPress env);
it only depends on ``numpy``, ``numcodecs`` and the upstream ``Compressor`` ABC plus the
plain-data packing table in :mod:`config`.
"""

from __future__ import annotations

from collections import defaultdict
from functools import partial

import numcodecs
import numpy as np
from climatebenchpress.compressor.compressors.abc import (
    Compressor,
    ErrorBound,
    NamedPerVariableCodec,
)
from numcodecs.abc import Codec
from numcodecs_combinators.stack import CodecStack

from ocean_taco.benchmarks.climatebenchpress.config import VARIABLE_PACKING

ZLIB_LEVEL = 5


class OceanTacoFixedScaleInt16(Codec):
    """Faithful int16 scale/offset packing filter with NaN -> fill_value.

    Matches ``xarray`` CF packing bit-for-bit: encode rounds
    ``(value - add_offset) / scale_factor`` in float32 and casts to int16, mapping NaN
    to ``fill_value``; decode reverses this and restores NaN at the sentinel.
    """

    codec_id = "oceantaco_fixed_scale_int16"

    def __init__(
        self, scale_factor: float, add_offset: float, fill_value: int = -32767
    ):
        """Store the per-variable packing parameters."""
        self.scale_factor = float(scale_factor)
        self.add_offset = float(add_offset)
        self.fill_value = int(fill_value)

    def encode(self, buf) -> np.ndarray:
        """Pack a float32 array to int16, mapping NaN to ``fill_value``."""
        arr = np.ascontiguousarray(buf, dtype=np.float32)
        nan_mask = np.isnan(arr)
        # Replace NaN before the cast so the int16 cast is well-defined; the sentinel
        # is written back afterwards.
        safe = np.where(nan_mask, np.float32(self.add_offset), arr)
        scaled = np.round(
            (safe - np.float32(self.add_offset)) / np.float32(self.scale_factor)
        )
        packed = scaled.astype(np.int16)
        packed[nan_mask] = self.fill_value
        return packed

    def decode(self, buf, out=None) -> np.ndarray:
        """Unpack int16 back to float32, restoring NaN at the ``fill_value`` sentinel."""
        enc = np.frombuffer(bytes(buf), dtype=np.int16)
        # Match xarray's CF unpacking: int16 * float64 scalar -> float64, then cast to
        # float32. Doing the multiply in float64 (not float32) is what makes the
        # reconstruction bit-for-bit identical to reading the shipped NetCDF.
        dec = (
            enc.astype(np.float64) * self.scale_factor + self.add_offset
        ).astype(np.float32)
        dec[enc == self.fill_value] = np.float32(np.nan)
        if out is not None:
            out_arr = np.asarray(out)
            out_arr[:] = dec.reshape(out_arr.shape)
            return out
        return dec

    def get_config(self) -> dict:
        """Return the numcodecs config dict for (de)serialisation."""
        return {
            "id": self.codec_id,
            "scale_factor": self.scale_factor,
            "add_offset": self.add_offset,
            "fill_value": self.fill_value,
        }


numcodecs.register_codec(OceanTacoFixedScaleInt16)


def _oceantaco_codec(scale_factor: float, add_offset: float, fill_value: int) -> CodecStack:
    """Build the fixed OceanTACO scale/offset + Zlib codec stack for one variable."""
    return CodecStack(
        OceanTacoFixedScaleInt16(scale_factor, add_offset, fill_value),
        numcodecs.Zlib(level=ZLIB_LEVEL),
    )


class OceanTaco(Compressor):
    """OceanTACO's fixed scaled-int16 + zlib archival encoding."""

    name = "oceantaco"
    description = "OceanTACO scaled-int16 + zlib"

    @staticmethod
    def abs_bound_codec(error_bound, *, dtype=None, **kwargs) -> Codec:
        """Satisfy the ABC; the real per-variable codec is built in :meth:`build`.

        OceanTACO ships one fixed scale per variable, so a codec cannot be built from an
        error bound alone (the variable identity is required). The variable-aware path is
        :meth:`build`; this method is only reached if a single-variable dataset omits the
        packing table, which is a configuration error.
        """
        raise RuntimeError(
            "OceanTaco.abs_bound_codec should not be called directly; the variable-aware "
            "build() override constructs the fixed per-variable codec."
        )

    @classmethod
    def build(
        cls,
        dtypes,
        data_abs_min,
        data_abs_max,
        data_min,
        data_max,
        data_min_2d,
        data_max_2d,
        error_bounds: list[dict[str, ErrorBound]],
    ) -> dict[str, list[NamedPerVariableCodec]]:
        """Emit the fixed per-variable codec at every requested error-bound level.

        The codec is identical across levels (a single operating point); emitting it once
        per level lets it line up with the adaptive competitors in the rate-distortion
        sweep. Variable selection uses the OceanTACO packing table keyed by variable name.
        """
        codecs: dict[str, list[NamedPerVariableCodec]] = defaultdict(list)
        for eb_per_var in error_bounds:
            new_codecs = {}
            for var, eb in eb_per_var.items():
                if var not in VARIABLE_PACKING:
                    raise KeyError(
                        f"No OceanTACO packing parameters registered for variable '{var}'. "
                        f"Known variables: {sorted(VARIABLE_PACKING)}."
                    )
                packing = VARIABLE_PACKING[var]
                new_codecs[var] = partial(
                    _oceantaco_codec,
                    packing.scale_factor,
                    packing.add_offset,
                    packing.fill_value,
                )
            error_bound_name = "_".join(
                f"{var}-{eb.name}" for var, eb in sorted(eb_per_var.items())
            )
            codecs[cls.name].append(
                NamedPerVariableCodec(name=error_bound_name, codecs=new_codecs)
            )
        return codecs
