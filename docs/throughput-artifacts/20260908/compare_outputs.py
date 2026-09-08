"""Check the 18-sample oceantaco_equivalence.py corpus against trusted snapshots."""

import argparse
import json
from pathlib import Path

import torch


def assert_equal(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_equal(left, right)
    elif isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    else:
        assert actual == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("optimized", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    # These local snapshots contain PatchSpec objects as well as tensors.
    baseline = torch.load(args.baseline, map_location="cpu", weights_only=False)
    optimized = torch.load(args.optimized, map_location="cpu", weights_only=False)
    assert len(baseline) == len(optimized) == 18
    allowed, changed = [], []
    for index, (actual, expected) in enumerate(zip(optimized, baseline, strict=True)):
        # Fixed corpus order: six Native, six 32x32, six 128x128 samples.
        comparable = dict(actual)
        if index >= 6:
            for field in ("context", "target"):
                if field == "target" and field not in expected:
                    continue
                left = actual if field == "context" else actual[field]
                right = expected if field == "context" else expected[field]
                vector = left["currents"]
                mask = vector["source_valid"]
                assert mask.dtype == torch.bool
                assert mask.shape == vector["valid_mask"].shape
                assert_equal(mask, vector["support"] > 0)
                path = f"samples[{index}].{field}.currents.source_valid"
                allowed.append(path)
                if not torch.equal(mask, right["currents"]["source_valid"]):
                    changed.append(path)
                record = {**vector, "source_valid": right["currents"]["source_valid"]}
                if field == "context":
                    comparable["currents"] = record
                else:
                    comparable[field] = {**left, "currents": record}
        assert_equal(comparable, expected)
    report = {
        "status": "passed",
        "samples": 18,
        "source_outputs_per_context": 11,
        "renderers": ["Native", "Resample(32,32)", "Resample(128,128)"],
        "comparison": "Exact dtype, shape, values and NaN locations; rtol=atol=0",
        "exception": "Only resampled currents.source_valid may differ; must equal support > 0 on the output grid",
        "validated_exception_fields": allowed,
        "changed_exception_fields": changed,
        "native_vectors_and_all_other_fields": "exact match",
    }
    args.json_output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
