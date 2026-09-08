# Throughput implementation validation — 2026-09-08

The implementation delivers cropped reads before regional merging, a bounded
process-local file LRU, projected ML variables, bulk draw/replay rank resolution,
and batch-local daily crop reuse. Standard PyTorch `DataLoader` still handles
batching and workers; OceanTACO's source loaders handle catalog planning and
source access. See [loader usage](dataset-ml-loader.md).

## Correctness and final review

The final local-data test run passed **143 tests**, with **4 skipped and
1 deselected** (31 warnings, 46.02 s). The repository's default selection excludes
slow/remote tests. [Saved test output](throughput-artifacts/20260908/tests-final.txt).
The original offline baseline was 103 passed, 7 skipped, 1 deselected. Ruff on
changed runtime, benchmark, and test files and `git diff --check` passed.

Regressions cover crop boundaries, coordinate jitter and ties, seam conflicts,
wrapped/empty crops, integer native dtype, actual cropped backend reads, variable
projection, eviction and serialization, ordered draw/replay digests, shared
GLORYS variables, Argo fields, context/target reuse, support thresholds, and
warmed-parent fork/spawn persistent workers. Synthetic SWOT cases exercise
calibration, gap, and science dates. Final review also fixed batch setup cleanup:
an unknown token previously left batch state set; unknown-token and missing-file
failures now both allow a subsequent batch to succeed.

The saved original implementation outputs were compared with freshly generated
final outputs: **18 samples with 11 source outputs each**, plus target records.
Six patch specifications run through Native, 32×32, and 128×128 rendering cover
interior, regional seam, antimeridian, equator, both hemispheres, and missing
context/target dates. Dtypes, shapes, values, NaN locations, coordinates,
availability, time padding, support, and validity compare at zero tolerance.
[Machine-readable comparison](throughput-artifacts/20260908/equivalence-postfix.json).

**User-authorized exception:** resampled vector `source_valid` now equals
`support > 0` on the output grid, using joint component support. The previous
native-grid mask prevented collation when native shapes varied or records were
missing. The checker validates this formula and output mask shape for all
14 resampled context/target vector records; 12 context fields actually differ.
Native vectors and every other field match exactly, including vector data,
support, threshold masks, final validity, and coordinates.

## Recorded local throughput

Default benchmark: 128 rows, batch size 16, four workers, L3 SSH and SWOT,
128 km footprints, 128×128 output. Epoch timing includes worker startup and
retrieval but excludes QuerySet reading, drawing, and planning. Filesystem caches
were uncontrolled, so these are local measurements rather than universal rates.

| Run | Epoch time | Samples/s |
| --- | ---: | ---: |
| Original, repeat 1 | 56.09 s | 2.28 |
| Original, repeat 2 | 58.35 s | 2.19 |
| Optimized, exact defaults | 3.418 s | 37.45 |
| Optimized, persistent workers, epoch 1 | 3.071 s | 41.68 |
| Persistent epochs 2–5 | 1.146–1.256 s | 101.92–111.73 |

The exact-default improvement is **16.4–17.1×**. Both sources remain available
for all 128 samples. Evidence: [original run 1](throughput-artifacts/20260908/oceantaco-baseline-benchmark.txt),
[original run 2](throughput-artifacts/20260908/oceantaco-baseline-repeat.txt),
[optimized defaults](throughput-artifacts/20260908/oceantaco-default-final.json),
and [five persistent epochs](throughput-artifacts/20260908/oceantaco-optimized-final.json).

The persistent run spent 4.743 s reading the QuerySet, 1.252 s drawing, and
0.785 s constructing/planning the dataset; its first batch arrived in 1.876 s.
Epoch 1 recorded 60 process-local opens and 204 hits; every later epoch recorded
zero additional opens and 264 hits. Maximum per-process peak RSS increased from
2.035 to 2.047 GiB over five epochs. Fork RSS includes inherited parent memory;
these values must not be summed as exclusive worker memory.

Additional representative runs completed:

| Footprints / output | Workers / prefetch | Epochs | Samples/s, first → last |
| --- | --- | ---: | ---: |
| Published 256 km / 32×32 | 0 / unused | 3 | 3.81 → 17.39 |
| Published 512 km / 64×64, shuffled | 2 / 1 | 3 | 8.41 → 28.37 |
| Diagnostic 64/128/512 km / 256×256, shuffled | 8 / 2 | 5 | 10.20 → 19.50 |

[256 km results](throughput-artifacts/20260908/oceantaco-256-32-w0.json),
[512 km results](throughput-artifacts/20260908/oceantaco-512-64-w2.json), and
[mixed-source diagnostics](throughput-artifacts/20260908/oceantaco-diagnostics-w8-fixed.json)
retain full configurations, startup timings, delivery/service percentiles,
per-process counters, and availability. The diagnostic run includes sparse/dense
scalars, shared GLORYS SST/currents, Argo, seams, antimeridian, and missing dates.
Some requested Argo/SWOT crops have no observations.

The separate sequential all-source comparison originally took 382.41 s and
12,616,204 KiB peak RSS; the final optimized run took 30.45 s and 801,788 KiB.
This is a different workload from the batch benchmark. The final run overlapped
the test suite, so its timing is diagnostic. [Original log](throughput-artifacts/20260908/oceantaco-equivalence-baseline.txt),
[final log](throughput-artifacts/20260908/equivalence-postfix.txt).

## Scope and reproduction

The local port contains only 2023-03-29. Multi-date coverage uses missing dates
and synthetic multi-day fixtures. This is representative coverage, not a full
factorial sweep. Real science-phase SWOT, remote retrieval, GPU transfer/pinned
memory, and long-duration memory behavior remain unmeasured. Bounded-cache tests
and the five-epoch measurements support cache reuse but do not establish an
absence of all long-run memory growth. No additional long-run benchmark was
required to finish this implementation; workload-specific tuning remains useful.

From the repository root:

```bash
source env.sh
export OCEANTACO_LOCAL_PORT=/p/project1/hai_uqmethodbox/nils/oceanTACO/results/generation_audit_20260828/port_20230329_verified/taco/OceanTACO
python -m pytest tests -q --disable-warnings
python scripts/dev/benchmark_local_draw.py --taco-path "$OCEANTACO_LOCAL_PORT" \
    --json-output /tmp/throughput-default.json
python scripts/dev/benchmark_local_draw.py --taco-path "$OCEANTACO_LOCAL_PORT" \
    --epochs 5 --persistent-workers --json-output /tmp/throughput-persistent.json
```

The archived [generation harness](throughput-artifacts/20260908/oceantaco_equivalence.py)
accepts an implementation source root and output snapshot path. The large trusted
snapshots remain in workspace scratch, while the checker and result are retained
here. To repeat the final comparison:

```bash
python docs/throughput-artifacts/20260908/oceantaco_equivalence.py "$PWD" \
    tmp/throughput-handoff-20260908/oceantaco-equivalence-postfix.pt
python docs/throughput-artifacts/20260908/compare_outputs.py \
    tmp/throughput-handoff-20260908/oceantaco-equivalence-baseline.pt \
    tmp/throughput-handoff-20260908/oceantaco-equivalence-postfix.pt \
    --json-output /tmp/throughput-equivalence.json
```

Benchmark JSON files retain their original output paths as provenance; linked
copies above are the durable artifacts. The historical plan and handoff remain
workspace documents. Changes are uncommitted; no commit, push, or PR was requested.
