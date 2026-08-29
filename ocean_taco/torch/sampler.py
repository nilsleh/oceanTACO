"""Deterministic shape bucketing for opt-in dense Native batches."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


class ShapeBucketSampler(Sampler[list[int]]):
    """Batch indices with equal native shapes without dropping any sample."""

    def __init__(self, shapes: Sequence[tuple[int, int] | tuple[int, int, int]], *, batch_size: int, seed: int = 0, shuffle: bool = True) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.shapes, self.batch_size, self.seed, self.shuffle = tuple(shapes), batch_size, seed, shuffle

    def __iter__(self) -> Iterator[list[int]]:
        buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for index, shape in enumerate(self.shapes):
            buckets[tuple(shape)].append(index)
        generator = random.Random(self.seed)
        batches: list[list[int]] = []
        for shape in sorted(buckets):
            indices = buckets[shape]
            if self.shuffle:
                generator.shuffle(indices)
            batches.extend(indices[offset : offset + self.batch_size] for offset in range(0, len(indices), self.batch_size))
        if self.shuffle:
            generator.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        buckets: dict[tuple[int, ...], int] = defaultdict(int)
        for shape in self.shapes:
            buckets[tuple(shape)] += 1
        return sum((count + self.batch_size - 1) // self.batch_size for count in buckets.values())
