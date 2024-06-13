"""Transformations extending `hyperpyper` functionality."""

from collections.abc import Callable, Iterable
from copy import deepcopy
from typing import Any, Mapping

import numpy as np


class TransformOptions:
    """Profides alternative transormations for the `ExperimentAggregator`."""

    def __init__(self, transforms: dict[Any, Callable | list[Callable]]):
        self.transforms = {}
        for key, transform in transforms.items():
            if not isinstance(transform, list):
                transform = [transform]

            self.transforms[key] = transform

    def get_keys(self):
        """Return experiment names."""
        return list(self.transforms.keys())

    def __len__(self):
        return len(self.transforms)

    def __iter__(self):
        return iter(self.transforms.items())

    def __getitem__(self, idx):
        return self.transforms[idx]

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            "Pipelines inclution option have to be executed within an 'ExperimentAggregator'."
        )


class CollectTransformedData:
    """Samples from all call inputs and stres them for later review."""

    def __init__(self, name, indices=None):
        self.name = name

        self.index_perm = None
        self.index_rev_perm = None
        self.indices = None

        if indices is not None:
            self.index_perm = np.argsort(indices)
            self.index_rev_perm = np.argsort(self.index_perm)
            self.indices = np.array(indices)[self.index_perm]

        self.reset()

    def __call__(self, data):
        if self.indices is None or self.cur_index in self.indices:
            if isinstance(data, Mapping):
                for k, v in data.items():
                    name = "_".join([self.name, k])
                    if name not in self.collection:
                        self.collection[name] = []
                    self.collection[name].append(v)
            else:
                if self.name not in self.collection:
                    self.collection[self.name] = []
                self.collection[self.name].append(data)

        self.cur_index += 1
        return data

    def reset(self):
        """Resets the collection of the elements. Required after evert pipeline execution."""
        self.cur_index = 0
        self.collection = {}

    def get_data(self) -> dict:
        """Return the collected transformed inputs."""
        if self.index_rev_perm is not None:
            results = {}
            for k, v in self.collection.items():
                results[k] = [v[i] for i in self.index_rev_perm]
            return results
        else:
            return dict(self.collection)


class MultiTransformer:
    """Applies multiple transformation to a input and collects the results."""

    def __init__(
        self, transforms: Iterable[Callable], collect_fn: Callable[[Iterable], Any]
    ):
        self.transforms = transforms
        self.collect_fn = collect_fn

    def __call__(self, data):
        trans = [t(deepcopy(data)) for t in self.transforms]
        return self.collect_fn(trans)
