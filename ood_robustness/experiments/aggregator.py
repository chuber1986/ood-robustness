"""Extending hyperpyper aggregators."""

import copy
import multiprocess as mp
from collections.abc import Callable
from pathlib import Path
from typing import Any, Mapping

from hyperpyper.aggregator import DataAggregator
from torchvision.transforms import Compose

from ood_robustness.transforms import CollectTransformedData, TransformOptions
from ood_robustness.utils.random import seed_everything


class AdvancedDataAggregator(DataAggregator):
    """Aggregator allowing to return intermetiary transformations."""

    def __init__(self, files, transforms, root=None, batch_size=8, num_workers=0):
        super().__init__(
            files,
            Compose(transforms),
            root=root,
            batch_size=batch_size,
            num_workers=0,
        )

        if num_workers > 0:
            self.data_loader.num_workers = num_workers
            self.data_loader.prefetch_factor = 2

        self.collectors: list[CollectTransformedData] = self._get_collectors(transforms)
        for collector in self.collectors:
            collector.reset()

    def _get_collectors(self, transforms):
        return [t for t in transforms if isinstance(t, CollectTransformedData)]

    def transform(self, cache_file=None):
        if cache_file is not None:
            if Path(cache_file).exists():
                self._full_batch = Pickler.load_data(cache_file)
            else:
                self._full_batch = self.__transform()
                Pickler.save_data(self._full_batch, cache_file)
        else:
            self._full_batch = self.__transform()

        for collector in self.collectors:
            self._full_batch |= collector.get_data()

        return self._full_batch

    def __transform(self):
        if self._full_batch is None:
            mini_batches = []
            for batch in self.data_loader:
                mini_batches.append(batch)

            self._full_batch = self.collate_fn(mini_batches)

        return self._full_batch


class ExperimentAggregator:
    """Aggregates results from independet `DataAggregator`s."""

    def __init__(
        self,
        files: list[Path],
        transforms: list[Callable],
        root: Path | None = None,
        batch_size: int = 8,
        num_workers: int = 0,
        num_processes: int = 4,
        seed: int | None = None,
    ):
        self._validate_selectors(transforms)
        self.options: list[str] = self._get_options(transforms)
        self.num_processes = num_processes
        self.seed = seed

        self.aggregators = {}
        for opt in self.options:
            trns = []
            for trn in transforms:
                if isinstance(trn, TransformOptions):
                    trns += copy.deepcopy(trn[opt])
                else:
                    trns.append(copy.deepcopy(trn))

            agg = AdvancedDataAggregator(
                root=root,
                files=files,
                transforms=trns,
                batch_size=batch_size,
                num_workers=0,
            )

            if num_workers > 0:
                agg.data_loader.num_workers = num_workers
                agg.data_loader.prefetch_factor = 2

            self.aggregators[opt] = agg

        if not self.options:
            self.aggregators["base"] = copy.deepcopy(transforms)

    def _get_selectors(self, transforms: list[Callable]) -> list[TransformOptions]:
        return [t for t in transforms if isinstance(t, TransformOptions)]

    def _get_options(self, transforms: list[Callable]):
        selectors = self._get_selectors(transforms)
        if len(selectors) > 0:
            return selectors[0].get_keys()

        return []

    def _validate_selectors(self, transforms: list[Callable]):
        selectors = self._get_selectors(transforms)
        if len(selectors) > 0 and any(
            selectors[0].get_keys() != s.get_keys() for s in selectors
        ):
            raise ValueError(
                "All Options are reqired to provade the same numper of alternatives."
            )

    def fit(self, cache_file=None):
        return self.transform(cache_file)

    def fit_transform(self, cache_file=None):
        return self.transform(cache_file)

    def _transform(self, agg, cache_file=None) -> Any:
        seed_everything(self.seed)
        return agg.transform(cache_file)

    def transform(self, cache_file=None):
        if self.num_processes == 0:
            transformed = {}
            for name, agg in self.aggregators.items():
                transformed[name] = self._transform(agg, cache_file)

            return transformed

        num_processes = min(self.num_processes, mp.cpu_count())
        print(f"Use {num_processes} processes.")
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass
        with mp.Pool(processes=num_processes) as pool:
            args = []
            for v in self.aggregators.values():
                args.append((v, cache_file))

            results = pool.map(lambda x: self._transform(*x), args)

        return dict(zip(self.aggregators.keys(), results))
