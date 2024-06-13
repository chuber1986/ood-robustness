"""Experiment definitions"""

import random
import functools
from collections.abc import Callable, Iterable

import numpy as np
import torch
from hyperpyper.transforms import FileToPIL, PyTorchOutput, ToDevice
from torchvision import transforms

from ood_robustness.experiments import AdvancedDataAggregator, ExperimentAggregator
from ood_robustness.model import OODModelUtil
from ood_robustness.transforms import (
    CollectTransformedData,
    MultiTransformer,
    PixelTransform,
    TransformOptions,
)


def build_pipeline(
    model: OODModelUtil,
    image_transformations: Callable | Iterable[Callable] | None = None,
    tensor_transformations: Callable | Iterable[Callable] | None = None,
    idx_store_image=(),
    idx_store_tensor=(),
    device="cpu",
):
    if image_transformations is None:
        image_transformations = []

    if tensor_transformations is None:
        tensor_transformations = []

    if not isinstance(image_transformations, list):
        image_transformations = [image_transformations]

    if not isinstance(tensor_transformations, list):
        tensor_transformations = [tensor_transformations]

    return [
        FileToPIL(),
        transforms.Resize(32),
        transforms.CenterCrop(32),
        *image_transformations,
        CollectTransformedData("transformed_image", indices=idx_store_image),
        transforms.ToTensor(),
        *tensor_transformations,
        CollectTransformedData("transformed_tensor", indices=idx_store_tensor),
        ToDevice(device),
        PyTorchOutput(model.get_energy_model(), device=device),
        ToDevice("cpu"),
    ]


def single_aggregation(
    cfg, model, files, sample_idx, transformation: Iterable | None = None
):
    if transformation is None:
        transformation = []

    if isinstance(transformation, Iterable):
        transformation = list(transformation)

    pipe = build_pipeline(
        model,
        image_transformations=transformation,
        idx_store_image=sample_idx,
        device=cfg.device,
    )

    agg = AdvancedDataAggregator(
        files,
        pipe,
        root=cfg.dataset.root,
        batch_size=cfg.batch_size,
        num_workers=0,
    )

    return {cfg.experiment.name: agg.transform(cache_file=None)}


def _init_transform(transform, param_name, alpha):
    if not isinstance(transform, functools.partial):
        return transform

    if param_name is None:
        return transform(alpha)

    return transform(**{param_name: alpha})


def multi_aggregation(
    cfg,
    model,
    files,
    sample_idx,
    label_template,
    build_transform_fns,
    param_name,
    alphas,
    is_image_trans=True,
):
    if not isinstance(build_transform_fns, Iterable):
        build_transform_fns = [build_transform_fns]

    if is_image_trans:
        ttag = "image_transformations"
        itag = "idx_store_image"
    else:
        ttag = "tensor_transformations"
        itag = "idx_store_tensor"

    pipe = build_pipeline(
        model,
        device=cfg.device,
        **{
            ttag: TransformOptions(
                {"Raw": lambda x: x}
                | {
                    label_template.format(a): [
                        _init_transform(fn, param_name, a) for fn in build_transform_fns
                    ]
                    for a in alphas
                }
            ),
            itag: sample_idx,
        },
    )

    agg = ExperimentAggregator(
        files,
        pipe,
        root=cfg.dataset.root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        num_processes=cfg.num_processes,
    )

    return agg.transform(cache_file=None)


def pixel_experiments(cfg, model, files, sample_idx, transformer_fn):
    mt = MultiTransformer(
        transforms=[
            PixelTransform(fn=transformer_fn, position=p) for p in range(32 * 32)
        ],
        collect_fn=lambda x: torch.stack(x, 0),
    )
    pipe = build_pipeline(model, tensor_transformations=mt, idx_store_tensor=sample_idx, device=cfg.device)

    # samples = [files[i] for i in sample_idx]
    agg = AdvancedDataAggregator(
        files,
        pipe,
        root=cfg.dataset.root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    output = agg.transform(cache_file=None)

    energies = AdvancedDataAggregator(
        files,
        build_pipeline(model, device=cfg.device),
        root=cfg.dataset.root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    ).transform(cache_file=None)["item"]

    output["heatmap"] = output["item"].reshape(-1, 32, 32).transpose(-1, -2)
    output["item"] = energies

    return output


def advanced_pixel_experiments(
    cfg,
    model,
    files,
    sample_idx,
    label_dec,
    transformer_fns,
    n_samples=None,
):
    if n_samples:
        files = random.sample(files, min(n_samples, len(files)))

    notrans = AdvancedDataAggregator(
        files,
        build_pipeline(model, device=cfg.device),
        root=cfg.dataset.root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    ).transform(cache_file=None)

    lbls = label_dec(files)
    mts = {
        name: MultiTransformer(
            transforms=[
                PixelTransform(fn=transformer_fn, position=p) for p in range(32 * 32)
            ],
            collect_fn=lambda x: torch.stack(x, 0),
        )
        for name, transformer_fn in transformer_fns.items()
    }
    mt = TransformOptions(mts)
    pipe = build_pipeline(model, tensor_transformations=mt, idx_store_tensor=sample_idx, device=cfg.device)

    agg = ExperimentAggregator(
        files,
        pipe,
        root=cfg.dataset.root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        num_processes=cfg.num_processes,
    )

    output = agg.transform(cache_file=None)

    for _, res in output.items():
        res["heatmap"] = res["item"].reshape(-1, 32, 32).transpose(-1, -2)
        res["item"] = notrans["item"]
        res["label"] = lbls

    notrans["heatmap"] = notrans["transformed_image"]

    tmp = {"Raw": notrans}
    tmp |= output
    output = tmp

    return output
