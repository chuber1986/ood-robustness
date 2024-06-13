"""Prepare datasets"""

import os
import argparse
import logging
from typing import Any
import pickle
from pathlib import Path
from itertools import product
import multiprocessing as mp

from omegaconf import OmegaConf
import numpy as np
from hyperpyper.utils import FolderScanner
import pyprojroot
import dotenv

from ood_robustness.utils.random import seed_everything
from ood_robustness.utils.config import instantiate, parse_config #, print_config


pyprojroot.find_root(".git")
dotenv.load_dotenv()


def _save_config(cfg, file):
    with file.open("w") as f:
        OmegaConf.save(cfg, f)


def run_experiment(
    config_file, model, dataset, experiment, dev_queue=None, **kwargs
) -> dict[str, Any]:
    if dev_queue is not None:
        kwargs["device"] = dev_queue.get()

    logging.info(f"Read config: {model} - {dataset} - {experiment}")
    overrides = [f"{k}={v}" for k, v in kwargs.items()] + [
        f"model={model}",
        f"dataset={dataset}",
        f"experiment={experiment}",
    ]
    cfg = parse_config(
        config_file,
        overrides,
    )
    # print_config(cfg)

    ofile = Path(cfg.out_path) / model / dataset / experiment
    ofile.mkdir(parents=True, exist_ok=True)

    _save_config(cfg, ofile / "cfg.yaml")
    cfg = instantiate(cfg)

    logging.info(f"Load model: {cfg.model.name}")
    seed_everything(cfg.random_seed - 42)
    module = cfg.model.create(device=cfg.device)

    logging.info(f"Prepare dataset: {cfg.dataset.name}")
    seed_everything(cfg.random_seed)
    files = FolderScanner.get_image_files(
        folders=cfg.dataset.root,
        relative_to=cfg.dataset.root,
        n_samples=cfg.n_samples,
        recursive=True,
    )
    files = set(sorted(files))

    assert len(files) > 0, f"No files found in directory: {cfg.dataset.root}"
    rand_file_indices = np.random.choice(len(files), cfg.n_plot_samples, replace=False)
    # rand_files = PathList([files[i] for i in rand_file_indices])

    logging.info(f"Run experiment: {cfg.experiment.name}")
    seed_everything(cfg.random_seed + 42)
    output = cfg.experiment.function(cfg, module, files, rand_file_indices)

    logging.info("Save results")
    with open(ofile / "output.pkl", "wb") as file:
        pickle.dump(output, file)

    logging.info(f"Finished: {model} - {dataset} - {experiment}")

    if dev_queue is not None:
        dev_queue.put(kwargs["device"])

    return output


def _collect_cfgs(path):
    return [f.stem for f in path.glob("*.yaml")]


def run_all_experiment(config_file, models=None, datasets=None, experiments=None, exp_per_gpu=1):
    logging.info("Run all experiment (using all availible GPUs)")
    devs = os.environ.get("CUDA_VISIBLE_DEVICES").split(",")
    logging.info(f"Found {len(devs)} GPUs")
    gpus = mp.Queue(len(devs) * exp_per_gpu)
    for dev in devs:
        for _ in range(exp_per_gpu):
            gpus.put(f"cuda:{dev}")

    if not len(devs):
        gpus.put("cpu")

    cpath = config_file.parent
    if models is None:
        models = _collect_cfgs(cpath / "model")
    if datasets is None:
        datasets = _collect_cfgs(cpath / "dataset")
    if experiments is None:
        experiments = _collect_cfgs(cpath / "experiment")

    processes = []
    for model, dataset, experiment in product(models, datasets, experiments):
        kwargs = dict(
            config_file=config_file,
            model=model,
            dataset=dataset,
            experiment=experiment,
            dev_queue=gpus,
        )
        p = mp.Process(target=run_experiment, kwargs=kwargs)
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    logging.info("All experiments finished")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        help="path to main configuration file",
        type=Path,
        default=Path(os.environ.get("CONFIG_FILE", "./config/config.yaml")),
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(filename)s:%(lineno)s: %(message)s",
    )

    # run_all_experiment(parse_arguments().cfg, exp_per_gpu=1)

    # Resize \ Rotation
    run_all_experiment(
        parse_arguments().cfg,
        datasets=["shapetastic_gs_rot", "shapetastic_ni_32", "shapetastic_ni_64"],
        experiments=["no_transform"],
        exp_per_gpu=1,
    )
    # Gaussian Blur
    run_all_experiment(
        parse_arguments().cfg,
        datasets=["shapetastic_ni_32"],
        experiments=["gaussianblur"],
        exp_per_gpu=1,
    )
    # Compression
    run_all_experiment(
        parse_arguments().cfg,
        datasets=["shapetastic_cb_cs", "cifar10", "imagenet"],
        experiments=["compression_jpeg"],
        exp_per_gpu=1,
    )
    # Hueshift
    run_all_experiment(
        parse_arguments().cfg,
        datasets=["cifar10"],
        experiments=["hueshift"],
        exp_per_gpu=1,
    )
