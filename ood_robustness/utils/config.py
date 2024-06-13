"""Experiment configuration utils."""

import os
from pathlib import Path

import hydra.utils
import rich
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from rich.tree import Tree
from rich.syntax import Syntax

instantiate = hydra.utils.instantiate


def get_path(key: str, default: str | None = ""):
    p = Path(key)

    if p.exists():
        return p

    path = get_value(key, default)
    p = Path(path)

    if not p.exists():
        raise ValueError(f"Path does not exist: {path}")

    return p


def get_value(key: str, default: str | None = None):
    return os.environ.get(str(key), default)


def parse_config(config_file, overrides=None):
    config_file = Path(config_file).resolve()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(config_file.parent),
        job_name=config_file.parent.stem,
    ):
        return compose(config_name=config_file.stem, overrides=overrides)


def print_config(cfg):
    style = "dim cyan"
    tree = Tree("CONFIG", style=style, guide_style=style)

    # generate config tree from queue
    for field in cfg:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=True)
        else:
            branch_content = str(config_group)

        branch.add(Syntax(branch_content, "yaml"))

    rich.print(tree)
