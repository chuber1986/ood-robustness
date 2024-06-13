"""Prepare datasets"""

import os
import argparse
import logging
import pickle
import shutil
from pathlib import Path
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_utils import create_sample
from hyperpyper.utils import VisionDatasetDumper
from torchvision.datasets import VisionDataset, CIFAR10
from torchvision.transforms import ToPILImage

from ood_robustness.utils.random import seed_everything
from ood_robustness.utils.zenodo import DataDownloader


URL_EBO = r"https://zenodo.org/records/11548962/files/ebo_oe.zip"
URL_HB = r"https://zenodo.org/records/11548962/files/hopfield-boosting.zip"
URL_HB_VIT = r"https://zenodo.org/records/11548962/files/hopfield-boosting-vit.zip"
URL_POEM = r"https://zenodo.org/records/11548962/files/poem.zip"

URL_SHAPETASTIC = r"https://zenodo.org/records/11518866/files/Shapetastic_OOD.zip"


class ImageNet(VisionDataset):

    def __init__(self, root, size=64, transform=None, **_):
        self.root = root
        self.size = size
        self.labels = []

        data_file = self.root / "val_data"
        with open(data_file, "rb") as fo:
            d = pickle.load(fo)

        x = d["data"]
        size2 = size**2
        x = np.dstack((x[:, :size2], x[:, size2 : 2 * size2], x[:, 2 * size2 :]))
        self.data = x.reshape((x.shape[0], size, size, 3))

        self.labels = np.asarray(d["labels"])
        self.targets = np.unique(self.labels)

        self.N = len(self.labels)
        self.transform = transform

    def __getitem__(self, index):
        img = ToPILImage()(self.data[index])
        if self.transform is not None:
            img = self.transform(img)

        return img, self.labels[index]

    def __len__(self):
        return self.N


def _prep_CIFAR10(root, trg):
    dumper = VisionDatasetDumper(CIFAR10, root=root, dst=trg / "test", train=False)
    dumper.dump()


def _prep_ImageNet(root, trg):
    dumper = VisionDatasetDumper(ImageNet, root=root, dst=trg / "test")
    dumper.dump()


def _gen_sample(trg, smpl):
    fn = getattr(shapes, smpl.plot_function)
    trgfile = trg / smpl.plot_function / smpl.file_name
    if trgfile.exists():
        return

    background_color = smpl.background_color
    width = smpl.image_width
    height = smpl.image_height

    params = smpl.drop(
        [
            "plot_function",
            "file_name",
            "background_color",
            "image_width",
            "image_height",
        ]
    ).to_dict()

    for k, v in params.items():
        if v != v:
            params[k] = None

    fig = create_sample(
        width,
        height,
        background_color,
        file_name=trgfile,
        plot_function=fn,
        **params,
    )
    plt.close(fig)


def _gen_partition(src, trg):
    shutil.copy(src, trg)
    samples = pd.read_csv(src)
    trg.mkdir(parents=True, exist_ok=True)
    samples.apply(partial(_gen_sample, trg), axis=1)


def _download_Shapetastic(trg):
    if DataDownloader.download_and_unpack(URL_SHAPETASTIC, trg):
        logging.info("Dataset ready to use.")
    else:
        logging.error("Downloading dataset failed.")
        raise IOError


def _gen_Shapetastic(src, trg):
    # Requires https://github.com/berni-lehner/shapetastic in PYTHONPATH
    import shapes

    for gfile in src.glob("*/*.csv"):
        logging.info(f"Prepare Shapetastic {str(gfile)}")
        cname = gfile.parent.stem
        trgdir = trg / cname
        _gen_partition(gfile, trgdir)


def _prep_Shapetastic(src, trg, generate):
    if generate:
        _gen_Shapetastic(src, trg)
    else:
        _download_Shapetastic(trg)


def download_models(trg):
    s1 = DataDownloader.download_and_unpack(URL_EBO, trg)
    s2 = DataDownloader.download_and_unpack(URL_HB, trg)
    s3 = DataDownloader.download_and_unpack(URL_HB_VIT, trg)
    s4 = DataDownloader.download_and_unpack(URL_POEM, trg)
    if s1 and s2 and s3 and s4:
        logging.info("Models ready to use.")
    else:
        logging.error("Downloading models failed.")
        raise IOError


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--seed", help="random seed for selecting samples", type=int, default=42
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        help="number of samples to prepare",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--model_trg",
        help="target directory for models",
        type=Path,
        default=Path(os.environ.get("MODEL_PATH", "./data/models")),
    )
    parser.add_argument(
        "--cifar10_root",
        help="root to default CIFAR10 (downloaded if not present)",
        type=Path,
        default=Path.cwd() / "data" / "tmp" / "CIFAR10",
    )
    parser.add_argument(
        "--cifar10_trg",
        help="target directory for CIFAR10",
        type=Path,
        default=Path(os.environ.get("CIFAR10_ROOT", "./data/datasets/CIFAR10")),
    )
    parser.add_argument(
        "--imagenet_root",
        help="root to default ImageNet (manuell download required)",
        type=Path,
        default=Path.cwd() / "data" / "tmp" / "ImageNet",
    )
    parser.add_argument(
        "--imagenet_trg",
        help="target directory for ImageNet",
        type=Path,
        default=Path(os.environ.get("IMAGENET_ROOT", "./data/datasets/ImageNet")),
    )
    parser.add_argument(
        "--shapetastic_generate",
        help="generate Shapetastic data instead of downloading them (requires shapetstic repo in PYTHONPATH)",
        action='store_true',
    )
    parser.add_argument(
        "--shapetastic_root",
        help="root to default Sahpetastic",
        type=Path,
        default=Path.cwd() / "data" / "shapetastic_definitions",
    )
    parser.add_argument(
        "--shapetastic_trg",
        help="target directory for Shapetastic",
        type=Path,
        default=Path(os.environ.get("SHAPETASTIC_ROOT", "./data/datasets/shapetastic")),
    )

    return parser.parse_args()


def prepare_all(args):
    logging.info("Download models")
    download_models(args.model_trg)
    logging.info("Prepare CIFAR10")
    seed_everything(args.seed)
    _prep_CIFAR10(args.cifar10_root, args.cifar10_trg)
    logging.info("Prepare ImageNet")
    seed_everything(args.seed)
    _prep_ImageNet(args.imagenet_root, args.imagenet_trg)
    logging.info("Prepare Shapetastic")
    seed_everything(args.seed)
    _prep_Shapetastic(
        args.shapetastic_root, args.shapetastic_trg, args.shapetastic_generate
    )
    logging.info("Preperations done.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(filename)s:%(lineno)s: %(message)s",
    )

    prepare_all(parse_arguments())
