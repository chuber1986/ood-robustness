"""Utils for loading OODModels."""

import logging
from pathlib import Path

import torch
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

from ood_robustness.utils.model import evaluate
from ood_robustness.utils.config import get_path
from ood_robustness.energy import ClassifierEnergy


class CNNOODWrapper(nn.Module):
    def __init__(self, cnn, preprocess) -> None:
        super(CNNOODWrapper, self).__init__()
        self.preprocess = preprocess
        self.module = cnn

    def forward(self, x):
        assert x.shape[-2:] == (32, 32)  # ensure CIFAR-10 size
        with torch.no_grad():
            x = self.preprocess(x)
        x = self.module(x)
        return x


# helper function to load a model
def model_file(model_path, version=99, name="model"):
    """Return model checkpoint file."""
    file = model_path / str(version) / (name + ".ckpt")
    return file


def load_model(file, map_location="cpu"):
    """Load model checkpoint."""
    assert file.exists(), f"Model file '{str(file)}' does not exist."
    return torch.load(file, map_location=map_location)


def create_model(model_path, model_config="cfg.yaml", device="cuda"):
    """Create the model as specified in the model config."""
    with open(model_path / model_config) as p:
        config = OmegaConf.create(yaml.safe_load(p))

    resnet = instantiate(config["model"])
    projection_head = instantiate(config["projection_head"])
    classifier = nn.Linear(512, config.num_classes)

    return (
        config,
        resnet.eval().to(device),
        projection_head.eval().to(device),
        classifier.eval().to(device),
    )


def load_model_weights(
    model_path, resnet, projection_head, classifier, version=99, device="cuda"
):
    """Load weigths for backbone, projection and classification head."""
    resnet.load_state_dict(load_model(model_file(model_path, version=version), device))
    projection_head.load_state_dict(
        load_model(
            model_file(model_path, name="projection_head", version=version), device
        )
    )
    classifier.load_state_dict(
        load_model(model_file(model_path, name="classifier", version=version), device)
    )

    return (
        resnet.eval(),
        projection_head.eval(),
        classifier.eval(),
    )


def load_model_embeddings(model_path, version=99, device="cuda"):
    """Load embeddigs for the energy model."""
    return load_model(
        model_file(model_path, name="embeddings", version=version), device
    )


class OODModelUtil:
    """Wrapper class providon convinience method for working with OODModels."""

    def __init__(
        self,
        model_path,
        version: int | str = ".",
        load_weights: bool = True,
        load_embeddings: bool = True,
        use_cifar10_idd: bool = True,
        use_imagenet_ood: bool = True,
        device: str = "cpu",
    ) -> None:
        self.model_path = Path(model_path)

        self.weights_loaded = False
        self.embeddings_loaded = False
        self.use_cifar10_idd = use_cifar10_idd
        self.use_imagenet_ood = use_imagenet_ood
        self.device = device
        self.version = version

        self.config, self.backbone, self.ood_head, self.cls_head = create_model(
            self.model_path, device=device
        )

        
        # Fix EBO config
        if self.version in ["2ry2vylx", "cuh3svnb", "f51su832", "l0l3spqr", "uqldmzsz"]:
            self.energy_factory = ClassifierEnergy
        else:
            self.energy_factory = instantiate(self.config.energy)

        self.idd_embeddings_tensor = None
        self.ood_embeddings_tensor = None
        self.energy_model = None
        self.ref_idd_energy = None
        self.ref_ood_energy = None

        self.compute_energy_model(load_weights, load_embeddings)

    def load_model_weights(self):
        """Load  model weights."""
        self.weights_loaded = True
        load_model_weights(
            self.model_path,
            self.backbone,
            self.ood_head,
            self.cls_head,
            version=self.version,
            device=self.device,
        )

    def load_model_embeddings(self):
        """Load energy embeddings."""
        self.embeddings_loaded = True
        embd = load_model_embeddings(
            self.model_path,
            version=self.version,
            device=self.device,
        )
        self.idd_embeddings_tensor = embd["idd_embeddings"]
        self.ref_idd_energy = embd["idd_energy"]
        self.ood_embeddings_tensor = embd["ood_embeddings"]
        self.ref_ood_energy = embd["ood_energy"]

    def _save_model_embeddings(self):
        file = model_file(self.model_path, name="embeddings", version=self.version)
        if file.exists():
            return

        idd_energy = self.get_idd_reference_energy()
        ood_energy = self.get_ood_reference_energy(max_samples=len(idd_energy))

        torch.save(
            obj={
                "idd_embeddings": self.idd_embeddings_tensor.to("cpu"),
                "idd_energy": idd_energy,
                "ood_embeddings": self.ood_embeddings_tensor.to("cpu"),
                "ood_energy": ood_energy,
            },
            f=file,
        )

    def _get_idd_dataloader(self, train: bool):
        try:
            root = get_path("CIFAR10_ROOT" if self.use_cifar10_idd else "CIFAR100_ROOT")
        except ValueError as e:
            raise ValueError(
                "Path to IDD - CIFAR - dataset has to be provided vai 'CIFAR_ROOT' environmental variable."
            ) from e

        if self.use_cifar10_idd:
            idd_train = CIFAR10(
                root=str(root),
                train=train,
                transform=transforms.Compose([transforms.ToTensor()]),
            )
        else:
            idd_train = CIFAR100(
                root=str(root),
                train=train,
                transform=transforms.Compose([transforms.ToTensor()]),
            )

        return torch.utils.data.DataLoader(
            idd_train, batch_size=1000, shuffle=True, drop_last=False
        )

    def _get_ood_dataloader(self, train: bool):
        # create Imagenet dataset + loader
        if self.use_imagenet_ood:
            try:
                root = get_path("IMAGENET_ROOT")
            except ValueError as e:
                raise ValueError(
                    "Path to OOD - ImageNet - dataset has to be provided vai 'IMAGENET_ROOT' environmental variable."
                ) from e

            # print("Using ImageNet for OOD")
            try:
                from hopbield_boosting.data.datasets import ImageNet
            except ImportError:
                logging.error(
                    "Requirs 'opfield_boostring' from 'https://github.com/claushofmann/hopfield-classifier.git'."
                )

            ood_data = ImageNet(
                root=root.parent / "Imagenet64_train",
                transform=transforms.Compose(
                    [
                        # transforms.Resize(32),
                        transforms.CenterCrop(32),
                        transforms.ToTensor(),
                    ]
                ),
            )
        else:
            try:
                root = get_path("CIFAR10_ROOT")
            except ValueError as e:
                raise ValueError(
                    "Path to OOD - CIFAR - dataset has to be provided vai 'CIFAR_ROOT' environmental variable."
                ) from e

            # print("Using CIFAR-10 with random augmentation for OOD")
            ood_data = CIFAR10(
                root=root,
                train=train,
                transform=transforms.Compose(
                    [
                        transforms.Resize(32),
                        transforms.CenterCrop(32),
                        transforms.ToTensor(),
                    ]
                ),
            )

        return torch.utils.data.DataLoader(ood_data, batch_size=1000, shuffle=True)

    def compute_energy_model(
        self, load_weights: bool = False, load_embeddings: bool = False
    ):
        """Compute or load energy model."""
        if load_weights:
            assert not self.weights_loaded, "Weights are already loaded."
            self.load_model_weights()

        if load_embeddings:
            assert not self.embeddings_loaded, "Weights are already loaded."
            self.load_model_embeddings()
        else:
            self.idd_embeddings_tensor = self.ood_embeddings(
                self._get_idd_dataloader(train=True), to_cpu=False
            )
            self.ood_embeddings_tensor = self.ood_embeddings(
                self._get_ood_dataloader(train=True),
                max_samples=len(self.idd_embeddings_tensor),
                to_cpu=False,
            )

        self.energy_model = self.energy_factory(
            self.idd_embeddings_tensor,
            self.ood_embeddings_tensor,
        )

        self._save_model_embeddings()

    def get_idd_reference_energy(self, max_samples=None, to_cpu: bool = True):
        """Return reference energies from InDistribution samples."""
        if self.ref_idd_energy is None:
            self.ref_idd_energy = self.ood_energy(
                self._get_idd_dataloader(train=False),
                max_samples=max_samples,
                to_cpu=False,
            )

        ref = self.ref_idd_energy[:max_samples]
        if to_cpu:
            ref = ref.cpu()
        return ref

    def get_ood_reference_energy(self, max_samples=None, to_cpu: bool = True):
        """Return reference energies from OutOfDistribution samples."""
        if self.ref_ood_energy is None:
            self.ref_ood_energy = self.ood_energy(
                self._get_ood_dataloader(train=False),
                max_samples=max_samples,
                to_cpu=False,
            )

        ref = self.ref_ood_energy[:max_samples]
        if to_cpu:
            ref = ref.cpu()
        return ref

    def get_projection_model(self):
        """Return projection model."""
        return nn.Sequential(self.backbone, self.ood_head)

    def get_classification_model(self):
        """Return classification model."""
        return nn.Sequential(self.backbone, self.cls_head)

    def get_energy_model(self):
        """Return energy model."""
        assert self.energy_model, "Energy model was not computed."
        return nn.Sequential(self.backbone, self.ood_head, self.energy_model)

    def ood_embeddings(self, dataloader, max_samples=None, to_cpu: bool = True):
        """Return OOD embeddings."""
        ood_model = self.get_projection_model()
        res = evaluate(dataloader, ood_model, device=self.device, max_samples=max_samples)
        if to_cpu:
            res = res.cpu()

        return res

    def cls_emdeddings(self, dataloader, max_samples=None, to_cpu: bool = True):
        """Return Classification embeddings."""
        cls_model = self.get_classification_model()
        res = evaluate(dataloader, cls_model, device=self.device, max_samples=max_samples)
        if to_cpu:
            res = res.cpu()

        return res

    def ood_energy(self, dataloader, max_samples=None, to_cpu: bool = True):
        """Return OutOfDistribution energies."""
        model = self.get_energy_model()
        res = evaluate(dataloader, model, device=self.device, max_samples=max_samples)
        if to_cpu:
            res = res.cpu()

        return res
