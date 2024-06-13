"Stolen from 'https://github.com/claushofmann/hopfield-classifier/blob/vit/hopfield_boosting/encoder/vit.py'"
import timm
import torch
from torch import nn


def vit_cifar_10():

    model = timm.create_model(
        "timm/vit_base_patch16_224.orig_in21k_ft_in1k", pretrained=False
    )
    model.head = nn.Linear(model.head.in_features, 10)
    model.load_state_dict(
        torch.hub.load_state_dict_from_url(
            "https://huggingface.co/edadaltocg/vit_base_patch16_224_in21k_ft_cifar10/resolve/main/pytorch_model.bin",
            map_location="cpu",
            file_name="vit_base_patch16_224_in21k_ft_cifar10.pth",
        )
    )
    model.head = nn.Linear(model.head.in_features, 512)
    return model
