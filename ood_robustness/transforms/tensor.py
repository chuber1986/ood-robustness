"""Collection of image transformations."""

import random
from collections.abc import Callable

import torch


class PixelTransform:
    """
    Applies a transform function to a singe pixel.

    Methods:
        __call__(self, img: torch.Tensor) -> torch.Tensor: Apply the transformation to a single pixel of an input image.
    """

    def __init__(self, fn: Callable, position: int | tuple[int, int] = -1) -> None:
        self.function = fn
        self.position = position

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply the transformation to the input image.

        Parameters:
            img (PIL.Image.Image): Input grayscale image in "RGB" mode.

        Returns:
            PIL.Image.Image: Transformed image with replaced black and white pixels.
        """
        if img.ndim not in (2, 3):
            raise RuntimeError(
                f"Expected input tensor with 2 or 3 dimensions, {img.ndim} dimensions were provided."
            )

        if img.ndim == 3 and img.shape[0] not in (1, 3):
            raise RuntimeError(
                f"Expected input tensor with 1 or 3 channels, {img.shape[0]} channels were provided."
            )

        if img.ndim == 2:
            img = img.unsqueeze(0)

        _, width, height = img.shape

        if not isinstance(self.position, tuple):
            position = self.position
            if position < 0:
                position = random.randint(0, width*height - 1)

            xpos = position % width
            ypos = position // width
        else:
            xpos, ypos = self.position

        if not (0 <= xpos <= width and 0 <= ypos <= height):
            raise RuntimeError(
                f"Specified position {self.position} is invalid for image of shape {img.shape}."
            )

        img[:, xpos, ypos] = self.function(img[:, xpos, ypos])
        return img
