"""Collection of image transformations."""

import random
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.filters import gaussian
from torchvision.transforms import functional as F
from wand.image import Image as WImage


class BWToRandColor:
    """
    Transform to replace black and white pixels with random colors in an RGB image.

    Methods:
        __call__(self, img: Image.Image) -> Image.Image: Apply the transformation to the input image.
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply the transformation to the input image.

        Parameters:
            img (PIL.Image.Image): Input grayscale image in "RGB" mode.

        Returns:
            PIL.Image.Image: Transformed image with replaced black and white pixels.
        """
        if img.mode == "RGB":
            # Generate random colors for black and white replacement
            color1 = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            color2 = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )

            # Convert image to NumPy array
            img_array = np.array(img)

            # Create a mask for black pixels
            black_mask = (
                (img_array[:, :, 0] == 0)
                & (img_array[:, :, 1] == 0)
                & (img_array[:, :, 2] == 0)
            )

            # Create a mask for white pixels
            white_mask = (
                (img_array[:, :, 0] == 255)
                & (img_array[:, :, 1] == 255)
                & (img_array[:, :, 2] == 255)
            )

            # Replace black pixels with random color1
            img.paste(color1, None, mask=Image.fromarray(black_mask))

            # Replace white pixels with random color2
            img.paste(color2, None, mask=Image.fromarray(white_mask))

            # Convert the NumPy array back to a PIL image
            img = Image.fromarray(np.uint8(img))

        return img


class GrayToRandColor:
    """
    Transform to interpolate colors based on grayscale intensity in an RGB image.

    Methods:
        __call__(self, img: Image.Image) -> Image.Image: Apply the transformation to the input image.
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply the transformation to the input image.

        Parameters:
            img (PIL.Image.Image): Input grayscale image in "RGB" mode.

        Returns:
            PIL.Image.Image: Transformed image with colors interpolated based on grayscale intensity.
        """
        if img.mode == "RGB":
            # Generate random colors for smooth fade
            color1 = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            color2 = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )

            # Convert image to NumPy array
            img_array = np.array(img)

            # Calculate grayscale intensity
            gray_intensity = np.mean(img_array, axis=-1)

            # Normalize the intensity to range [0, 1]
            normalized_intensity = gray_intensity / 255.0

            # Interpolate between color1 and color2 based on intensity
            interpolated_colors = (
                (1 - normalized_intensity) * color1[0]
                + normalized_intensity * color2[0],
                (1 - normalized_intensity) * color1[1]
                + normalized_intensity * color2[1],
                (1 - normalized_intensity) * color1[2]
                + normalized_intensity * color2[2],
            )

            # Replace pixels with interpolated colors
            img_array[:, :, 0] = interpolated_colors[0]
            img_array[:, :, 1] = interpolated_colors[1]
            img_array[:, :, 2] = interpolated_colors[2]

            # Convert the NumPy array back to a PIL image
            img = Image.fromarray(np.uint8(img_array))

        return img


class ColorJitter(torch.nn.Module):
    """Change the brightness, contrast, saturation and hue of an image.
    If the image is torch Tensor, it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, mode "1", "I", "F" and modes with transparency (alpha channel) are not supported.

    Args:
        brightness float: How much to jitter brightness.
            brightness_factor should be non negative numbers.
        contrast float: How much to jitter contrast.
            contrast_factor should be non-negative numbers.
        saturation float: How much to jitter saturation.
            saturation_factor should be non negative numbers.
        hue float: How much to jitter hue.
            hue_factor should have 0<= hue <= 0.5.
            To jitter hue, the pixel values of the input image has to be non-negative for conversion to HSV space;
            thus it does not work if you normalize your image to an interval with negative values,
            or use an interpolation that generates negative values before using this function.
    """

    def __init__(
        self,
        brightness: float | None = None,
        contrast: float | None = None,
        saturation: float | None = None,
        hue: float | None = None,
    ) -> None:
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def forward(self, img):
        if self.brightness is not None:
            img = F.adjust_brightness(img, self.brightness)
        if self.contrast is not None:
            img = F.adjust_contrast(img, self.contrast)
        if self.saturation is not None:
            img = F.adjust_saturation(img, self.saturation)
        if self.hue is not None:
            img = F.adjust_hue(img, self.hue)

        return img


class Gray2BW(torch.nn.Module):
    """Converts a gray scale image into a black and white image.
    If the image is torch Tensor, it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, mode "1", "I", "F" and modes with transparency (alpha channel) are not supported.

    Args:
        alpha float: How much to shift towards BW.
            alpha values should be non negative numbers between [0, 1].
    """

    def __init__(
        self,
        alpha: float,
    ) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, img):
        img = F.adjust_saturation(img, 0.0)
        t = F.pil_to_tensor(img) / 255
        t[t < 0.5] -= self.alpha * 0.5
        t[t >= 0.5] += self.alpha * 0.5
        t = torch.clip(t, 0, 1)
        img = F.to_pil_image(t)
        return img


class MotionBlur(torch.nn.Module):
    """Add a motion blur effect to an iamge.
    If the image is torch Tensor, it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, mode "1", "I", "F" and modes with transparency (alpha channel) are not supported.

    Args:
        radius float: Size of the bluring kernel.
            radius values should be non negative numbers.
        sigma float: Intensity of the bluring operation.
            sigma values should be non negative numbers.
        angle float: Size of the bluring kernel.
            angle values should be numbers between [0., 360.].
    """

    def __init__(
        self, radius: float = 0.0, sigma: float = 0.0, angle: float | None = None
    ):
        super().__init__()
        self.radius = radius
        self.sigma = sigma
        self.angle = angle

    def _get_angle(self):
        if self.angle is None:
            return np.random.randint(0, 360)

        return self.angle

    def forward(self, img):
        bio = BytesIO()
        img.save(bio, format="png")
        wim = WImage(blob=bio.getvalue())

        angle = self._get_angle()
        wim.motion_blur(radius=self.radius, sigma=self.sigma, angle=angle)

        return Image.frombytes("RGB", wim.size, np.array(wim))


class GlassBlur(torch.nn.Module):
    """Add a frozen lense effect to an image.
    If the image is torch Tensor, it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, mode "1", "I", "F" and modes with transparency (alpha channel) are not supported.

    Args:
        radius float: Size of the bluring kernel.
            radius values should be non negative numbers.
        sigma float: Intensity of the bluring operation.
            sigma values should be non negative numbers.
        angle float: Size of the bluring kernel.
            angle values should be numbers between [0., 180.].
    """

    def __init__(
        self,
        radii: Iterable[float] = (0.0, 0.0),
        max_displacement: int = 0,
        ksize: int = 1,
        sigma: float = 1.0,
        iterations: int = 1,
    ):
        super().__init__()
        self.max_displacement = max_displacement
        self.iterations = iterations
        self.sradii = [r**2 for r in radii]
        self.sclean = [max(r - ksize, 0) ** 2 for r in radii]
        self.ksize = ksize
        self.sigma = sigma

    def _get_centers(self, w, h):
        cx = w // 2
        cy = h // 2

        dx, dy = np.zeros(len(self.sradii)), np.zeros(len(self.sradii))
        if self.max_displacement > 0:
            dx, dy = np.random.randint(
                -self.max_displacement,
                self.max_displacement,
                size=(2, len(self.sradii)),
            )

        return cx + dx, cy + dy

    def _is_close(self, dists, radii):
        for d, r in zip(dists, radii):
            if d < r:
                return True
        return False

    def forward(self, img):
        array = np.array(img)
        width, height, _ = array.shape

        cx, cy = self._get_centers(width, height)

        r = self.ksize
        array_orig = array.copy()
        for _ in range(self.iterations):
            for h in range(r, array.shape[0] - r + 1):
                for w in range(r, array.shape[1] - r + 1):
                    dists = (cx - w) ** 2 + (cy - h) ** 2
                    if not self._is_close(dists, self.sradii):
                        dx, dy = np.random.randint(-r, r, size=2)
                        h_ = h + dx
                        w_ = w + dy

                        array[h, w], array[h_, w_] = (
                            array_orig[h_, w_],
                            array_orig[h, w],
                        )

        array = (
            np.clip(gaussian(array / 255.0, sigma=self.sigma, channel_axis=-1), 0, 1)
            * 255
        )

        for h in range(r, array.shape[0] - r + 1):
            for w in range(r, array.shape[1] - r + 1):
                dists = (cx - w) ** 2 + (cy - h) ** 2

                if self._is_close(dists, self.sclean):
                    array[h, w] = array_orig[h, w]

        return Image.fromarray(np.uint8(array))


class AddFrost(torch.nn.Module):
    """
    Add a frozt effect to an image by overlaying with an icey image.
    If the image is torch Tensor, it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, mode "1", "I", "F" and modes with transparency (alpha channel) are not supported.

    Args:
        img_factor float: intensity factor for the original image.
            img_factor values should be non negative numbers between [0.0, 1.0].
        frost_factor float: intensity factor for the frost image.
            frost_factor values should be non negative numbers between [0.0, 1.0].
        frost_img_folder Path | str: directory path to frost images.
            directory containing several frost images.
    """

    def __init__(
        self,
        img_factor: float = 0.7,
        frost_factor: float = 0.3,
        frost_img_folder: Path | str = Path("./data/frost"),
    ):
        super().__init__()
        self.img_factor = img_factor
        self.frost_factor = frost_factor
        self.frost_images = list(Path(frost_img_folder).glob("*.*"))

    def forward(self, img):
        array = np.array(img)
        w, h, _ = array.shape
        frost_file = np.random.choice(self.frost_images, size=1)[0]

        scale = 0.2
        with Image.open(frost_file) as frost:
            size = (int(scale * frost.size[0]), int(scale * frost.size[1]))
            frost = frost.convert("RGB")
            frost = frost.resize(size)
            frost = np.array(frost)

        x_start = np.random.randint(0, frost.shape[0] - w)
        y_start = np.random.randint(0, frost.shape[1] - h)

        x_end = x_start + w
        y_end = y_start + h

        frost = frost[x_start:x_end, y_start:y_end]

        array = self.img_factor * array + self.frost_factor * frost
        array = np.clip(array, 0, 255)
        return Image.fromarray(np.uint8(array))
