import torch


def inv_pixel(x: torch.Tensor) -> torch.Tensor:
    if torch.is_floating_point(x):
        return 1.0 - x
    return 255 - x


def change_pixel_rgb(
    x: torch.Tensor,
    r: int | float | None = None,
    g: int | float | None = None,
    b: int | float | None = None,
) -> torch.Tensor:
    if torch.is_floating_point(x):
        assert not isinstance(r, int)
        assert not isinstance(g, int)
        assert not isinstance(b, int)

    xnew = x.clone()
    if r is not None:
        xnew[..., 0] = r
    if g is not None:
        xnew[..., 1] = g
    if b is not None:
        xnew[..., 2] = b

    return xnew
