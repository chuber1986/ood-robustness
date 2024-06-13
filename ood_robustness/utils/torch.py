from functools import reduce
from typing import Callable
import torch


def get_module_names(module: torch.nn.Module) -> list[str]:
    return [name for name, _ in module.named_modules()]


def get_module_by_name(module: torch.nn.Module, access_string: str) -> torch.nn.Module:
    names = access_string.split(sep=".")
    if not names or len(names) == 1 and names[0] == "":
        return module
    return reduce(getattr, names, module)


def get_activation(name: str, out: dict) -> Callable:

    def hook(model, input_, output):
        _ = model, input_
        store = out.get(name, list())
        store.append(output.detach())
        out[name] = store

    return hook


def add_hook(
    module: torch.nn.Module, hook: Callable
) -> torch.utils.hooks.RemovableHandle:
    return module.register_forward_hook(hook)


def rm_hook(handle: torch.utils.hooks.RemovableHandle):
    handle.remove()
