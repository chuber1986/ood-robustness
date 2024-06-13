"""Plotting functionality."""

from typing import Union

import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from hyperpyper.utils import HistogramPlotter, MultiFigurePlotter, MultiImagePlotter

from ood_robustness.utils.metrics import FPR95OODMetric


def get_figsize(figwidth=3.48761, aspect_ratio=6 / 8, scale=1.0):
    # textwidth = 7.1413
    # colwidth = 3.48761

    width = figwidth * scale
    height = width * aspect_ratio

    return width, height


def plot_transformed_samples(output, title, feat_name, indices, figsize=get_figsize()):
    files = [output["file"][i] for i in indices]
    energies = [output["item"][i] for i in indices]
    features = output[feat_name]

    titles = [f"{f}\nEnergy: {float(e):.3f}" for f, e in zip(files, energies)]

    mip = MultiImagePlotter(
        images=features,
        titles=titles,
        suptitle=title,
        figsize=figsize,
    )
    return mip.plot()


def _to_image(t):
    if isinstance(t, Image.Image):
        return t

    if t.ndim == 3:
        tt = t.moveaxis(0, -1)
    else:
        return t

    if tt.dtype == torch.float:
        it = np.uint8(tt.cpu().numpy() * 255)
    else:
        it = np.uint8(tt.cpu().numpy())

    return Image.fromarray(it)


def plot_transformed_samples_comparison(
    output, title, feat_name, indices, rotate=False, figsize=None
):
    files = None
    energies = None
    features = []
    titles = []

    for exp, res in output.items():
        if files is None or energies is None:
            files = [res["file"][i] for i in indices]
            energies = [res["item"][i] for i in indices]

        titles += [
            # f"{exp}\n{f}\nEnergy: {float(e):.3f}" for f, e in zip(files, energies)
            f"{exp}"
            for f, e in zip(files, energies)
        ]

        feat = res[feat_name]
        if len(feat) > len(indices):
            feat = [feat[i] for i in indices]

        features += feat

    features = [_to_image(f) for f in features]
    layout = (len(output), len(indices))

    if rotate:
        features = features[::2] + features[1::2]
        titles = titles[::2] + titles[1::2]

    fsize = figsize or get_figsize(
        figwidth=7.1413,
        aspect_ratio=len(indices) / (len(output) - 0.5),
        scale=1.0,
    )
    fsize = tuple(f * len(indices) for f in fsize)

    mip = MultiImagePlotter(
        images=features,
        titles=titles,
        suptitle=title,
        layout=layout,
        rotate=rotate,
        figsize=fsize,
    )

    fig = mip.plot()
    fig.tight_layout()
    return fig


from torcheval.metrics.functional import binary_precision_recall_curve


def _min_max_norm(v, vmin, vmax):
    return (v - vmin) / (vmax - vmin)


def _get_normalized_errors(eng, ref, vmin, vmax):
    e = _min_max_norm(eng, vmin, vmax)
    r = _min_max_norm(ref, vmin, vmax)
    return (e - r).abs().max(), ((e - r) ** 2).mean()


def compute_metrics(model, output_energy, reference=None):
    energy = output_energy.ravel()
    idd_energy = model.get_idd_reference_energy().ravel()
    ood_energy = model.get_ood_reference_energy().ravel()

    if reference is None:
        ref = energy
    else:
        ref = reference.ravel()

    labels = torch.concatenate(
        [torch.ones(len(idd_energy)), torch.zeros(len(ood_energy))], axis=0
    )
    score = torch.concatenate([idd_energy, ood_energy], axis=0)
    max_ae, mse = _get_normalized_errors(
        energy, ref, idd_energy.median(), ood_energy.median()
    )

    _, recalls, ths = binary_precision_recall_curve(score, labels)
    th95 = ths[torch.argmin(torch.abs(recalls - 0.95))]
    th99 = ths[torch.argmin(torch.abs(recalls - 0.99))]

    fp95 = (energy > th95).sum()
    tn95 = (energy <= th95).sum()
    fpr95 = fp95 / (fp95 + tn95)

    fp99 = (energy > th99).sum()
    tn99 = (energy <= th99).sum()
    fpr99 = fp99 / (fp99 + tn99)

    labels = np.concatenate([np.zeros(len(idd_energy)), np.ones(len(energy))], axis=0)
    score = np.concatenate([idd_energy, energy], axis=0)
    fpr95_metric = FPR95OODMetric()

    max_wd = wasserstein_distance(idd_energy, ood_energy)
    wd = wasserstein_distance(idd_energy, energy)
    normed_wd = wd / max_wd

    metrics = {
        "AUROC": roc_auc_score(labels, score),
        "FPR95": fpr95_metric(-idd_energy, -energy),
        "MSE": mse,
        "maxAE": max_ae,
        "Wasserstein distance": wd,
        "Max. Wasserstein distance": max_wd,
        "Norm. Wasserstein distance": normed_wd,
        # Additional
        "ROC": roc_curve(labels, score, pos_label=1),
        "PR": precision_recall_curve(labels, score, pos_label=1),
        "ID_FPR95": 1 - fpr95_metric(-idd_energy, -energy),
        "ID_FPR95_": fpr95,
        "ID_FPR99_": fpr99,
        "ID_Threshold95": th95,
        "ID_Threshold99": th99,
    }

    return metrics


def print_metrics(model, output):
    tbl = Table("OOD Metrics")

    metric_names = [
        "AUROC",
        "FPR95",
        "Wasserstein distance",
        "Max. Wasserstein distance",
        "Norm. Wasserstein distance",
        # "ID_FPR95",
        # "ID_FPR95_",
        # "ID_FPR99_",
        # "ID_Threshold95",
        # "ID_Threshold99",
    ]

    reference = None
    if "Raw" in output:
        metric_names.append("MSE")
        reference = output["Raw"]["item"]

    for m in metric_names:
        tbl.add_column(m, justify="center", style="green")

    metrics = {}
    for exp_name, results in output.items():
        if exp_name.startswith("res_"):
            continue
        mets = compute_metrics(model, results["item"], reference)
        tbl.add_row(exp_name, *[f"{mets[k]:.3e}" for k in metric_names])
        metrics[exp_name] = mets

    console = Console()
    console.print(tbl)
    return metrics


def _to_string(val):
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def plot_energy_histograms(model, energy, label, n_samples, figsize=get_figsize()):
    x = [
        model.get_idd_reference_energy(max_samples=n_samples),
        model.get_ood_reference_energy(max_samples=n_samples),
        energy.ravel(),
    ]
    lbl = [
        "Energy CIFAR",
        "Energy ImageNet",
        _to_string(label),
    ]
    fig = HistogramPlotter(x, lbl, figsize=figsize).plot()
    fig.tight_layout()
    return fig


def plot_experiemnt_energies(model, output, n_samples, figsize=get_figsize()):
    figs = []
    for exp, res in output.items():
        if exp.startswith("res_"):
            continue
        fig = plot_energy_histograms(
            model, res["item"], exp, n_samples, figsize=figsize
        )
        figs.append(fig)

    fsize = tuple(float(f * np.ceil(np.sqrt(len(figs)))) for f in figsize)
    fig = MultiFigurePlotter(figs, suptitle="Energy histograms", figsize=fsize).plot()
    for f in figs:
        del f

    fig.tight_layout()
    return fig


def _repr(v):
    try:
        return f"{v.squeeze():.3f}"
    except AttributeError:
        pass

    return str(v)


def print_sample_stats(stats: list[dict]):
    console = Console()
    tbl = Table()

    for k in stats[0]:
        tbl.add_column(k)

    for sample in stats:
        tbl.add_row(*[_repr(s) for s in sample.values()])

    console.print(tbl)


def add_colorbar(fig, vmin, vmax, label="", ontop=True):
    y = 0.125 if ontop else -0.125
    ax = fig.add_axes([0, y, 1, 0.5])
    ax.set_axis_off()
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap="viridis"
        ),
        ax=ax,
        label=label,
        orientation="horizontal",
        location="top" if ontop else "bottom",
    )
    cbar.ax.tick_params(axis="both", labelsize=20)
    cbar.ax.xaxis.label.set_size(20)


def add_vcolorbar(fig, vmin, vmax, label="", right=True):
    x = 0.15 if right else -0.15
    ax = fig.add_axes([x, 0.05, 1, 0.8])
    ax.set_axis_off()
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap="viridis"
        ),
        ax=ax,
        label=label,
        orientation="vertical",
        location="right" if right else "left",
    )
    cbar.ax.tick_params(axis="both", labelsize=20)
    cbar.ax.xaxis.label.set_size(20)


def plot_statistics(stats: dict):
    supervecs = stats["grouped_hm"]
    supervecs = {k: s.reshape(-1) for k, s in supervecs.items()}
    labels = list(supervecs.keys())
    hist = HistogramPlotter(
        list(supervecs.values()), labels, bins="doane", density=True
    ).plot()

    std_imgs = [stats["avg_std"]] + list(stats["grouped_std"].values())
    impact_imgs = [stats["avg_impact"]] + list(stats["grouped_impact"].values())

    vmin_std = min(float(torch.min(img)) for img in std_imgs)
    vmax_std = max(float(torch.max(img)) for img in std_imgs)

    vmin_impact = min(float(torch.min(img)) for img in impact_imgs)
    vmax_impact = max(float(torch.max(img)) for img in impact_imgs)

    imgs = std_imgs + impact_imgs

    lbls = (
        ["Average Standard Deviation"]
        + [f"Avg. Standard Deviation: {lbl}" for lbl in labels]
        + ["Average Impact"]
        + [f"Avg. Impact: {lbl}" for lbl in labels]
    )
    fig = MultiImagePlotter(
        imgs, lbls, layout=(2, int(np.ceil(len(imgs)) // 2)), figsize=(20, 8)
    ).plot()

    axes = fig.get_axes()
    nmaps = len(std_imgs)
    for ax in axes[:nmaps]:
        imgs = ax.get_images()
        if imgs:
            imgs[0].set_clim(vmin=vmin_std, vmax=vmax_std)

    for ax in axes[nmaps:]:
        imgs = ax.get_images()
        if imgs:
            imgs[0].set_clim(vmin=vmin_impact, vmax=vmax_impact)

    add_colorbar(fig, vmin_std, vmax_std, "Standard Deviation", ontop=True)
    add_colorbar(fig, vmin_impact, vmax_impact, "Impact", ontop=False)

    return hist, fig


def plot_pie_chart(
    data: Union[list, np.ndarray],
    custom_cmap: Union[str, Colormap] = None,
    startangle=0,
    counterclock: "bool" = True,
    label:str="Energy"
) -> None:
    """
    Plots a pie chart with segments colored according to the values and adds a colorbar.

    Args:
        data (list or numpy.ndarray): 1D array or list of values.
        custom_cmap (str or matplotlib.colors.Colormap, optional): Custom colormap to use for coloring segments.
            If None, a default colormap will be used. Default is None.
        startangle (int, optional): Starting angle for the pie chart. Default is 0.

    Returns:
        None
    """
    # Convert data to NumPy array if it's a list
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    # Normalize the data to the range [0, 1]
    norm = Normalize(vmin=np.min(data), vmax=np.max(data))
    if custom_cmap:
        colormap = plt.cm.get_cmap(custom_cmap)
    else:
        colormap = plt.cm.plasma

    colors_mapped = colormap(norm(data))

    # Create equal values for equally sized segments
    n_segments = len(data)
    equal_values = np.ones(n_segments)

    fig, ax = plt.subplots()
    ax.pie(
        equal_values,
        colors=colors_mapped,
        startangle=startangle,
        counterclock=counterclock,
    )

    # Create a ScalarMappable object for the colorbar
    sm = ScalarMappable(norm=norm, cmap=colormap)
    sm.set_array([])  # No data is needed for the scalar mappable

    # Add the colorbar to the figure
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label(label)

    return fig
