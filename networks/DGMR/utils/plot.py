import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats


def plot_rain_field(pred, target, pred_title, true_title, output_path=None):
    if isinstance(pred, torch.Tensor):
        pred = pred.detach()
    if isinstance(target, torch.Tensor):
        target = target.detach()

    arrs = torch.stack((target, pred), dim=0).cpu().numpy()
    title = [true_title, pred_title]
    batch = [1, 0]
    S = arrs.shape[1]
    fig = plt.figure(figsize=(3*S, 15))

    for b in batch:
        arr = arrs[b] # S, H, W
        for s in range(S):
            img = arr[s, :, :]

            ax = fig.add_subplot(b+1, S, s+1)
            im = ax.imshow(img, vmin=arrs.min(), vmax=arrs.max(), cmap="jet")
            ax.set_title(title[b][s])
            ax.set_xticklabels(labels=[])
            ax.set_yticklabels(labels=[])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(im, cax=cax, orientation="vertical")

    if output_path: plt.savefig(output_path)
    else: return fig
    return

def plot_metric_boxplot(metric_time, ylabel, title=None, showfliers=True):
    fontsize=16
    if isinstance(metric_time, torch.Tensor):
        metric_time = metric_time[~torch.any(metric_time.isnan(), dim=1)]
        metric_time = metric_time.cpu().numpy() # (N, time_steps)
    fig = plt.figure(figsize=(8, 6))
    plt.boxplot(metric_time, showfliers=showfliers)
    plt.xticks([i+1 for i in range(metric_time.shape[-1])], [5*(t+1) for t in range(metric_time.shape[-1])], fontsize=12)
    plt.xlabel("Prediction lead time (min)", fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    if title is not None:
        plt.title(title, fontsize=fontsize)
    return fig

def plot_metric_mean(metric_time, ylabel, title=None):
    fontsize=16
    if isinstance(metric_time, torch.Tensor):
        metric_time = metric_time[~torch.any(metric_time.isnan(), dim=1)]
        metric_time = metric_time.cpu().numpy() # (N, time_steps)
    fig = plt.figure(figsize=(8, 6))
    plt.plot([t+1 for t in range(metric_time.shape[-1])], metric_time.mean(axis=0), marker="o")
    plt.xticks([i+1 for i in range(metric_time.shape[-1])], [5*(t+1) for t in range(metric_time.shape[-1])], fontsize=12)
    plt.xlabel("Prediction lead time (min)", fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    if title is not None:
        plt.title(title, fontsize=fontsize)
    return fig

def plot_psd():
    return

if __name__ == "__main__":
    from pysteps.utils.spectral import rapsd
    from pysteps.visualization.spectral import plot_spectrum1d
    # 78021:78031
    rain = np.memmap("/home/wangup/Documents/Nimrod/Nimrod_2019.dat", shape=(288*365, 512, 512), dtype=np.int16)[78021:78031] / 32
    
    total_psd, total_freq = [], []
    for x in rain:
        psd, freq = rapsd(x, return_freq=True)
        print(freq.sum())
        total_psd += psd.tolist()
        total_freq += (np.log(freq)).tolist()
    plt.scatter(total_freq, total_psd)
    plt.savefig("psd.png")
    #fig = plot_spectrum1d(total_freq, total_psd, "km", "dBZ")
    pass