import torch
import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.stats import wasserstein_distance

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def euclidean_distance(output, target):
    with torch.no_grad():
        assert output.shape == target.shape
        return torch.norm(output - target, p=2).item()

def _heatmaps_pair_apply_fn(heatmaps, fn, flatten:bool = False):
    """
    Applies a function to all pairs of heatmaps.
    Args:
        heatmaps (torch.Tensor): Tensor of shape (n, H, W).
        fn (callable): Function to apply to each pair of heatmaps.
        flatten (bool): If True, flattens each heatmap to a 1D vector before applying the function.
    Returns:
        float: Mean value from applying the function to all pairs.
    """
    with torch.no_grad():
        n = heatmaps.shape[0]

        if flatten:
            heatmaps = heatmaps.reshape(n, -1)
        
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = fn(heatmaps[i], heatmaps[j])
                dists.append(dist)
        return sum(dists) / len(dists) if dists else 0.0


def euclidean_distance_hm(heatmaps, k:int = 2):
    """
    Computes mean Euclidean distance between all pairs of the first k heatmaps.
    Args:
        heatmaps (torch.Tensor): Tensor of shape (n, H, W) or (k, H, W).
        k (int): Number of heatmaps to consider from the start.
    Returns:
        float: Mean Euclidean distance between all pairs.
    """
    metric = _heatmaps_pair_apply_fn(heatmaps[:k], euclidean_distance, flatten=True)
    return metric

def kl_divergence(output, target, eps:float = 1e-10):
    with torch.no_grad():
        assert output.shape == target.shape
        output = output / (output.sum(dim=0, keepdim=True) + eps)  # Normalize output
        target = target / (target.sum(dim=0, keepdim=True) + eps)  # Normalize target

        kl_div = torch.nn.functional.kl_div(torch.log(output + eps), target + eps, reduction='mean')
        return kl_div.item()

def kl_divergence_hm(heatmaps, k:int = 2):
    """
    Computes mean KL divergence between all pairs of the first k heatmaps.
    Args:
        heatmaps (torch.Tensor): Tensor of shape (n, H, W) or (k, H, W).
        k (int): Number of heatmaps to consider from the start.
    Returns:
        float: Mean KL divergence between all pairs.
    """
    metric = _heatmaps_pair_apply_fn(heatmaps[:k], kl_divergence)
    return metric

def mi_hm(heatmaps, k:int = 2):
    """
    Computes mean mutual information between all pairs of the first k heatmaps.
    Args:
        heatmaps (torch.Tensor): Tensor of shape (n, H, W) or (k, H, W).
        k (int): Number of heatmaps to consider from the start.
    Returns:
        float: Mean mutual information between all pairs.
    """
    heatmaps_k = heatmaps[:k]

    # to numpy
    if isinstance(heatmaps_k, torch.Tensor):
        heatmaps_k = heatmaps_k.cpu().numpy()

    metric = _heatmaps_pair_apply_fn(heatmaps_k, mutual_info_score) 
    return metric

def wasserstein_hm(heatmaps, k:int = 2):
    """
    Computes mean Wasserstein distance between all pairs of the first k heatmaps.
    Args:
        heatmaps (torch.Tensor): Tensor of shape (n, H, W) or (k, H, W).
        k (int): Number of heatmaps to consider from the start.
    Returns:
        float: Mean Wasserstein distance between all pairs.
    """

    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.cpu().numpy()

    heatmaps = heatmaps / (heatmaps.sum(axis=(1, 2), keepdims=True) + np.finfo(float).eps)  # Normalize heatmaps

    metric = _heatmaps_pair_apply_fn(heatmaps[:k], wasserstein_distance, flatten=True)
    return  metric