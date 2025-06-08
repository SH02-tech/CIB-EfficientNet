import torch


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


def euclidean_distance_hm(heatmaps, k:int = 2):
    """
    Computes mean Euclidean distance between all pairs of the first k heatmaps.
    Args:
        heatmaps (torch.Tensor): Tensor of shape (n, H, W) or (k, H, W).
        k (int): Number of heatmaps to consider from the start.
    Returns:
        float: Mean Euclidean distance between all pairs.
    """
    with torch.no_grad():
        heatmaps = heatmaps[:k].reshape(k, -1)  # flatten each heatmap
        dists = []
        for i in range(k):
            for j in range(i + 1, k):
                dist = euclidean_distance(heatmaps[i], heatmaps[j])
                dists.append(dist)
        return sum(dists) / len(dists) if dists else 0.0


def kl_divergence(output, target):
    with torch.no_grad():
        assert output.shape == target.shape
        return torch.nn.functional.kl_div(
            torch.log_softmax(output, dim = 0),
            torch.softmax(target, dim = 0),
            reduction='batchmean'
        ).item()


def kl_divergence_hm(heatmaps, k:int = 2):
    """
    Computes mean KL divergence between all pairs of the first k heatmaps.
    Args:
        heatmaps (torch.Tensor): Tensor of shape (n, H, W) or (k, H, W).
        k (int): Number of heatmaps to consider from the start.
    Returns:
        float: Mean KL divergence between all pairs.
    """
    with torch.no_grad():
        heatmaps = heatmaps[:k].reshape(k, -1)  # flatten each heatmap
        dists = []
        for i in range(k):
            for j in range(k):
                if i != j:
                    dist = kl_divergence(heatmaps[i], heatmaps[j])
                    dists.append(dist)
        return sum(dists) / len(dists) if dists else 0.0
