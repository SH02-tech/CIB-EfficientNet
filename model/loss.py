import torch
import torch.nn.functional as F
import torch.nn as nn

class NLLLoss(nn.Module):
    """
    Negative Log Likelihood loss function
    """
    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, output, target):
        """
        Number of input parameters done for consistency.
        """
        return F.nll_loss(output, target)

def kl_loss(features):
    loss = 0
    num_channels = int(features.shape[1])
    log_features = F.log_softmax(features)

    for i in range(num_channels):
        for j in range(num_channels):
            source_channel = log_features[:,i]
            target_channel = log_features[:,j]

            loss = loss + F.kl_div(source_channel, target_channel, log_target = True)

    return loss

def cov_loss(features):
    batch_size = features.shape[0]
    mean_features = torch.mean(features, dim=0, keepdim=True)
    center_features = features - mean_features

    cov = torch.matmul(center_features.T, center_features) / batch_size

    id_matrix = torch.eye(cov.size(0), device = cov.device)

    decorr_loss = torch.norm(cov - id_matrix, p=2) ** 2

    return decorr_loss

def l1_loss(features):
    return torch.norm(features, p=1)

def l2_loss(features):
    return torch.norm(features, p=2)

# def ortho_loss(weights):
#     loss = 0
#     num_channels = int(weights.shape[1])

#     for i in range(num_channels):
#         for j in range(i+1, num_channels):
#             first_weights = weights[:,i]
#             second_weights = weights[:,j]

#             loss = loss + F.cosine_similarity(first_weights, second_weights, dim = 0)

#     return loss

def ortho_loss(weights, type="row"):
    if  len(weights.shape) > 2: # weights from CNN layer
        weights = weights.view(weights.size(0), -1)

    cov = torch.matmul(weights, weights.T) if type == "row" else torch.matmul(weights.T, weights)
    id_matrix = torch.eye(cov.size(0), device=cov.device)
    loss = torch.norm(cov - id_matrix, p=2) ** 2

    return loss

class XMILoss(nn.Module):
    """
    Mutual Information loss function
    """
    def __init__(self, w_entropy: float = 1.0, w_mi: float = 0.2, w_cov: float = 0.2, w_ortho: float = 0.1, w_l1: float = 0.1, w_l2: float = 0.1):
        super(XMILoss, self).__init__()
        self.w_entropy = w_entropy
        self.w_mi = w_mi
        self.w_cov = w_cov
        self.w_ortho = w_ortho
        self.w_l1 = w_l1
        self.w_l2 = w_l2

    def forward(self, output, target, weights, features):
        loss_nll = F.nll_loss(output, target)
        loss_kl  = kl_loss(features)
        loss_cov = cov_loss(features)
        loss_ortho = ortho_loss(weights,  type="row")
        loss_l1 = l1_loss(weights)
        loss_l2 = l2_loss(weights)

        loss = self.w_entropy * loss_nll + self.w_mi * loss_kl + \
            self.w_cov * loss_cov + self.w_ortho * loss_ortho + \
            self.w_l1 * loss_l1 + self.w_l2 * loss_l2

        info_dict = {
            "nll_loss": loss_nll,
            "kl_loss": loss_kl,
            "cov_loss": loss_cov,
            "ortho_loss": loss_ortho,
            "l1_loss": loss_l1,
            "l2_loss": loss_l2
        }

        return loss, info_dict
