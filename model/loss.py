import torch.nn.functional as F
import torch.nn as nn

def nll_loss(output, target):
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

class XMILoss(nn.Module):
    """
    Mutual Information loss function
    """
    def __init__(self, w_entropy: float = 1.0, w_mi: float = 0.2):
        super(XMILoss, self).__init__()
        self.w_entropy = w_entropy
        self.w_mi = w_mi

    def forward(self, output, target, features):
        """
        Number of input parameters done for consistency.

        output. pair of (hotmap_predictions, features)
        target. target hotmap_gt
        """
        loss_nll = nll_loss(output, target)
        loss_kl  = kl_loss(features)

        loss = self.w_entropy * loss_nll + self.w_mi * loss_kl

        info_dict = {
            "nll_loss": loss_nll,
            "kl_loss": loss_kl,
            # "loss": loss
        }

        return loss, info_dict
