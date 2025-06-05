import torch

from zennit.composites import EpsilonPlusFlat
from zennit.canonizers import SequentialMergeBatchNorm
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names

def denormalize(tensor, mean, std):
    """
    Denormalizes a tensor image.
    Args:
        tensor (torch.Tensor): Normalized image tensor (C, H, W).
        mean (list or tuple): Mean values for each channel.
        std (list or tuple): Standard deviation values for each channel.
    Returns:
        torch.Tensor: Denormalized image tensor.
    """
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean).view(tensor.shape[0], 1, 1)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std).view(tensor.shape[0], 1, 1)

    return tensor * std + mean

class ZennitHandler:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.composite = EpsilonPlusFlat([SequentialMergeBatchNorm()])
        self.attribution = CondAttribution(model, no_param_grad=True)
        self.layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])
        self.cc = ChannelConcept()

    def get_topk_relevances(self, input_tensor, zennit_list, layer_name, k=5):
        """
        Get the top-k relevances for a given layer in the model.
        Args:
            input_tensor (torch.Tensor): Input tensor for the model.
            zennit_list (list): List of Zennit dictionaries containing relevant information for layer, as specified in CRP.
            layer_name (str): Name of the layer to analyze.
            k (int): Number of top relevances to return.
        Returns:
            torch.Tensor: Top-k relevances and their corresponding concept IDs.
        """
        input_zennit = input_tensor.clone().detach()
        input_zennit = input_zennit.unsqueeze(0)
        input_zennit.requires_grad = True

        attr = self.attribution(input_zennit, zennit_list, self.composite, record_layer=self.layer_names)
        rel_c = self.cc.attribute(attr.relevances[layer_name], abs_norm=True)

        rel_values, concept_ids = torch.topk(rel_c[0], k=k)
        return rel_values, concept_ids, attr
