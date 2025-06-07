import torch
import copy

from torchvision.models.efficientnet import MBConv
from torchvision.ops.misc import SqueezeExcitation

from zennit.composites import EpsilonPlusFlat
from zennit import canonizers as canonizers
from zennit import layer as zlayer
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

# Canonizer for EfficientNet obtained from:
# Optimizing Explanations by Network Canonization and Hyperparameter Search
# CVPR2023 Workshop
# Pahde et al.  

class SignalOnlyGate(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x1,x2):
        return x1*x2 

    @staticmethod
    def backward(ctx,grad_output):
        return torch.zeros_like(grad_output), grad_output


class SECanonizer(canonizers.AttributeCanonizer):
    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        if isinstance(module, SqueezeExcitation):
            attributes = {
                'forward': cls.forward.__get__(module),
                'fn_gate': SignalOnlyGate()
            }
            return attributes
        return None

    @staticmethod
    def forward(self, input):
        scale = self._scale(input)
        return self.fn_gate.apply(scale, input)

class MBConvCanonizer(canonizers.AttributeCanonizer):
    '''Canonizer specifically for MBConvBlock of Mobile Net v2 type models.'''

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        if isinstance(module, MBConv):
            attributes = {
                'forward': cls.forward.__get__(module),
                'canonizer_sum': zlayer.Sum()
            }
            return attributes
        return None

    @staticmethod
    def forward(self, input):
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)

            # result += input
            result = torch.stack([input, result], dim=-1)
            result = self.canonizer_sum(result)
        return result

class EfficientNetCanonizer(canonizers.CompositeCanonizer):
    def __init__(self):
        super().__init__((
            SECanonizer(),
            MBConvCanonizer(),
        ))

class EfficientNetBNCanonizer(canonizers.CompositeCanonizer):
    def __init__(self):
        super().__init__((
            SECanonizer(),
            MBConvCanonizer(),
            canonizers.SequentialMergeBatchNorm()
        ))

class ZennitHandler:
    def __init__(self, model_xmi):
        # deep copy
        model = copy.deepcopy(model_xmi)
        
        # change requires_grad to True for all parameters
        for param in model.parameters():
            param.requires_grad = True

        model.eval()
        self.composite = EpsilonPlusFlat([EfficientNetBNCanonizer()])
        self.layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])
        self.attribution = CondAttribution(model, no_param_grad=True)
        self.cc = ChannelConcept()
        

    def get_heatmap_top(self, input_tensor, target_layer, label, k=5):
        input_zennit = input_tensor.clone().detach()
        input_zennit = input_zennit.unsqueeze(0)
        input_zennit.requires_grad = True

        conditions = [{'y': [label]}]

        attr = self.attribution(input_zennit, conditions, self.composite, record_layer=self.layer_names)
        rel_c = self.cc.attribute(attr.relevances[target_layer], abs_norm=True)
        _, concept_ids = torch.topk(rel_c[0], k)

        conditions = [{target_layer: [id], 'y': [label]} for id in concept_ids]

        heatmap, _, _, _ = self.attribution(input_zennit, conditions, self.composite)

        return heatmap