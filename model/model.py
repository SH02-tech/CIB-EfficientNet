import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b2 as EfficientNetB2
from torchvision.models import EfficientNet_B2_Weights
from base import BaseModel


class fCRPEfficientNet(BaseModel):
    def __init__(self, num_classes=7, imagenet_weights=False):
        super(fCRPEfficientNet, self).__init__()

        imagenet_weights = EfficientNet_B2_Weights.DEFAULT if imagenet_weights else None
        self.model = EfficientNetB2(weights=imagenet_weights)

        classifier = self.model.classifier
        classifier[-1] = nn.Linear(classifier[-1].in_features, num_classes) # change the last layer
        self.model.classifier = classifier

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)
