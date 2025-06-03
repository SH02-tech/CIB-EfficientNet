import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b2 as EfficientNetB2
from torchvision.models import EfficientNet_B2_Weights
from base import BaseModel


class fCRPEfficientNet(BaseModel):
    def __init__(self, num_classes=7, imagenet_weights=False):
        super(fCRPEfficientNet, self).__init__()

        imagenet_weights = EfficientNet_B2_Weights.DEFAULT if imagenet_weights else None
        model = EfficientNetB2(weights=imagenet_weights)

        self.features = model.features
        self.avgpool = model.avgpool

        classifier = model.classifier
        classifier[-1] = nn.Linear(classifier[-1].in_features, num_classes) # change the last layer
        self.classifier = classifier

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

class xMIEfficientNet(BaseModel):
    def __init__(self, num_classes=7, n_mi_layer = 30, fcrp_weights: str = None):
        super(xMIEfficientNet, self).__init__()

        # Previous model

        pre_model = fCRPEfficientNet(num_classes, imagenet_weights=True)

        if fcrp_weights is not None:
            chkpt_data = torch.load(fcrp_weights, weights_only=False)
            pre_state_dict = chkpt_data['state_dict']
            pre_model.load_state_dict(pre_state_dict)

        # get necessary layer sizes
        pre_classifier = pre_model.classifier
        last_features_size = pre_classifier[-1].in_features

        self.features = pre_model.features
        self.avgpool = pre_model.avgpool
        self.mi_layer = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(last_features_size, n_mi_layer),
        )
        
        pre_classifier[-1] = nn.Linear(n_mi_layer, num_classes)
        self.classifier = pre_classifier

    def forward(self, x, output_features=False):
        # from EfficientNet
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # MI layer
        x = self.mi_layer(x)
        inter_features = x

        # Classifier
        x = self.classifier(x)

        if output_features:
            return F.log_softmax(x, dim=1), self.mi_layer[-1].weight, inter_features
        else:
            return F.log_softmax(x, dim=1)
