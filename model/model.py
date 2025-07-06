import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b2 as EfficientNetB2
from torchvision.models import EfficientNet_B2_Weights
from torchvision.models import vgg16_bn as VGG
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
    def __init__(self, num_classes=7, n_mi_layer = 0, use_inter_layers: bool = False, fcrp_weights: str = None):
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

        if n_mi_layer > 0:
            if not use_inter_layers:
                self.mi_layer = nn.Sequential(
                    nn.Dropout(p=0.3, inplace=True),
                    nn.Linear(last_features_size, n_mi_layer),
                )
                pre_classifier[-1] = nn.Linear(n_mi_layer, num_classes)
            else:
                self.mi_layer = nn.Sequential(
                    # adapted from Koh et al. (2020)
                    nn.Dropout(p=0.3, inplace=True),
                    nn.Linear(last_features_size, n_mi_layer),
                    nn.ReLU(inplace=True),
                    nn.Linear(n_mi_layer, 50),
                    nn.ReLU(inplace=True),
                    nn.Linear(50, 50)
                )
                pre_classifier[-1] = nn.Linear(50, num_classes)
        else:
            self.mi_layer = None

        self.classifier = pre_classifier

    def forward(self, x, output_features=False):
        # from EfficientNet
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # MI layer

        if self.mi_layer is not None:
            x = self.mi_layer(x)

        inter_features = x

        # Classifier
        x = self.classifier(x)

        if output_features:
            weights = self.mi_layer[1].weight if self.mi_layer is not None else self.features[-1][0].weight
            return F.log_softmax(x, dim=1), weights, inter_features
        else:
            return F.log_softmax(x, dim=1)

class xMIVGG(BaseModel):
    def __init__(self, num_classes=7, vgg_weights: str = 'DEFAULT'):
        super(xMIVGG, self).__init__()

        # Previous model
        base_model = VGG(weights=vgg_weights)

        self.features = base_model.features
        self.avgpool = base_model.avgpool

        base_classifier = base_model.classifier
        self.intermediate_layers = nn.Sequential(*base_classifier[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(base_classifier[-1].in_features, num_classes)
        )

    def forward(self, x, output_features=False):
        # from VGG
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.intermediate_layers(x)
        inter_features = x

        # Classifier
        x = self.classifier(x)

        if output_features:
            weights = self.intermediate_layers[-3].weight
            return F.log_softmax(x, dim=1), weights, inter_features
        else:
            return F.log_softmax(x, dim=1)
