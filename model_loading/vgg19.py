import torch
import torch.nn as nn
from torchvision.models import VGG19_Weights, vgg19


class VGG19(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        if pretrained:
            self.model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        else:
            self.model = vgg19()
        self.model.classifier[0] = nn.Linear(in_features=512 * 7 * 7, out_features=512, bias=True)
        self.model.classifier[3] = nn.Linear(in_features=512, out_features=1024, bias=True)
        self.model.classifier[6] = nn.Linear(in_features=1024, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.model(x)
