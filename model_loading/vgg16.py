import torch
import torch.nn as nn
from torchvision.models import VGG16_Weights, vgg16


class VGG16(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        if pretrained:
            self.model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        else:
            self.model = vgg16()
        self.model.classifier[0] = nn.Linear(in_features=512 * 7 * 7, out_features=512, bias=True)
        self.model.classifier[3] = nn.Linear(in_features=512, out_features=1024, bias=True)
        self.model.classifier[6] = nn.Linear(in_features=1024, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.model(x)
