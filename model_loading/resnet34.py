import torch.nn as nn
from torchvision.models import ResNet34_Weights, resnet34


class ResNet34(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        if pretrained:
            self.model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        else:
            self.model = resnet34()
        self.model.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        return self.model(x)
