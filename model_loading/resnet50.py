import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNet50(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        if pretrained:
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.model = resnet50()
        self.model.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        return self.model(x)
