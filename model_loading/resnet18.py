import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        if pretrained:
            self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.model = resnet18()
        self.model.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        return self.model(x)
