import torch
import torch.nn as nn


class MobileNetV2(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=pretrained)
        self.model.classifier[1] = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        return self.model(x)