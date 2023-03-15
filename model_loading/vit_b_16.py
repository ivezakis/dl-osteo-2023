import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, vit_b_16
from torchvision.models.feature_extraction import create_feature_extractor


class ViT_B_16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.model = create_feature_extractor(self.model, return_nodes=['getitem_5'])
        self.classifier =  nn.Sequential(
            nn.Flatten(),
            nn.Linear(768, num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        x = x["getitem_5"]
        x = self.classifier(x)
        return x
