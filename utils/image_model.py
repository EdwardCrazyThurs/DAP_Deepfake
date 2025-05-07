import torch
import torch.nn as nn
from torchvision import models

class DeepfakeDetectionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepfakeDetectionModel, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=False)
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, x):
        return self.efficientnet(x)
