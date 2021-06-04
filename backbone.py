from torch import nn
from torchvision import models


class ResNet18(nn.Module):
    def __init__(self, dims=(512, 100), args=None, pretrained=True):
        super().__init__()

        self.args = args

        # pretrained feature extractor
        self.FE = models.resnet18(pretrained=pretrained)
        self.FE.fc = nn.Linear(512, dims[0])
        self.clf_layer = nn.Linear(dims[0], dims[1])

    def forward(self, x, classify=True):
        features = self.FE(x)
        if classify:
            return self.clf_layer(features)
        return features