import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout

class VGG11Classifier(nn.Module):
    def __init__(self, num_classes=37, in_channels=3, dropout_p=0.5, use_bn=True):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels, use_bn=use_bn)
        
        # Fully connected layers for categorization
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes)
        )

    def forward(self, inputs):
        # Extract features and flatten for the FC layers
        features = self.encoder(inputs, return_features=False)
        return self.classifier(features.view(features.size(0), -1))

# """Classification components"""
# import torch
# import torch.nn as nn
# from .vgg11 import VGG11Encoder
# from .layers import CustomDropout

# class VGG11Classifier(nn.Module):
#     def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5, use_bn: bool = True):
#         super().__init__()
#         self.encoder = VGG11Encoder(in_channels, use_bn=use_bn)
        
#         self.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(inplace=True),
#             CustomDropout(p=dropout_p),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             CustomDropout(p=dropout_p),
#             nn.Linear(4096, num_classes)
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.encoder(x, return_features=False)
#         x = torch.flatten(x, 1)
#         return self.classifier(x)
