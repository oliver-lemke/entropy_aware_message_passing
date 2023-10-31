import torch
from torch import nn


class ModelTemplate(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.MaxPool2d(2),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=1, dilation=2),
            nn.Conv2d(128, 256, kernel_size=2, stride=2, dilation=2),
            nn.MaxPool2d(2),
            nn.GELU(),
            nn.Conv2d(256, 512, kernel_size=2, stride=1, dilation=2),
            nn.MaxPool2d(2),
            nn.GELU(),
        )

        self.linear = nn.Linear(2048, 15)

        self.log_softmax = nn.LogSoftmax(dim=0)

    def forward(self, x):
        hidden = self.conv(x)

        hidden = torch.flatten(hidden, start_dim=1)

        output = self.log_softmax(self.linear(hidden))
        return output
