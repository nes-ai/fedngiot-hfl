import torch.nn as nn
import torch.nn.functional as F
import math

class SimpleCNN(nn.Module):
    def __init__(self,
                 input_channels=1,
                 input_size=(28, 28),
                 num_classes=10,
                 n_conv_layers=1,
                 hidden_dim=128):
        super().__init__()
        self.convs = nn.ModuleList()
        channels = [input_channels] + [32 * (2 ** i) for i in range(n_conv_layers)]

        # convolution layers
        for i in range(n_conv_layers):
            self.convs.append(
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1)
            )

        # pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # calculate output size after conv + pooling
        h, w = input_size
        for _ in range(n_conv_layers):
            h = h // 2
            w = w // 2
        flatten_dim = channels[-1] * h * w

        self.fc1 = nn.Linear(flatten_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        for conv in self.convs:
            x = self.pool(F.relu(conv(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x