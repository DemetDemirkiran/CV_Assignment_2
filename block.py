import torch
import numpy as np
import torch.nn as nn

class basicBlock(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.act = nn.ReLU()
        self.pooling = nn.AvgPool2d((3, 3))

    def forward(self, data):

        data = self.conv(data)
        data = self.act(data)
        data = self.pooling(data)

        return data
