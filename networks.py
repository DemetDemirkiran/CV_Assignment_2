import torch
import numpy as np
import torch.nn as nn

class basicCNN(nn.Module):

    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels

        self.conv_1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.act_1 = nn.ReLU()
        self.pooling1 = nn.AvgPool2d((3, 3))

        self.conv_2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.act_2 = nn.ReLU()
        self.pooling2 = nn.AdaptiveAvgPool2d(1)
        self.feature = nn.Sequential()
        self.fc = nn.Linear(128, num_labels)

    def forward(self, data):

        data = self.conv_1(data)
        data = self.act_1(data)
        data = self.pooling1(data)

        data = self.conv_2(data)
        data = self.act_2(data)
        data = self.pooling2(data)

        data = data.view(data.size(0), -1)
        data = self.fc(data)

        return data

    ...

