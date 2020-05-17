import torch
import numpy as np
import torch.nn as nn

class extendedCNN(nn.Module):

    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels

        self.conv_1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.act_1 = nn.ReLU()
        self.pooling1 = nn.AvgPool2d((3, 3))

        self.conv_2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.act_2 = nn.ReLU()
        self.pooling2 = nn.AvgPool2d((3, 3))

        self.conv_3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.act_3 = nn.ReLU()
        self.pooling3 = nn.AvgPool2d((3, 3))

        self.conv_4 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.act_4 = nn.ReLU()
        self.pooling4 = nn.AvgPool2d((3, 3))

        self.conv_5 = nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.act_5 = nn.ReLU()
        self.pooling5 = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(1024, num_labels)

    def forward(self, data):

        data = self.conv_1(data)
        data = self.act_1(data)
        data = self.pooling1(data)

        data = self.conv_2(data)
        data = self.act_2(data)
        data = self.pooling2(data)

        data = self.conv_3(data)
        data = self.act_3(data)
        data = self.pooling3(data)

        data = self.conv_4(data)
        data = self.act_4(data)
        data = self.pooling4(data)

        data = data.view(data.size(0), -1)
        data = self.fc(data)

        return data

    ...

