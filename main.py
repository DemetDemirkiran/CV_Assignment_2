import torch
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import torch.utils.data as data
from torch.utils.data.dataloader import DataLoader
import networks as nw
import extendednetworks as en
import torch.nn as nn
from sklearn.metrics import classification_report
from dataset import getDataset


if __name__ == '__main__':
    data_path = "D:\PycharmProjects\CV_Assignment_2-master\\fer2013.csv"
    batch_size = 250
    num_labels = 7
    num_filters = 64
    epochs = 100
    ckpt_step = 25
    gd = getDataset(os.path.expanduser(data_path), mode='Training')
    data_loader = DataLoader(gd, batch_size, False)

    device = torch.device("cuda")
    model = en.blockExtended(num_labels, num_filters)
    model.to(device)
    loss = nn.CrossEntropyLoss()
    loss.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    model.train()
    for ep in tqdm(range(epochs)):
        for label, image in data_loader:
            opt.zero_grad()
            label = label.to(device).long()
            image = image.to(device)
            output = model(image)
            error = loss(output, label)
            error.backward()
            opt.step()
        print(error.data)
        if ep % ckpt_step == 0 and ep > 0:
            torch.save(model.state_dict(), "{}_{}_{}.pth".format(type(model).__name__, num_filters, ep))
    torch.save(model.state_dict(), "{}_{}_{}.pth".format(type(model).__name__, num_filters, ep))

