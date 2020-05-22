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
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, required=True, help='Path to yaml file.')


if __name__ == '__main__':
    args = parser.parse_args()
    model_dict = {'blockCNN': nw.blockCNN,
                  'extendedCNN': en.extendedCNN}
    with open(args.config_file, 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)

    gd = getDataset(os.path.expanduser(config['data_path']), mode='Training')
    data_loader = DataLoader(gd, config['batch_size'], False)

    device = torch.device(config['device'])

    model = model_dict[config['model']](config['num_labels'], config['num_filters'], config['kernel_size'])
    model.to(device)
    loss = nn.CrossEntropyLoss()
    loss.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(config['learning_rate']))
    model.train()
    for ep in tqdm(range(config['epochs'])):
        for label, image in data_loader:
            opt.zero_grad()
            label = label.to(device).long()
            image = image.to(device)
            output = model(image)
            error = loss(output, label)
            error.backward()
            opt.step()
        print(error.data)
        if ep % config['ckpt_step'] == 0 and ep > 0:
            torch.save(model.state_dict(), "{}_{}_{}.pth".format(type(model).__name__, config['num_filters'], ep))
    torch.save(model.state_dict(), "{}_{}_{}.pth".format(type(model).__name__, config['num_filters'], ep))

