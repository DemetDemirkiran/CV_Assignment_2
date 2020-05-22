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
from sklearn.metrics import classification_report, precision_score
from dataset import getDataset
import pprint
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, required=True, help='Path to yaml file.')


def evaluate(model, ckpt_path, loader):
    
    model.load_state_dict(torch.load(ckpt_path))
    
    y_true = []
    y_pred = []
    with torch.no_grad():
        for label, image in tqdm(loader):
            label = label.to(device)
            image = image.to(device)
            output = model(image)
            prediction = torch.sigmoid(output)
            prediction = torch.argmax(prediction, dim=1)
            y_true.append(label.cpu().numpy())
            y_pred.append(prediction.cpu().numpy())
    # Flatten arrays
    y_true = [g for f in y_true for g in f]
    y_pred = [g for f in y_pred for g in f]
    cl_report = classification_report(y_true, y_pred, labels=list(range(7)), output_dict=True)
    pr = precision_score(y_true, y_pred, average='micro')
    print('Precision:', pr)
    return cl_report


if __name__ == '__main__':
    args = parser.parse_args()
    model_dict = {'blockCNN': nw.blockCNN,
                  'extendedCNN': en.extendedCNN}
    with open(args.config_file, 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)

    public = getDataset(os.path.expanduser(config['data_path']), mode='Public')
    private = getDataset(os.path.expanduser(config['data_path']), mode='Private')
    public_loader = DataLoader(public, config['batch_size'], False)
    private_loader = DataLoader(private, config['batch_size'], False)

    device = torch.device(config['device'])
    model = model_dict[config['model']](config['num_labels'], config['num_filters'], config['kernel_size'])
    model.to(device)
    
    #validation
    model.eval()

    public_res = evaluate(model, config['ckpt_path'], public_loader)
    private_res = evaluate(model, config['ckpt_path'], private_loader)

    pp = pprint.PrettyPrinter()
    print('Public results')
    pp.pprint(public_res)
    print('_________________________')
    print('Private results')
    pp.pprint(private_res)
    print('_________________________')