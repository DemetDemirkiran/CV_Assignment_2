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
    data_path = "D:\PycharmProjects\CV_Assignment_2-master\\fer2013.csv"
    ckpt_path = 'D:\PycharmProjects\CV_Assignment_2\\blockExtended_64_99.pth'#'D:\PycharmProjects\CV_Assignment_2\\blockCNN_64_999.pth'
    batch_size = 250
    num_labels = 7
    num_filters = 64
    kernel_size = (3, 3)

    public = getDataset(os.path.expanduser(data_path), mode='Public')
    private = getDataset(os.path.expanduser(data_path), mode='Private')
    public_loader = DataLoader(public, batch_size, False)
    private_loader = DataLoader(private, batch_size, False)

    device = torch.device("cuda")
    model = en.blockExtended(num_labels, num_filters, kernel_size)
    model.to(device)
    
    #validation
    model.eval()

    public_res = evaluate(model, ckpt_path, public_loader)
    private_res = evaluate(model, ckpt_path, private_loader)

    pp = pprint.PrettyPrinter()
    print('Public results')
    pp.pprint(public_res)
    print('_________________________')
    print('Private results')
    pp.pprint(private_res)
    print('_________________________')