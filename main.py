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

class getDataset(data.Dataset):

    def __init__(self, csvpath, mode):
        self.csvpath = csvpath
        self.train_list, self.pub_list, self.priv_list = self.loadcsv()
        self.mode = mode

    def loadcsv(self):
        print("Loading CSV...")
        with open(self.csvpath, 'r') as f:
            text = f.readlines()
            text = [t.split(',') for t in text]

        emotions = []
        images = []
        usage = []
        training_usage = 0
        publictest_usage = 0
        privatetest_usage = 0

        for i in text[1:]:
            emotions.append(int(i[0]))
            usage.append(i[2])
            images.append( np.reshape(np.asarray(i[1].split(" "), dtype=int), (48, 48)) )

        usage = np.array(usage)
        emotions = np.array(emotions)
        images = np.array(images)

        training_usage = usage == 'Training\n'
        publictest_usage = emotions[usage == 'PublicTest\n']
        privatetest_usage = emotions[usage == 'PrivateTest\n']

        training_emotions = emotions[training_usage]
        training_public = emotions[publictest_usage]
        training_priv = emotions[privatetest_usage]

        training_pixels = np.uint8(images[training_usage]).astype("float32") / 255
        testingpub_pixels = np.uint8(images[publictest_usage]).astype("float32") / 255
        testingpriv_pixels = np.uint8(images[privatetest_usage]).astype("float32") / 255

        return (training_emotions, training_pixels), (training_public, testingpub_pixels), \
               (training_priv, testingpriv_pixels)

    def visualize(self, x=3, y=4):

        fig, ax = plt.subplots(x, y)
        for i in range(x):
            for j in range(y):
                idx = np.random.choice( range( len (self.train_list[0] ) ) )
                ax[i][j].imshow( self.train_list[1][idx].reshape((48, 48)), cmap=cm.gray )
        plt.show()

    def __len__(self):
        if self.mode == 'Training':
            return len(self.train_list[0])

        elif self.mode == 'Public':
            return len(self.pub_list[0])

        elif self.mode == 'Private':
            return len(self.priv_list[0])

    def __getitem__(self, index):

        if self.mode == 'Training':
            return self.train_list[0][index], np.expand_dims(self.train_list[1][index], axis=0)

        elif self.mode == 'Public':
            return self.pub_list[0][index], np.expand_dims(self.pub_list[1][index], axis=0)

        elif self.mode == 'Private':
            return self.priv_list[0][index], np.expand_dims(self.priv_list[1][index], axis=0)


if __name__ == '__main__':

    gd = getDataset(os.path.expanduser('~/Desktop/CV_2020/fer2013/fer2013.csv'), mode='Training')
    data_loader = DataLoader(gd, 2, False)
    #gd.visualize()

    device = torch.device("cpu")
    model = nw.blockCNN(7, 64)
    model.to(device)
    loss = nn.CrossEntropyLoss()
    loss.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    model.train()
    for ep in range(10):
        for label, image in tqdm(data_loader):
            opt.zero_grad()
            label = label.to(device)
            image = image.to(device)
            output = model(image)
            error = loss(output, label)
            error.backward()
            opt.step()
        print(error.data)
        torch.save(model.state_dict(), "ckpt")

    #validation
    model.eval()

    pt = getDataset(os.path.expanduser('~/Desktop/CV_2020/fer2013/fer2013.csv'), mode='Public')
    data_loader = DataLoader(pt, 2, False)
    correct_count = 0

    y_true = []
    y_pred = []
    with torch.no_grad():
        for label, image in tqdm(data_loader):
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











    ...