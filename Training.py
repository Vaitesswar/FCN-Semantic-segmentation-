#!/usr/bin/env python
# coding: utf-8


from Data import Dataset
from FCN_Model import encoder,decoder
from UNet_Model import UNet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


image_dir = "Stanford background dataset\images"
label_dir = "Stanford background dataset\labels"
dataset = Dataset(image_dir,label_dir)
train_size = round(len(dataset)*0.9) # 90 % for training
test_size = len(dataset) - train_size # 10 % for testing
train_data,test_data = torch.utils.data.random_split(dataset, [train_size,test_size], generator = torch.manual_seed(42))
train_loader = DataLoader(train_data, batch_size = 2,shuffle = True, num_workers = 0)
test_loader = DataLoader(test_data, batch_size = 1,shuffle = True, num_workers = 0)


def save_checkpoint(state, filename = 'checkpoint.pth.tar'):
    torch.save(state, filename)

# Model training
net = decoder(encoder(), 9) # FCN
# net = UNet(9).float() # U-Net
net = net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum = 0.9)
epochs = 20

running_loss = 0
running_error = 0
num_batches = 0

for epoch in range(epochs):
    for i, data in enumerate(train_loader):
        image_name,label_name,inputs,labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        inputs.requires_grad_()

        scores = net(inputs.cuda())
        loss = criterion(scores,labels)
        
        if not math.isnan(loss.item()):
            loss.backward()
            optimizer.step()

            # compute and accumulate stats

            running_loss += loss.detach().item()
            num_batches += 1

            if (i+1) % 50 == 0:
                print('Epoch: ' + str(epoch) + '  ' + 'Batch: ' + str(i+1) + '/' + str(len(train_loader)) + '  ' + 'loss: ' + str(running_loss/num_batches))

    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict()
        })


    total_loss = running_loss/num_batches
    total_error = running_error/num_batches