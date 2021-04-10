#!/usr/bin/env python
# coding: utf-8

# In[1]:


from Data import Dataset
from FCN_Model import encoder,decoder
from UNet_Model import UNet
import torch
import torch.nn as nn
import torch.nn.functional as F
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
dataset = Dataset2(image_dir,label_dir)
train_size = round(len(dataset)*0.9) # 90 % for training
test_size = len(dataset) - train_size # 10 % for testing
train_data,test_data = torch.utils.data.random_split(dataset, [train_size,test_size], generator = torch.manual_seed(42))
train_loader = DataLoader(train_data, batch_size = 1,shuffle = True, num_workers = 0)
test_loader = DataLoader(test_data, batch_size = 1,shuffle = True, num_workers = 0)
net = decoder(encoder(), 9) # FCN
# net = UNet(9).float() # U-Net
net = net.cuda()
weights = torch.load('150 epochs.pth.tar') # Change this line
net.load_state_dict(weights['state_dict'])


'''
0 = unknown (black)
1 = sky (red)
2 = tree (green)
3 = road (olive)
4 = grass (blue)
5 = water (purple)
6 = building (blue-green)
7 = mountain (gray)
8 = foreground object (brown)
'''

def decode_segmap(image, nc = 9):
    label_colors = np.array([(0, 0, 0),   
                            
                           (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                           # , , 8=foreground object
                           (0, 128, 128), (128, 128, 128), (64, 0, 0)])
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)

    return rgb

def accuracy(pred, target):

    return np.sum(target == pred)/(np.shape(target)[0]*np.shape(pred)[1])

def mean_accuracy(pred, target):
    
    unique_classes = np.unique(target)
    accuracy = list()
    ind = 0
    
    for j in unique_classes:
        if np.sum(target == j) > 10:
            acc = np.sum(np.logical_and((pred == j),(target == j)))/np.sum((target == j))
            accuracy.append(acc)
            ind += 1
        
    mean_acc = sum(accuracy)/len(accuracy)

    return mean_acc

def mean_iou(pred, target):

    unique_classes = np.unique(target)
    IOU = list()
    ind = 0
    
    for j in unique_classes:
        if np.sum(target == j) > 10:
            u = np.sum(np.logical_or((pred == j),(target == j)))
            i = np.sum(np.logical_and((pred == j),(target == j)))
            iou = i/u
            IOU.append(iou)
            ind += 1
    
    mean_iou = sum(IOU)/len(IOU)

    return mean_iou


# Pixel accuracy, mean accuracy and mean IOU for 1 test image (for visualization)
for i, data in enumerate(train_loader):
    image_name,label_name,input,label = data
    input = input.cuda()
    label = label.cuda()
    
    score = net(input.float())
    label_pred = torch.argmax(score,dim = 1)
    c,h,w = label_pred.size()
    label_pred = label_pred.view([h,w])
    if i == 0:
        break

image = cv2.imread(os.path.join(image_dir,image_name[0]),cv2.IMREAD_COLOR)
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

label = pd.read_csv(os.path.join(label_dir,label_name[0]), sep = " ", header = None).values + 1
target = decode_segmap(label)
h,w,c = target.shape

label_pred = label_pred.cpu().numpy()
label_pred = cv2.resize(label_pred, (w,h), interpolation = cv2.INTER_NEAREST) # Resize to original size
pred = decode_segmap(label_pred)

get_ipython().run_line_magic('matplotlib', 'notebook')
plt.figure(figsize = [9,9])
plt.subplot(311)
plt.imshow(image)
plt.title('Original image')
plt.subplot(312)
plt.imshow(target)
plt.title('Actual map')
plt.subplot(313)
plt.imshow(pred)
plt.title('Predicted map')

acc = round(accuracy(label_pred, label)*100,1)
mean_acc = round(mean_accuracy(label_pred, label)*100,1)
iou = round(mean_iou(label_pred, label)*100,1)

print('The pixel accuracy is ' + str(acc) + '%.')
print('The mean accuracy is ' + str(mean_acc) + '%.')
print('The mean iou is ' + str(iou) + '%.')


# Pixel accuracy, mean acuuracy and mean IOU for all test images
ACC = list()
MEAN_ACC = list()
IOU = list()

for i, data in enumerate(test_loader):
    image_name,label_name,input,label = data
    input = input.cuda()
    
    score = net(input.float())
    label_pred = torch.argmax(score,dim = 1)
    c,h,w = label_pred.size()
    label_pred = label_pred.view([h,w])
    
    label = pd.read_csv(os.path.join(label_dir,label_name[0]), sep = " ", header = None).values + 1
    h,w = label.shape
    label_pred = label_pred.cpu().numpy()
    label_pred = cv2.resize(label_pred, (w,h), interpolation = cv2.INTER_NEAREST) # Resize to original size
    
    
    acc = round(accuracy(label_pred, label)*100,2)
    mean_acc = round(mean_accuracy(label_pred, label)*100,2)
    iou = round(mean_iou(label_pred, label)*100,2)
    
    ACC.append(acc)
    MEAN_ACC.append(mean_acc)
    IOU.append(iou)
    
acc = round(sum(ACC)/len(ACC),1)
mean_acc = round(sum(MEAN_ACC)/len(MEAN_ACC),1)
iou = round(sum(IOU)/len(IOU),1)

print('The pixel accuracy is ' + str(acc) + '%.')
print('The mean accuracy is ' + str(mean_acc) + '%.')
print('The mean iou is ' + str(iou) + '%.')

