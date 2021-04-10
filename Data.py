#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import pandas as pd


# In[ ]:


class Dataset2(Dataset):
    """Stanford background dataset."""

    def __init__(self, image_dir, label_dir):
        """
        Args:
            image_dir (string): Directory with all the images.
            label_dir (string): Directory with all the labels.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.label_list = sorted(os.listdir(label_dir))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        image_ind = idx
        image = cv2.imread(os.path.join(self.image_dir,self.image_list[image_ind]),cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224,224), interpolation = cv2.INTER_NEAREST) # Resize to (224,224)
        image = Image.fromarray(image)
        data_transforms = transforms.Compose([
        transforms.ColorJitter(brightness = 1, contrast = 1, saturation = 1, hue = 0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = data_transforms(image)
        
        label_ind = idx*3 + 1
        label = pd.read_csv(os.path.join(self.label_dir,self.label_list[label_ind]), sep = " ", header = None)
        label = label.values
        label = cv2.resize(label, (224,224), interpolation = cv2.INTER_NEAREST) # Resize to (224,224)
        label = torch.LongTensor(label + 1) # Labels start from -1 where -1 means unknown.
        
        
        label_name = self.label_list[label_ind]
        image_name = self.image_list[image_ind]

        return image_name,label_name,image,label