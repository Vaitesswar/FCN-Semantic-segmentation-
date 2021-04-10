#!/usr/bin/env python
# coding: utf-8

# In[17]:


import torch
from torchvision import models
import torch.nn.functional as F
from torch import nn

class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.decode(x)


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        model = models.vgg13_bn(pretrained = True)
        self.enc1 = _EncoderBlock(3, 64)
        weights = model.features[0:6].state_dict()
        self.enc1.encode.state_dict().update(weights)
        self.enc1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = _EncoderBlock(64, 128)
        weights = model.features[7:13].state_dict()
        self.enc2.encode.state_dict().update(weights)
        self.enc2_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = _EncoderBlock(128, 256)
        weights = model.features[14:20].state_dict()
        self.enc3.encode.state_dict().update(weights)
        self.enc3_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc4 = _EncoderBlock(256, 512)
        weights = model.features[21:27].state_dict()
        self.enc4.encode.state_dict().update(weights)
        self.enc4_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.center = _DecoderBlock(512, 1024, 512)
        self.dec4 = _DecoderBlock(1024, 512, 256)
        self.dec3 = _DecoderBlock(512, 256, 128)
        self.dec2 = _DecoderBlock(256, 128, 64)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        self.upfinal = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=26, stride=2)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc1_pool = self.enc1_pool(enc1)
        enc2 = self.enc2(enc1_pool)
        enc2_pool = self.enc2_pool(enc2)
        enc3 = self.enc3(enc2_pool)
        enc3_pool = self.enc3_pool(enc3)
        enc4 = self.enc4(enc3_pool)
        enc4_pool = self.enc4_pool(enc4)
        center = self.center(enc4_pool)
        dec4 = self.dec4(torch.cat([center, F.interpolate(enc4, center.size()[2:], mode='bilinear')], 1))
        dec3 = self.dec3(torch.cat([dec4, F.interpolate(enc3, dec4.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.interpolate(enc2, dec3.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.interpolate(enc1, dec2.size()[2:], mode='bilinear')], 1))
        final = self.upfinal(self.final(dec1))
        return final