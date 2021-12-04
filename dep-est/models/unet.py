import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode


class HorizontalBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(inplace=True)
        self.bn = nn.BatchNorm2d(output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DownBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.max_pool(x)


class UpBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.tconv = nn.ConvTranspose2d(input_dim, input_dim // 2, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.tconv(x)
        return x


class ContractPath(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.hb1 = HorizontalBlock(input_dim, 64)
        self.hb2 = HorizontalBlock(64, 128)
        self.hb3 = HorizontalBlock(128, 256)
        self.hb4 = HorizontalBlock(256, 512)
        self.hb5 = HorizontalBlock(512, 1024)

        self.db = DownBlock()

    def forward(self, x):
        x1 = self.hb1(x)
        x1c = self.db(x1)

        x2 = self.hb2(x1c)
        x2c = self.db(x2)

        x3 = self.hb3(x2c)
        x3c = self.db(x3)

        x4 = self.hb4(x3c)
        x4c = self.db(x4)

        x5 = self.hb5(x4c)
        return x5, x4, x3, x2, x1


class ExpandPath(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.hb1 = HorizontalBlock(1024, 512)
        self.hb2 = HorizontalBlock(512, 256)
        self.hb3 = HorizontalBlock(256, 128)
        self.hb4 = HorizontalBlock(128, 64)
        self.out_conv = nn.Conv2d(64, output_dim, kernel_size=1)

        self.ub1 = UpBlock(1024)
        self.ub2 = UpBlock(512)
        self.ub3 = UpBlock(256)
        self.ub4 = UpBlock(128)

    def forward(self, cp):
        x = cp[0]
        x = self.ub1(x)  # 512

        x_padded = torch.zeros(*cp[1].shape, device=x.device)
        x_padded[:, :, :x.shape[2], :x.shape[3]] = x
        x = torch.cat((x_padded, cp[1]), 1)  # 1024
        x = self.hb1(x)  # 512
        x = self.ub2(x)  # 256

        x_padded = torch.zeros(*cp[2].shape, device=x.device)
        x_padded[:, :, :x.shape[2], :x.shape[3]] = x
        x = torch.cat((x_padded, cp[2]), 1)  # 512
        x = self.hb2(x)  # 256
        x = self.ub3(x)  # 128

        x_padded = torch.zeros(*cp[3].shape, device=x.device)
        x_padded[:, :, :x.shape[2], :x.shape[3]] = x
        x = torch.cat((x_padded, cp[3]), 1)  # 256
        x = self.hb3(x)  # 128
        x = self.ub4(x)  # 64

        x_padded = torch.zeros(*cp[4].shape, device=x.device)
        x_padded[:, :, :x.shape[2], :x.shape[3]] = x
        x = torch.cat((x_padded, cp[4]), 1)  # 128
        x = self.hb4(x)  # 64

        x = self.out_conv(x)  # num class

        return x


class UNet(nn.Module):
    def __init__(self, input_dim, num_class):
        super().__init__()
        self.cp = ContractPath(input_dim)
        self.ep = ExpandPath(num_class)

    def forward(self, x):
        feat_maps = self.cp(x)
        y = self.ep(feat_maps)
        return y


class RegUNet(nn.Module):
    def __init__(self, input_dim, last_dim):
        super().__init__()
        self.unet = UNet(input_dim, last_dim)
        self.out_conv = nn.Conv2d(last_dim, 1, kernel_size=1, stride=1)
        self.out_nonlin = nn.Tanh()

    def forward(self, x):
        x = self.unet(x)
        x = self.out_nonlin(self.out_conv(x))
        return x
