import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import time
from copy import deepcopy
import matplotlib.pyplot as plt
import os

from utils import *


class ConvProbe(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.probe = nn.Sequential(
            nn.Conv2d(21, 21 * 3, 3, 3),
            nn.BatchNorm2d(21 * 3),
            nn.LeakyReLU(),
            nn.Conv2d(21 * 3, 21 * 3, 3, 3),
            nn.BatchNorm2d(21 * 3),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(21 * 3, 21, 3, 3),
            nn.BatchNorm2d(21),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(21, out_dim, 3, 3),
            #   nn.Sigmoid()
        )

    def forward(self, x):
        x = self.probe(x)
        return x


class DepthWiseSeparableConv2d(nn.Module):
    def __init__(self, nin, kernels_per_layer, nout):
        super(DepthWiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, 3, 1, groups=nin)
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, 1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class DepthWiseSeparableConvProbe(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.probe = nn.Sequential(
            DepthWiseSeparableConv2d(21, 1, 63),
            nn.BatchNorm2d(63),
            nn.LeakyReLU(),
            DepthWiseSeparableConv2d(63, 1, 63),
            nn.BatchNorm2d(63),
            nn.LeakyReLU(),
            DepthWiseSeparableConv2d(63, 1, 63),
            nn.BatchNorm2d(63),
            nn.LeakyReLU(),
            DepthWiseSeparableConv2d(63, 1, out_dim),
            #   nn.Sigmoid()
        )

    def forward(self, x):
        x = self.probe(x)
        return x


class RegSegModel(nn.Module):
    def __init__(self, base_type="deeplab", cls_num=35):
        super().__init__()
        if base_type == "deeplab":
            self.base = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
        elif base_type == "fcn":
            self.base = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
        for param in self.base.backbone.parameters():
            param.requires_grad = False
        self.cls_num = cls_num
        self.seg_head = ConvProbe(self.cls_num)
        self.reg_head = ConvProbe(1)

    def forward(self, x):
        y_reg_ret = torch.ones(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        y_seg_ret = torch.ones(x.shape[0], self.cls_num, x.shape[2], x.shape[3], device=x.device)
        x = self.base(x)['out']
        y_reg = self.reg_head(x)
        y_seg = self.seg_head(x)

        y_reg_ret[:, :, :y_reg.shape[2], :y_reg.shape[3]] = y_reg
        y_seg_ret[:, :, :y_seg.shape[2], :y_seg.shape[3]] = y_seg
        return y_reg_ret, y_seg_ret

    def showInference(self, img_path, save_dir, preprocess, i):
        img = Image.open(img_path)
        img_t = preprocess(img)
        img_t = img_t.unsqueeze(0)
        with torch.no_grad():
            reg_pred, seg_pred = self.forward(img_t)
        reg_pred, seg_pred = regPred2Img(reg_pred), clsPred2Img(seg_pred)

        reg_pred_o, seg_pred_o = reg_pred.clone(), seg_pred.clone()

        reg_pred = reg_pred.max() - reg_pred
        reg_pred7 = reg_pred.clone()
        reg_pred7[seg_pred != 7] = 0
        reg_pred26 = reg_pred.clone()
        reg_pred26[seg_pred != 26] = 0
        reg_pred = reg_pred7 + reg_pred26
        reg_pred[reg_pred == 0] = float('nan')

        cmap = plt.cm.jet
        cmap.set_bad(color="black")

        plt.figure()
        plt.imshow(img)
        plt.savefig(os.path.join(save_dir, "infer_img" + str(i) + ".png"))

        plt.figure()
        # h = depth2Heatmap(reg_pred.numpy())
        # plt.imshow(h, alpha=0.9)
        plt.imshow(reg_pred.numpy(), cmap=cmap)
        plt.imshow(seg_pred.numpy(), alpha=0.3)
        plt.savefig(os.path.join(save_dir, "infer_pred" + str(i) + ".png"))

        plt.figure()
        # h = depth2Heatmap(reg_pred.numpy())
        # plt.imshow(h)
        plt.imshow(reg_pred.numpy(), cmap=cmap)
        plt.savefig(os.path.join(save_dir, "infer_pred_reg" + str(i) + ".png"))

        plt.figure()
        plt.imshow(seg_pred.numpy(), alpha=0.9)
        plt.savefig(os.path.join(save_dir, "infer_pred_seg" + str(i) + ".png"))

        return reg_pred_o, seg_pred_o


if __name__ == "__main__":
    pass
