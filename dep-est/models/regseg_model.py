import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import platform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision import transforms

from .model_utils import regPred2Img, clsPred2Img


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
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, 3, 3, groups=nin)
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, 1, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class Interpolate(nn.Module):
    def __init__(self, nin, nout):
        super(Interpolate, self).__init__()
        self.pointwise = nn.Conv2d(nin, nout, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, (3 * x.shape[2], 3 * x.shape[3]), mode='bilinear', align_corners=True)
        x = self.pointwise(x)
        return x
 

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
            Interpolate(63, 63),
            nn.BatchNorm2d(63),
            nn.LeakyReLU(),
            Interpolate(63, out_dim),
            #   nn.Sigmoid()
        )

    def forward(self, x):
        x = self.probe(x)
        return x


class ProbedDualTaskSeg(nn.Module):
    def __init__(self, backbone="deeplab", cls_num=35, depthwise=False):
        super().__init__()
        if backbone == "deeplab":
            self.backbone = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
        elif backbone == "fcn":
            self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
        for param in self.backbone.backbone.parameters():
            param.requires_grad = False
        self.cls_num = cls_num
        self.seg_head = ConvProbe(self.cls_num)
        self.reg_head = ConvProbe(1)
        if depthwise:
            self.seg_head = DepthWiseSeparableConvProbe(self.cls_num)
            self.reg_head = DepthWiseSeparableConvProbe(1)

    def forward(self, x):
        y_reg_ret = torch.ones(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        y_seg_ret = torch.ones(x.shape[0], self.cls_num, x.shape[2], x.shape[3], device=x.device)
        x = self.backbone(x)['out']
        y_reg = self.reg_head(x)
        y_seg = self.seg_head(x)

        y_reg_ret[:, :, :y_reg.shape[2], :y_reg.shape[3]] = y_reg
        y_seg_ret[:, :, :y_seg.shape[2], :y_seg.shape[3]] = y_seg
        return y_reg_ret, y_seg_ret


class DualTaskSeg(nn.Module):
    def __init__(self, backbone_type="deeplab", cls_num=35):
        super().__init__()
        if backbone_type == "deeplab":
            self.backbone = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
        elif backbone_type == "fcn":
            self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
        for param in self.backbone.backbone.parameters():
            param.requires_grad = False
        self.cls_num = cls_num
        self.head = nn.Conv2d(21, self.cls_num + 1, 3, 1)

    def forward(self, x):
        y_ret = torch.ones(x.shape[0], self.cls_num + 1, x.shape[2], x.shape[3], device=x.device)
        x = self.backbone(x)['out']
        y = self.head(x)
        y_ret[:, :, :y.shape[2], :y.shape[3]] = y

        y_reg_ret = y_ret[:,0,:,:].unsqueeze(1)
        y_seg_ret = y_ret[:,1:,:,:]

        return y_reg_ret, y_seg_ret


if __name__ == "__main__":
    pass
