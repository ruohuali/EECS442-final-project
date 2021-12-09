
import cv2
import numpy as np
import os
import platform
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

from .model_utils import regPred2Img, clsPred2Img


class HorizontalBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3)
        # self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3)
        self.relu = nn.LeakyReLU(inplace=True)
        self.bn = nn.BatchNorm2d(output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        # x = self.conv2(x)
        # x = self.bn(x)
        # x = self.relu(x)
        return x


class DownBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(2)

    def forward(self, x):
        return self.avg_pool(x)


class UpBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.interpolate(x, (x.shape[2] * 2, x.shape[3] * 2))
        return x


class ContractPath(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.hb1 = HorizontalBlock(input_dim, 24)
        self.hb2 = HorizontalBlock(24, 64)
        self.hb3 = HorizontalBlock(64, 128)

        self.db = DownBlock()

    def forward(self, x):  # 3, 1024
        x1 = self.hb1(x)  # 24, 1024
        x1c = self.db(x1)  # 24, 512

        x2 = self.hb2(x1c)  # 64, 512
        x2c = self.db(x2)  # 64, 256

        x3 = self.hb3(x2c)  # 128, 256

        return x3, x2, x1,


class ExpandPath(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.hb1 = HorizontalBlock(128, 64)
        self.hb2 = HorizontalBlock(128, 24)
        self.hb3 = HorizontalBlock(48, output_dim)

        self.ub = UpBlock()

    def forward(self, cp):
        x = cp[0]  # 128, 256
        x = self.hb1(x)  # 64, 256
        x = self.ub(x)  # 64, 512

        x = TF.resize(x, cp[1].shape[2:])
        x = torch.cat((x, cp[1]), 1)  # 128, 512
        x = self.hb2(x)  # 24, 512
        x = self.ub(x)  # 24, 1024

        x = TF.resize(x, cp[2].shape[2:])
        x = torch.cat((x, cp[2]), 1)  # 48, 1024
        x = self.hb3(x)

        return x


class UNet(nn.Module):
    def __init__(self, input_dim, num_class):
        super().__init__()
        self.cp = ContractPath(input_dim)
        self.ep = ExpandPath(num_class)

    def forward(self, x):
        feat_maps = self.cp(x)
        y = self.ep(feat_maps)
        y = TF.resize(y, x.shape[2:])
        return y


class DualTaskUNet(nn.Module):
    def __init__(self, input_dim=3, cls_num=35):
        super().__init__()
        self.cls_num = cls_num
        self.unet = UNet(input_dim, cls_num + 1)

    def forward(self, x):
        y = self.unet(x)
        y_reg_ret = y[:, 0, :, :].unsqueeze(1)
        y_reg_ret = TF.resize(y_reg_ret, x.shape[2:])
        y_seg_ret = y[:, 1:, :, :]
        y_seg_ret = TF.resize(y_seg_ret, x.shape[2:])
        return y_reg_ret, y_seg_ret

    def showInference(self, img_path, preprocess=transforms.Compose([transforms.ToTensor(),
                                                                     transforms.Resize((200, 640)),
                                                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                          std=[0.229, 0.224, 0.225])])):
        """
        @func img_path -> 3 np arrays of results
        """

        def clearTemp():
            if platform.system() != "Windows":
                os.system("rm temp.png")
            else:
                raise NotImplementedError("write win cmd for removing temp file")

        img = Image.open(img_path)
        img_t = preprocess(img)
        img_t = img_t.unsqueeze(0)
        with torch.no_grad():
            reg_pred, seg_pred = self.forward(img_t)
        reg_pred, seg_pred = regPred2Img(reg_pred), clsPred2Img(seg_pred)

        reg_pred_o, seg_pred_o = reg_pred.clone(), seg_pred.clone()
        img_o = np.array(img)
        img_o = cv2.resize(img_o, (reg_pred_o.shape[1], reg_pred_o.shape[0]))

        reg_pred = reg_pred.max() - reg_pred
        reg_pred7 = reg_pred.clone()
        reg_pred7[seg_pred != 7] = 0
        reg_pred26 = reg_pred.clone()
        reg_pred26[seg_pred != 26] = 0
        reg_pred = reg_pred7 + reg_pred26
        reg_pred[reg_pred == 0] = float('nan')

        ##
        cmap = plt.cm.jet
        cmap.set_bad(color="black")

        plt.figure()
        plt.imshow(reg_pred.numpy(), cmap=cmap, alpha=0.97)
        plt.imshow(img_o, alpha=0.6)
        plt.savefig("temp.png", bbox_inches='tight', pad_inches=0)

        img_arr = cv2.imread("temp.png")
        clearTemp()

        ##
        plt.figure()
        plt.imshow(reg_pred.numpy(), cmap=cmap)
        plt.savefig("temp.png", bbox_inches='tight', pad_inches=0)

        reg_pred_arr = cv2.imread("temp.png")
        clearTemp()

        ##
        plt.figure()
        plt.imshow(seg_pred.numpy())
        plt.savefig("temp.png", bbox_inches='tight', pad_inches=0)

        seg_pred_arr = cv2.imread("temp.png")
        clearTemp()

        return reg_pred_arr, seg_pred_arr, img_arr


if __name__ == "__main__":
    x = torch.ones(8, 3, 640, 120)
    unet = DualTaskUNet(3, 10)
    y = unet(x)
    print(y[0].shape, y[1].shape)
