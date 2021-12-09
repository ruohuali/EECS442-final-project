import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from torchvision.models.feature_extraction import create_feature_extractor


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
        x = self.avg_pool(x)
        return x


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
        cls_net = torchvision.models.mobilenet_v3_small(pretrained=True, progress=True)
        self.backbone = cls_net.features
        backbone_out_dim = self.backbone[-1].out_channels
        self.cls_num = cls_num
        self.convert_conv = nn.Conv2d(backbone_out_dim, 24, 1)
        self.unet = UNet(24, cls_num + 1)
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        _, _, input_h, input_w = x.shape
        x = self.backbone(x)
        x = self.convert_conv(x)
        x = F.interpolate(x, (input_h, input_w))
        y = self.unet(x)
        y = TF.resize(y, x.shape[2:])
        y_reg_ret = y[:, 0, :, :].unsqueeze(1)
        y_seg_ret = y[:, 1:, :, :]
        return y_reg_ret, y_seg_ret


if __name__ == "__main__":
    b = torchvision.models.mobilenet_v3_small(pretrained=True, progress=True, dilated=True)
    bf = b.features
    print(bf)
    x = torch.ones(8, 3, 320, 320)
    y = b(x)
    print(y.shape)
    y = bf(x)
    print(y.shape)


    stage_indices = [0] + [i for i, b in enumerate(bf) if getattr(b, "_is_cn", False)] + [len(bf) - 1]
    out_pos = stage_indices[-1]  # use C5 which has output_stride = 16
    out_inplanes = bf[out_pos].out_channels
    print("stage ids")
    print(stage_indices)
    print("out pos")
    print(out_pos)
    print("out inplane")
    print(out_inplanes)
    print("type", type(bf), bf[-1].out_channels)
