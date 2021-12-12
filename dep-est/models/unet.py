import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision


def small_resize(x, h, w):
    ret = torch.zeros(x.shape[0], x.shape[1], h, w, device=x.device)
    ret[:, :, :min(x.shape[2], h), :min(x.shape[3], w)] = x
    return ret


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
        return x


class DownBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(x)
        return x


class UpBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.interpolate(x, (x.shape[2] * 2, x.shape[3] * 2))
        return x


class ContractPath(nn.Module):
    def __init__(self, input_dim, channel_nums=[24, 64, 128]):
        super().__init__()
        self.hb1 = HorizontalBlock(input_dim, channel_nums[0])
        self.hb2 = HorizontalBlock(channel_nums[0], channel_nums[1])
        self.hb3 = HorizontalBlock(channel_nums[1], channel_nums[2])


        self.db = DownBlock()

    def forward(self, x):  # 3, 1024
        x1 = self.hb1(x)  # 24, 1024
        x1c = self.db(x1)  # 24, 512

        x2 = self.hb2(x1c)  # 64, 512
        x2c = self.db(x2)  # 64, 256

        x3 = self.hb3(x2c)  # 128, 256

        return x3, x2, x1,


class ExpandPath(nn.Module):
    def __init__(self, output_dim, channel_nums=[128, 64, 24]):
        super().__init__()
        self.hb1 = HorizontalBlock(channel_nums[0], channel_nums[1])
        self.hb2 = HorizontalBlock(channel_nums[1]*2, channel_nums[2])
        self.hb3 = HorizontalBlock(channel_nums[2]*2, output_dim)

        self.ub = UpBlock()

    def forward(self, cp):
        x = cp[0]  # 128, 256
        x = self.hb1(x)  # 64, 256
        x = self.ub(x)  # 64, 512

        x = small_resize(x, *cp[1].shape[2:])
        x = torch.cat((x, cp[1]), 1)  # 128, 512
        x = self.hb2(x)  # 24, 512
        x = self.ub(x)  # 24, 1024

        x = small_resize(x, *cp[2].shape[2:])
        x = torch.cat((x, cp[2]), 1)  # 48, 1024
        x = self.hb3(x)

        return x


class UNet(nn.Module):
    def __init__(self, input_dim, num_class, channel_nums=[24, 64, 128]):
        super().__init__()
        self.cp = ContractPath(input_dim, channel_nums=channel_nums)
        self.ep = ExpandPath(num_class, channel_nums=list(reversed(channel_nums)))

    def forward(self, x):
        feat_maps = self.cp(x)
        y = self.ep(feat_maps)
        y = small_resize(y, *x.shape[2:])
        return y


class DualTaskUNet(nn.Module):
    def __init__(self, input_dim=3, cls_num=35):
        super().__init__()
        cls_net = torchvision.models.mobilenet_v3_small(pretrained=True, progress=True)
        self.backbone = cls_net.features
        backbone_out_dim = self.backbone[-1].out_channels
        self.cls_num = cls_num
        self.unet = UNet(backbone_out_dim + 3, cls_num + 1)
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        _, _, input_h, input_w = x.shape
        xb = self.backbone(x)
        xb = F.interpolate(xb, (input_h // 5, input_w // 5))
        xo = TF.resize(x, xb.shape[2:])
        x = torch.cat((xb, xo), axis=1)
        y = self.unet(x)
        y = F.interpolate(y, (input_h, input_w))
        y_reg_ret = y[:, 0, :, :].unsqueeze(1)
        y_seg_ret = y[:, 1:, :, :]
        return y_reg_ret, y_seg_ret

    def combinedInference(self, x, feat):
        with torch.no_grad():
            _, _, input_h, input_w = x.shape
            xb = F.interpolate(feat, (input_h // 5, input_w // 5))
            xo = TF.resize(x, xb.shape[2:])
            x = torch.cat((xb, xo), axis=1)
            y = self.unet(x)
            y = F.interpolate(y, (input_h, input_w))
            y_reg_ret = y[:, 0, :, :].unsqueeze(1)
            y_seg_ret = y[:, 1:, :, :]
        return y_reg_ret, y_seg_ret


if __name__ == "__main__":
    unet = DualTaskUNet()
    md = torch.load("../train-history/trained_model24_dict.pth")
    unet.load_state_dict(md)