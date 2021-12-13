import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
import torch.nn.functional as F


class AtrousConv(nn.Module):
    def __init__(self, input_dim, output_dim, dilate_rate):
        super(AtrousConv, self).__init__()
        self.aconv = nn.Conv2d(input_dim, output_dim, 3, padding=dilate_rate, dilation=dilate_rate)
        self.bn = nn.BatchNorm2d(output_dim)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.aconv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x


class SPPooling(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SPPooling, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(input_dim, output_dim, 1)
        self.relu = nn.LeakyReLU()
        self.bn = nn.BatchNorm2d(output_dim)

    def forward(self, x):
        input_h, input_w = x.shape[-2:]
        x = self.pool(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        x = TF.resize(x, (input_h, input_w))
        return x


class ASPP(nn.Module):
    def __init__(self, input_dim, output_dim, dilate_rates=[4, 8, 16]):
        super(ASPP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilate_rates = dilate_rates
        self.atrous_convs = [AtrousConv(self.input_dim , 64, self.dilate_rates[0])]
        for dilate_rate in self.dilate_rates[1:]:
            self.atrous_convs.append(AtrousConv(self.input_dim, 64, dilate_rate))
        self.atrous_convs = nn.ModuleList(self.atrous_convs)
        self.sp_pool = SPPooling(self.input_dim, 64)
        self.project = nn.Conv2d(64 * (len(self.atrous_convs) + 1), self.output_dim, 3, padding=1)

    def forward(self, x):
        input_h, input_w = x.shape[-2:]
        outputs = []
        for aconv in self.atrous_convs:
            y = aconv(x)
            y = TF.resize(y, (input_h, input_w))
            outputs.append(y)
        outputs.append(self.sp_pool(x))
        outputs = torch.cat(outputs, 1)
        outputs = self.project(outputs)
        return outputs


if __name__ == "__main__":
    aspp = ASPP(3, 32)
    x = torch.ones(10, 3, 320, 320)
    y = aspp(x)
    print(y.shape)



