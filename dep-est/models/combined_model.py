import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision

from yolov1 import *
from unet import UNet

class TriTaskModel(nn.Module):
    def __init__(self, det_model, regseg_model):
        feature_extractor = torchvision.models.mobilenet_v3_small(pretrained=True, progress=True)
        self.feature_extractor = feature_extractor.features
        self.det_head = det_model
        self.regseg_head = regseg_model

    def forward(self, x):
        feat = self.feature_extractor(x)


        SingleDetectionInference(self.det_head, feat, )
