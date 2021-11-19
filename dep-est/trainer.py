import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary

import matplotlib.pyplot as plt
import cv2
import os
import time
from copy import deepcopy
import gc
from pdb import set_trace

from models import *
from utils import *
from loss import *
from PATH import *
from data import *
from train import *


def initTrain():
    model_device = torch.device("cuda")   
    data_device = device = torch.device("cpu")       

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize( (320, 320) ),     
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    target_transform = transforms.Compose([transforms.Resize( (320, 320) )])

    dataset = DIODE(TRAIN_PATHS, transform=preprocess, target_transform=target_transform, device=data_device)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    m = RegSegModel("deeplab").to(model_device)
    m.train()
    model = train(m, dataloader, True, num_epoch=500, device=model_device)

    test_dataset = dataset
    testViz(model, test_dataset, "train-history")


def modelSummary():
    m = RegDeepLab().to(torch.device("cuda"))
    m.eval()
    summary(m.dl, input_size=(8, 3, 320, 320), device="cuda")
    summary(m.probe, input_size=(8, 21, 320, 320), device="cuda") 
    summary(m.dl.backbone, input_size=(8, 3, 320, 320), device="cuda") 


def testModel(model_path):
    model_device = torch.device("cuda")   
    data_device = device = torch.device("cpu")       

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize( (320, 320) ),     
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    target_transform = transforms.Compose([transforms.Resize( (320, 320) )])

    dataset = DIODE(TRAIN_PATHS, transform=preprocess, target_transform=target_transform, device=data_device, original=True)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    model = torch.load(model_path)

    test_dataset = dataset
    testViz(model, test_dataset, "train-history", num_example=10)


if __name__ == '__main__':
    # initTrain()
    # modelSummary()
    testModel(os.path.join("train-history", "trained_model179.pth"))