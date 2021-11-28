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
from train_mult import *


def initTrainDIODE():
    model_device = torch.device("cuda")   
    data_device = device = torch.device("cpu")       

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize( (320, 320) ),     
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    target_transform = transforms.Compose([transforms.ToTensor(),
                                           transforms.Resize( (320, 320) )])

    dataset = DIODE(TRAIN_PATHS, transform=preprocess, target_transform=target_transform, device=data_device)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    test_dataset = DIODE(TEST_PATHS, transform=preprocess, target_transform=target_transform, device=data_device)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

    m = RegSegModel("deeplab").to(model_device)
    m.train()
    model = trainSingle(m, dataloader, test_dataloader, "reg", num_epoch=100, device=model_device)
    
    m.eval()
    test_dataset = DIODE(TEST_PATHS, transform=preprocess, target_transform=target_transform, device=data_device, original=True)
    testViz(model, test_dataset, "train-history")


def initTrainKITTIReg():
    model_device = torch.device("cuda")   
    data_device = device = torch.device("cpu")       

    dataset = KITTI_DEP(KITTI_TRAIN_RGB_PATHS, KITTI_TRAIN_LABEL_PATHS, device=data_device)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, drop_last=True)
    test_dataset = KITTI_DEP(KITTI_TEST_RGB_PATHS, KITTI_TEST_LABEL_PATHS, device=data_device)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, drop_last=True)

    m = RegSegModel().to(model_device)
    m.train()
    
    model = trainSingle(m, dataloader, test_dataloader, "reg", num_epoch=100, device=model_device)

    test_dataset = KITTI_DEP(KITTI_TEST_RGB_PATHS, KITTI_TEST_LABEL_PATHS, device=data_device, original=True)
    m.eval()
    testViz(model, test_dataset, "train-history")    


def initTrainKITTISeg():
    model_device = torch.device("cuda")   
    data_device = device = torch.device("cpu")       

    dataset = KITTI_SEM(KITTI_SEM_TRAIN_RGB_PATHS, KITTI_SEM_TRAIN_LABEL_PATHS, device=data_device)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, drop_last=True)
    test_dataset = KITTI_SEM(KITTI_SEM_TRAIN_RGB_PATHS, KITTI_SEM_TRAIN_LABEL_PATHS, device=data_device)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, drop_last=True)

    m = RegSegModel().to(model_device)
    m.train()
    model = trainSingle(m, dataloader, test_dataloader, "seg", num_epoch=100, device=model_device)

    test_dataset = KITTI_SEM(KITTI_SEM_TRAIN_RGB_PATHS, KITTI_SEM_TRAIN_LABEL_PATHS, device=data_device, original=True)
    m.eval()
    testViz(model, test_dataset, "train-history")    


def initTrainKITTIDual():
    model_device = torch.device("cuda")   
    data_device = device = torch.device("cpu")       

    reg_dataset = KITTI_DEP(KITTI_TRAIN_RGB_PATHS, KITTI_TRAIN_LABEL_PATHS, device=data_device)
    reg_dataloader = DataLoader(reg_dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True)
    test_reg_dataset = KITTI_DEP(KITTI_TEST_RGB_PATHS, KITTI_TEST_LABEL_PATHS, device=data_device)
    test_reg_dataloader = DataLoader(test_reg_dataset, batch_size=2, shuffle=False, num_workers=0, drop_last=True)    

    seg_dataset = KITTI_SEM(KITTI_SEM_TRAIN_RGB_PATHS, KITTI_SEM_TRAIN_LABEL_PATHS, device=data_device)
    seg_dataloader = DataLoader(seg_dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True)
    test_seg_dataset = KITTI_SEM(KITTI_SEM_TRAIN_RGB_PATHS, KITTI_SEM_TRAIN_LABEL_PATHS, device=data_device)
    test_seg_dataloader = DataLoader(test_seg_dataset, batch_size=2, shuffle=False, num_workers=0, drop_last=True)

    m = RegSegModel().to(model_device)
    m.train()
    print("train dataloader len", len(reg_dataloader), len(seg_dataloader))
    model = trainDual(m, reg_dataloader, test_reg_dataloader, seg_dataloader, test_seg_dataloader, num_epoch=100, device=model_device)

    test_dataset = KITTI_DEP(KITTI_TEST_RGB_PATHS, KITTI_TEST_LABEL_PATHS, device=data_device, original=True)
    m.eval()
    testViz(model, test_dataset, "train-history")    

    test_dataset = KITTI_SEM(KITTI_SEM_TRAIN_RGB_PATHS, KITTI_SEM_TRAIN_LABEL_PATHS, device=data_device, original=True)
    m.eval()
    testViz(model, test_dataset, "train-history")        


def modelSummary():
    m = ConvProbe(21)
    m.eval()
    summary(m, input_size=(8, 21, 320, 320), device="cuda")


def testModel(model_path):
    model_device = torch.device("cuda")   
    data_device = device = torch.device("cpu")       

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize( (320, 320) ),     
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    target_transform = transforms.Compose([transforms.Resize( (320, 320) )])

    dataset = DIODE(TEST_PATHS, transform=preprocess, target_transform=target_transform, device=data_device, original=True)

    model = torch.load(model_path)

    test_dataset = dataset
    testViz(model, test_dataset, "train-history", num_example=10)


def testModelKITTIReg(model_path):
    model_device = torch.device("cuda")   
    data_device = device = torch.device("cpu")       

    rgb_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize( (200, 640) ),     
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    label_preprocess = transforms.Compose([#transforms.ToTensor(),
                                           transforms.GaussianBlur(15, sigma=3.0),
                                           transforms.Resize( (200, 640) )])


    test_dataset = KITTI_DEP(KITTI_TEST_RGB_PATHS, KITTI_TEST_LABEL_PATHS, transform=rgb_preprocess, target_transform=label_preprocess, device=data_device, original=True)

    model = torch.load(model_path)

    testViz(model, test_dataset, "train-history", num_example=10)    


def testModelKITTISeg(model_path):
    model_device = torch.device("cuda")   
    data_device = device = torch.device("cpu")       

    test_dataset = KITTI_SEM(KITTI_SEM_TEST_RGB_PATHS, KITTI_SEM_TEST_LABEL_PATHS, device=data_device, original=True)

    model = torch.load(model_path)

    testVizSeg(model, test_dataset, "train-history", num_example=10)    

    
if __name__ == '__main__':
    # initTrainKITTIDual()
    # initTrainKITTISeg()
    # initTrainKITTIReg()
    # initTrain()
    # modelSummary()
    # testModelKITTISeg(os.path.join("train-history", "trained_model49.pth"))
    testModelKITTIReg(os.path.join("train-history", "trained_model49.pth"))
    # testModelSeg(os.path.join("train-history", "trained_model199.pth"))