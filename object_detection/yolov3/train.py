#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 22:53:39 2021

@author: jessica
"""

import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from skimage.transform import resize
import numpy as np

import torch
from torchvision import datasets, transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable

from load_kitti import YoloData
from yolov3 import Darknet
from utils import convert_to_ori_cord, visualzie_detection

CFG_FILE = 'yolov3.cfg'
WEIGHTS_FILE = 'yolov3.weights'

DATA_PATH = '../data/kitti/'
TRAIN_DATA = DATA_PATH + 'data_object_image_2/training/image_2/'
TEST_DATA = DATA_PATH + 'data_object_image_2/testing/image_2/'
TRAIN_LABEL = DATA_PATH + 'data_object_label_2/training/label_2/'
YOLO_LABEL = DATA_PATH + 'data_object_label_2/training/yolo_label/'


train_batch_size = 1
test_batch_size = 4
num_epochs = 10
yolo_size = 416


class2idx = {'Car': 0, 'Van': 1, 'Truck': 2, 'Pedestrian': 3, 'Misc': 4, 'Cyclist': 5}
idx2class = {i:c for c, i in class2idx.items()}

if __name__ == '__main__':
    model = Darknet(CFG_FILE)
    model.load_weights(WEIGHTS_FILE)
    
    model.eval()
    

    train_dataloader = torch.utils.data.DataLoader(
        YoloData(TRAIN_DATA, YOLO_LABEL, yolo_size), batch_size=train_batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
 

    test_dataloader = torch.utils.data.DataLoader(YoloData(TEST_DATA, YOLO_LABEL, yolo_size), batch_size=test_batch_size, shuffle=False) 
    
    for batch_i, (nrow, ori_shape, ori_img_files, imgs, targets) in enumerate(train_dataloader):
        #visualzie_detection(torch.moveaxis(imgs[0], 0, -1), targets[0], nrow[0])
        con_cord = convert_to_ori_cord(ori_shape, targets, yolo_size)
        #visualzie_detection(np.array(Image.open(ori_img_files[0])), con_cord[0], nrow[0])
        prediction = model(Variable(imgs).float())
        break