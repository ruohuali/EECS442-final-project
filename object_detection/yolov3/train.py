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

from load_kitti import YoloData
from yolov3 import Darknet
from utils import convert_to_ori_cord, visualzie_detection, nms

CFG_FILE = 'yolov3.cfg'
WEIGHTS_FILE = 'yolov3.weights'

DATA_PATH = '../data/kitti/'
TRAIN_DATA = DATA_PATH + 'data_object_image_2/training/image_2/'
TEST_DATA = DATA_PATH + 'data_object_image_2/testing/image_2/'
TRAIN_LABEL = DATA_PATH + 'data_object_label_2/training/label_2/'
YOLO_LABEL = DATA_PATH + 'data_object_label_2/training/yolo_label/'


class2idx = {'Car': 0, 'Van': 1, 'Truck': 2, 'Pedestrian': 3, 'Misc': 4, 'Cyclist': 5}
idx2class = {i:c for c, i in class2idx.items()}

if __name__ == '__main__':
    # initialize yolov3
    model = Darknet(CFG_FILE)
    model.load_weights(WEIGHTS_FILE)
    
    # initialize args
    args = model.hyperparams
    num_epochs = 2
    train_batch  = int(args['train_batch'])
    train_subdivisions = int(args['train_subdivisions'])
    width=int(args['width'])
    height=int(args['height'])
    channels=int(args['channels'])
    momentum=float(args['momentum'])
    decay=float(args['decay'])
    angle=float(args['angle'])
    saturation = float(args['saturation'])
    exposure = float(args['exposure'])
    hue=float(args['hue'])

    learning_rate=float(args['learning_rate'])
    burn_in=int(args['burn_in'])
    max_batches = int(args['max_batches'])
    policy=args['policy']
    steps=list(map(int, args['steps'].split(',')))
    scales=list(map(float, args['scales'].split(',')))
    
    test_batch=int(args['test_batch'])
    test_subdivisions=int(args['test_subdivisions'])

    
    model.eval()
    
    # Get dataloader
    dataloader = torch.utils.data.DataLoader(
        YoloData(TRAIN_DATA, YOLO_LABEL, height), batch_size=train_batch, shuffle=True)
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    '''
    losses_x = losses_y = losses_w = losses_h = losses_conf = losses_cls = losses_recall = losses_precision = batch_loss= 0.0
    accumulated_batches = 4
    best_mAP = 0.0
    '''
    
    
    print("start training")
    '''
    loss_data_file = open('loss data.txt','w+')
    test_data_file = open('test_data.txt','w+')
    '''
    # use for test, get AP on valid test
    test_dataloader = torch.utils.data.DataLoader(YoloData(TEST_DATA, YOLO_LABEL, height), batch_size=test_batch, shuffle=False) 
    
    '''
    for epoch in range(num_epochs):
        # losses_x = losses_y = losses_w = losses_h = losses_conf = losses_cls = losses_recall = losses_precision = batch_loss= 0.0
        
        # Freeze darknet53.conv.74 layers for first some epochs
        if freeze_backbone:
            if epoch < 20:
                for i, (name, p) in enumerate(model.named_parameters()):
                    if int(name.split('.')[1]) < 75:  # if layer < 75
                        p.requires_grad = False
            elif epoch >= 20:
                for i, (name, p) in enumerate(model.named_parameters()):
                    if int(name.split('.')[1]) < 75:  # if layer < 75
                        p.requires_grad = True
                        
        optimizer.zero_grad()   
                       
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            imgs = Variable(imgs.type(Tensor))
            targets = Variable(targets.type(Tensor), requires_grad=False)
           
            loss = model(imgs, targets)
    
            loss.backward()
            '''   
    for batch_i, (nrow, ori_shape, ori_img_files, imgs, targets) in enumerate(dataloader):
        #visualzie_detection(torch.moveaxis(imgs[0], 0, -1), targets[0], nrow[0])
        #con_cord = convert_to_ori_cord(ori_shape, targets, yolo_size)
        #visualzie_detection(np.array(Image.open(ori_img_files[0])), con_cord[0], nrow[0])
        output = model(imgs.float())
        keep = nms(output[0,:,:4], output[0,:,4], tlbr=False, topk=10)
        print(keep)
        #print(output.shape)
        #print(targets.shape)
        #print(convert_to_ori_cord(ori_shape, prediction, yolo_size))
        
        break