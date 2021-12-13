#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 18:12:15 2021

@author: jessica
"""

import os

from PIL import Image
from skimage.transform import resize
import numpy as np

import torch

MAX_LABEL = 22

class YoloData():
    def __init__(self, img_path, label_path, size=416):
        self.images =  [img_path+i for i in os.listdir(img_path)]
        if label_path == None:
            self.labels = None
        else:
            self.labels = [label_path+i.replace('png','txt') for i in os.listdir(img_path)]
        self.shape = (size, size)
        self.max_label = MAX_LABEL

    def __getitem__(self, index):
        
        # load each image and add padding to make it of size 416*416
        img = np.array(Image.open(self.images[index]))
        ori_shape = img.shape
        h, w, c = img.shape
        
        diff = abs(w-h)
        pad = (diff//2, diff-diff//2)
        if h < w:
            img = np.pad(img, (pad, (0,0), (0,0)), 'constant', constant_values=0)
        else:
            img = np.pad(img, ((0,0), pad, (0,0)), 'constant', constant_values=0)
        pad_h, pad_w, pad_c = img.shape
        
        # check size
        assert(pad_h==pad_w)
        img = resize(img, self.shape)
        img = np.moveaxis(img, -1, 0)
        train_img = torch.from_numpy(img)  
        
        
        if self.labels == None :
            return ori_shape, self.images[index], train_img
        
        
        # adjust labels for the resized image
        with open(self.labels[index], 'rt') as f:
            label = np.loadtxt(f, ndmin=2) 
            nrow = label.shape[0]
            xtl, ytl, xbr, ybr = label[:, 1], label[:, 2], label[:, 3], label[:, 4]
            if pad_h:
                ytl += pad[0]
                ybr += pad[0]
            else:
                xtl += pad[0]
                xbr += pad[0]
            xratio = 1/pad_w
            yratio = 1/pad_h
            midx = ((xbr+xtl)/2)*xratio
            midy = ((ybr+ytl)/2)*yratio
            
            yolo_w = abs(xbr-xtl)*xratio
            yolo_h = abs(ybr-ytl)*yratio
            labels = np.full(shape=(MAX_LABEL, 5), fill_value=0, dtype=np.float32)
            
            labels[:nrow, 0] = label[:,0]
            labels[:nrow, 1] = midx
            labels[:nrow, 2] = midy
            labels[:nrow, 3] = yolo_w
            labels[:nrow, 4] = yolo_h

        return nrow, ori_shape, self.images[index], train_img, labels
    
    
    def __len__(self):
        return len(self.images)