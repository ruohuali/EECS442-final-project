#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 22:17:53 2021

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
from convert_kitti_to_yolo import idx2class

# def cal_loss(pred, label):
    
  
def convert_to_ori_cord(ori_shape, pred_mat, yolo_size):
    '''
    transform the yolo size cord to the original cord
    return (batch_size*max_labels*5) where 5 = (label,x,y,w,h)
    '''
    img_h, img_w, _ = ori_shape
    x, y, w, h = pred_mat[:,:,1], pred_mat[:,:,2], pred_mat[:,:,3], pred_mat[:,:,4]
    # resize image
    magnify = (torch.max(img_h, img_w)/yolo_size).reshape(-1,1)
    pred_mag = torch.zeros(pred_mat.shape)
    pred_mag[:,:,0] = pred_mat[:,:,0]
    pred_mag[:,:,1] = x*magnify
    pred_mag[:,:,2] = y*magnify
    pred_mag[:,:,3] = w*magnify
    pred_mag[:,:,4] = h*magnify
    
    # remove padding
    diff = abs(img_w-img_h)
    pad = (diff//2).reshape(-1,1)
    pred_mag[:,:,2][img_h<img_w] -= pad[img_h<img_w]
    pred_mag[:,:,1][img_h>=img_w] -= pad[img_h>=img_w]
    return pred_mag

def visualzie_detection(img, labels, num_labels):
    '''
    visualize one single image
    image size: (h,w,c)
    the input size for labels should be (max_labels*5) with 5 = (label,x,y,w,h)
    '''
    fig, ax = plt.subplots()
    ax.imshow(img)
    for i in range(num_labels):
        x, y, w, h = labels[i, 1], labels[i, 2], labels[i, 3], labels[i, 4]
        rect = patches.Rectangle((x-w/2, y-h/2), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x+w/2.4,y-h/2,idx2class[int(labels[i,0])],verticalalignment='top', backgroundcolor='black',
                color='white',fontsize=7)
    plt.show()

