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
    

def tlbr_to_xywh(tlbr_cord):
    xtl = tlbr_cord[...,0]
    ytl = tlbr_cord[...,1]
    xbr = tlbr_cord[...,2]
    ybr = tlbr_cord[...,3]
    xywh_cord = tlbr_cord.clone()
    xywh_cord[..., 0] = (xtl + xbr) / 2
    xywh_cord[..., 1] = (ytl + ybr) / 2
    xywh_cord[..., 2] = xbr - xtl
    xywh_cord[..., 3] = ybr - ytl
    return xywh_cord


def xywh_to_tlbr(xywh_cord):
    x = xywh_cord[...,0]
    y = xywh_cord[...,1]
    w = xywh_cord[...,2]
    h = xywh_cord[...,3]
    tlbr_cord = xywh_cord.clone()
    tlbr_cord[..., 0] = x - w / 2
    tlbr_cord[..., 1] = y - h / 2
    tlbr_cord[..., 2] = x + w / 2
    tlbr_cord[..., 3] = y + h / 2
    return tlbr_cord


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


def bbox_iou(box1, box2, tlbr=True):
    '''
    input:
        box1: (..., 4)
        box2: (..., 4)
        tlbr: whether the cord is in the form of  (xtl, ytl, xbr, ybr)
    output:
        the IoU of box1 and box2
    *all in the cord of (xtl, ytl, xbr, ybr)
    '''
    if not tlbr:
        p = xywh_to_tlbr(box1)
        bb = xywh_to_tlbr(box2)
    else:
        p = box1
        bb = box2
    xtl = p[...,0]
    ytl = p[...,1]
    xbr = p[...,2]
    ybr = p[...,3]
    x_tl = bb[...,0]
    y_tl = bb[...,1]
    x_br = bb[...,2]
    y_br = bb[...,3]
      
    inter_xtl = torch.max(xtl, x_tl)
    inter_ytl = torch.max(ytl, y_tl)
    inter_xbr = torch.min(xbr,x_br)
    inter_ybr = torch.min(ybr, y_br)
    zero = torch.zeros(inter_xtl.shape)
    intersection = torch.max(zero, inter_xbr-inter_xtl) * torch.max(zero, inter_ybr-inter_ytl)
    union = abs((x_br-x_tl)*(y_tl-y_br))+abs((xbr-xtl)*(ytl-ybr))-intersection
      
    iou = intersection/union
    return iou


def nms(boxes, scores, tlbr=True, iou_threshold=0.5, topk=None):
  """
  Non-maximum suppression removes overlapping bounding boxes.

  Inputs:
  - boxes: top-left and bottom-right coordinate values of the bounding boxes
    to perform NMS on, of shape Nx4
  - scores: scores for each one of the boxes, of shape N
  - iou_threshold: discards all overlapping boxes with IoU > iou_threshold; float
  - topk: If this is not None, then return only the topk highest-scoring boxes.
    Otherwise if this is None, then return all boxes that pass NMS.

  Outputs:
  - keep: torch.long tensor with the indices of the elements that have been
    kept by NMS, sorted in decreasing order of scores; of shape [num_kept_boxes]
  """

  if (not boxes.numel()) or (not scores.numel()):
    return torch.zeros(0, dtype=torch.long)

  keep = None
  sort = torch.argsort(scores, descending=True)
  for i in sort:
    i = torch.unsqueeze(i, 0)
    if keep==None: keep=torch.Tensor([i[0].item()]).long()
    prev = torch.index_select(boxes, 0, keep)
    iou = bbox_iou(prev, torch.index_select(boxes, 0, i), tlbr)
    print(iou)
    if torch.sum(iou > iou_threshold) > 0: continue
    keep = torch.cat((keep, i))
    if topk!=None and keep.size(0) >= topk: break

  return keep


def predict(x, anchors, num_classes, img_dim):
    A = len(anchors)
    B = x.size(0)
    G = x.size(2)
    stride = img_dim / G

    FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor

    prediction = x.view(B, A, 5 + num_classes, G, G).permute(0, 1, 3, 4, 2).contiguous()
    
    prediction[...,0] = torch.sigmoid(prediction[..., 0])
    prediction[...,1] = torch.sigmoid(prediction[..., 1])
    prediction[..., 4] = torch.sigmoid(prediction[..., 4])
    prediction[..., 5:] = torch.sigmoid(prediction[..., 5:])

    # Calculate offsets for each grid
    x = np.arange(G)
    y = np.arange(G)
    grid_x, grid_y = np.meshgrid(x, y)
    grid_x = torch.FloatTensor(grid_x).view(1,1,G,G)
    grid_y = torch.FloatTensor(grid_y).view(1,1,G,G)
    
    anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in anchors])
    anchors_w = anchors[...,0].view(1,A,1,1)
    anchors_h = anchors[...,1].view(1,A,1,1)
    
    # Add offset and scale with anchors
    prediction[..., 0] = prediction[..., 0] + grid_x
    prediction[..., 1] = prediction[..., 1] + grid_y
    prediction[..., 2] = torch.exp(prediction[..., 2]) * anchors_w
    prediction[..., 3] = torch.exp(prediction[..., 3]) * anchors_h
    prediction[..., :4] = prediction[..., :4] * stride
    prediction = prediction.view(B, A*G*G, -1)
    
    return prediction


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

