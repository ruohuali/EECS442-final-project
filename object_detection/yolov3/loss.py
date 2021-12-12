#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 02:10:33 2021

@author: jessica
"""

import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from skimage.transform import resize
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision import datasets
from torch.utils.data import DataLoader



def build_targets(
    pred_boxes, pred_conf, pred_cls, target, anchors, num_anchors, num_classes, grid_size, ignore_thres, img_dim
):
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    nG = grid_size
    mask = torch.zeros(nB, nA, nG, nG)
    conf_mask = torch.ones(nB, nA, nG, nG)
    tx = torch.zeros(nB, nA, nG, nG)
    ty = torch.zeros(nB, nA, nG, nG)
    tw = torch.zeros(nB, nA, nG, nG)
    th = torch.zeros(nB, nA, nG, nG)
    tconf = torch.ByteTensor(nB, nA, nG, nG).fill_(0)
    tcls = torch.ByteTensor(nB, nA, nG, nG, nC).fill_(0)

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(target.shape[1]):
            if target[b, t].sum() == 0:
                continue
            nGT += 1
            # Convert to position relative to box
            gx = target[b, t, 1] * nG
            gy = target[b, t, 2] * nG
            gw = target[b, t, 3] * nG
            gh = target[b, t, 4] * nG
            # Get grid box indices
            gi = int(gx)
            gj = int(gy)
            # Get shape of gt box
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            # Get shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
            # Calculate iou between gt and anchor shapes
            anch_ious = bbox_iou(gt_box, anchor_shapes)
            # Where the overlap is larger than threshold set mask to zero (ignore)
            conf_mask[b, anch_ious > ignore_thres, gj, gi] = 0
            # Find the best matching anchor box
            best_n = np.argmax(anch_ious)
            # Get ground truth box
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)
            # Get the best prediction
            pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)
            # Masks
            mask[b, best_n, gj, gi] = 1
            conf_mask[b, best_n, gj, gi] = 1
            # Coordinates
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            # Width and height
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)
            # One-hot encoding of label
            target_label = int(target[b, t, 0])
            tcls[b, best_n, gj, gi, target_label] = 1
            tconf[b, best_n, gj, gi] = 1

            # Calculate iou between ground truth and best matching prediction
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
            pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])
            score = pred_conf[b, best_n, gj, gi]
            if iou > 0.5 and pred_label == target_label and score > 0.5:
                nCorrect += 1

    return nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls


def loss_function(
        preds, 
        targets,
        G, 
        A,
        num_classes,
        ignore_thres,
        img_dim,
        is_cuda
        ):
    FloatTensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if is_cuda else torch.LongTensor
    ByteTensor = torch.cuda.ByteTensor if is_cuda else torch.ByteTensor
    
    mse_loss = nn.MSELoss(size_average=True)  # Coordinate loss
    bce_loss = nn.BCELoss(size_average=True)  # Confidence loss
    ce_loss = nn.CrossEntropyLoss()  # Class loss
    
    nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(
                pred_boxes=pred_boxes.cpu().data,
                pred_conf=pred_conf.cpu().data,
                pred_cls=pred_cls.cpu().data,
                target=targets.cpu().data,
                anchors=scaled_anchors.cpu().data,
                num_anchors=nA,
                num_classes=num_classes,
                grid_size=nG,
                ignore_thres=ignore_thres,
                img_dim=img_dim,
            )
    
    # Handle masks
    mask = Variable(mask.type(ByteTensor))
    conf_mask = Variable(conf_mask.type(ByteTensor))

    # Handle target variables
    tx = Variable(tx.type(FloatTensor), requires_grad=False)
    ty = Variable(ty.type(FloatTensor), requires_grad=False)
    tw = Variable(tw.type(FloatTensor), requires_grad=False)
    th = Variable(th.type(FloatTensor), requires_grad=False)
    tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
    tcls = Variable(tcls.type(LongTensor), requires_grad=False)
    
    # Get conf mask where gt and where there is no gt
    conf_mask_true = mask
    conf_mask_false = conf_mask - mask

    # Mask outputs to ignore non-existing objects
    loss_x = mse_loss(x[mask], tx[mask])
    loss_y = mse_loss(y[mask], ty[mask])
    loss_w = mse_loss(w[mask], tw[mask])
    loss_h = mse_loss(h[mask], th[mask])
    loss_conf = bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false]) + bce_loss(
        pred_conf[conf_mask_true], tconf[conf_mask_true]
    )
    loss_cls = (1 / nB) * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))
    loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
    