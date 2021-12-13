#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 02:10:33 2021

@author: jessica
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import bbox_iou


def criteria(preds, x, y, w, h, anchors, targets, num_classes, img_dim, ignore_thres):
    
    FloatTensor = torch.cuda.FloatTensor if preds.is_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if preds.is_cuda else torch.LongTensor
        
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    ce_loss = nn.CrossEntropyLoss()
     
    if x.is_cuda:
        mse_loss = mse_loss.cuda()
        bce_loss = bce_loss.cuda()
        ce_loss = ce_loss.cuda()
    
    pred_conf = preds[...,4]
    pred_cls = preds[...,5:]
    B = targets.size(0)
    A = len(anchors)
    C = num_classes
    G = x.shape[2]
    mask = torch.zeros(B, A, G, G)
    conf_mask = torch.ones(B, A, G, G)
    tx = torch.zeros(B, A, G, G)
    ty = torch.zeros(B, A, G, G)
    tw = torch.zeros(B, A, G, G)
    th = torch.zeros(B, A, G, G)
    tconf = torch.ByteTensor(B, A, G, G).fill_(0)
    tcls = torch.ByteTensor(B, A, G, G, C).fill_(0)
    
    
    for b in range(B):
        for t in range(targets.shape[1]):
            if targets[b, t].sum() == 0:
                continue
            # Convert to position relative to box
            gx = targets[b, t, 1] * G
            gy = targets[b, t, 2] * G
            gw = targets[b, t, 3] * G
            gh = targets[b, t, 4] * G
            # Get grid box indices
            gi = int(gx)
            gj = int(gy)
            # Get shape of gt box
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            # Get shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
            # Calculate iou between gt and anchor shapes
            anch_ious = bbox_iou(gt_box, anchor_shapes)
            # Where the overlap is larger than threshold set mask to zero
            conf_mask[b, anch_ious > ignore_thres, gj, gi] = 0
            # Find the best matching anchor box
            best_n = np.argmax(anch_ious)
            # Get ground truth box
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)
            # Masks
            mask[b, best_n, gj, gi] = 1
            conf_mask[b, best_n, gj, gi] = 1
            # Coordinates
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            # Width and height
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0])
            th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1])
            # One-hot encoding of label
            target_label = int(targets[b, t, 0])
            tcls[b, best_n, gj, gi, target_label] = 1
            tconf[b, best_n, gj, gi] = 1

    # Handle masks
    mask = Variable(mask.type(LongTensor))
    conf_mask = Variable(conf_mask.type(LongTensor))

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
    loss_cls = (1 / B) * ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))
    loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
    loss = Variable(loss, requires_grad = True)
        
    return loss
    
    
    
    
    
    