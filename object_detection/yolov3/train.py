#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 22:53:39 2021

@author: jessica
"""

import torch
import numpy as np

from PIL import Image

from load_kitti import YoloData
from yolov3 import Darknet
from utils import convert_to_ori_cord, visualzie_detection, nms, interpret_result

CFG_FILE = 'yolov3.cfg'
WEIGHTS_FILE = 'yolov3.weights'
SAVE_WEIGHTS_PATH = 'save_weights'

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
    
    # Get dataloader
    dataloader = torch.utils.data.DataLoader(
        YoloData(TRAIN_DATA, YOLO_LABEL, height), batch_size=train_batch, shuffle=True)
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))    
    
    

    # use for test, get AP on valid test
    test_dataloader = torch.utils.data.DataLoader(YoloData(TEST_DATA, None, height), batch_size=test_batch, shuffle=False) 

    
    # training state
    print("start training")
    for i in range(num_epochs):
        model.train()
        print('start epoch ', i)
        for batch_i, (nrow, ori_shape, ori_img_files, imgs, targets) in enumerate(dataloader):
            #visualzie_detection(torch.moveaxis(imgs[0], 0, -1), targets[0], nrow[0])
            #con_cord = convert_to_ori_cord(ori_shape, targets, height)
            #visualzie_detection(np.array(Image.open(ori_img_files[0])), con_cord[0], nrow[0])
            optimizer.zero_grad()
            loss = model(imgs.float(), targets)
            print('loss: ', loss)
            loss.backward()
            optimizer.step()
            #keep = nms(output[0,:,:4], output[0,:,4], tlbr=False, topk=3)
            #print(output)
            #print(output.shape)
            #print(targets.shape)
            #print(convert_to_ori_cord(ori_shape, prediction, yolo_size))
        model.save_weights("%s/%d.weights" % (SAVE_WEIGHTS_PATH, i))
    
    '''
    # testing
    model.eval()
    for batch_i, (ori_shape, ori_img_files, imgs) in enumerate(test_dataloader):
        predictions = model(imgs.float())
        # visualize for each img in batch
        for b in range(predictions.shape[0]):
            keep = nms(predictions[b,:,:4], predictions[b,:,4], tlbr=False, topk=3)
            output_boxes = predictions[:,keep,:]
            output_boxes = interpret_result(output_boxes)
            print(output_boxes)
            visualzie_detection(np.array(Image.open(ori_img_files[0])),output_boxes[b], len(keep))
        break
    '''
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        