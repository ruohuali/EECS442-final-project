#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 15:55:03 2021

@author: jessica
"""

import os

N = 10

DATA_PATH = '../data/kitti/'
TRAIN_DATA = DATA_PATH + 'data_object_image_2/training/image_2/'
TEST_DATA = DATA_PATH + 'data_object_image_2/testing/image_2/'
TRAIN_LABEL = DATA_PATH + 'data_object_label_2/training/label_2/'
YOLO_LABEL = DATA_PATH + 'data_object_label_2/training/yolo_label/'

class2idx = {'Car': 0, 'Van': 1, 'Truck': 2, 'Pedestrian': 3, 'Misc': 4, 'Cyclist': 5}
idx2class = {i:c for c, i in class2idx.items()}

def kitti2yolo(bbox, img_size):
    x = (bbox[0] + bbox[1]) / 2.0
    y = (bbox[2] + bbox[3]) / 2.0
    w = bbox[1] - bbox[0]
    h = bbox[3] - bbox[2]
    return [x/img_size[0], y/img_size[1], w/img_size[0], h/img_size[1]]

def change_to_yolo_labels(label_dir, image_dir):
    files = os.listdir(label_dir)

    max_labels = 0
    for i, file in enumerate(files):
        with open(label_dir+file, 'rt') as f:
            lines = f.readlines()
            
            yolo_text = open(YOLO_LABEL+files[i],"a")
            num_labels = 0
            
            for line in lines:
                data = line.split(' ')
                if data[0] in class2idx:
                    classidx= class2idx[data[0]]
                else:
                    continue
                yolo_bbox = [classidx,] + data[4:8]
                yolo_text.write(' '.join(str(s) for s in yolo_bbox))
                yolo_text.write('\n')
                num_labels += 1
                
            yolo_text.close()
            max_labels = max(max_labels, num_labels)
    print(max_labels)
    # output: 22
                
if __name__ == '__main__':
    change_to_yolo_labels(TRAIN_LABEL, TRAIN_DATA)










