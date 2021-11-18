import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import time
from copy import deepcopy
import matplotlib.pyplot as plt
import os

from PATH import *


def depth2Heatmap(depth_map, min_display=0, max_display=100):
    '''@param depth_map ~ (H x W)'''
    r = (np.max(depth_map) - np.min(depth_map)) / 10
    heatmap = np.zeros((*depth_map.shape, 3))
    for i in range(depth_map.shape[0]):
        for j in range(depth_map.shape[1]):
            if not min_display < depth_map[i,j] < max_display:
                continue
            if depth_map[i,j] < r:
                heatmap[i,j,2] = 255 - depth_map[i,j] / r * 255    
                heatmap[i,j,0] = depth_map[i,j] / r * 255           
            elif r < depth_map[i,j] < 5*r:
                heatmap[i,j,0] = 255 - (depth_map[i,j]-r) / (4*r) * 255               
                heatmap[i,j,1] = (depth_map[i,j]-r) / (4*r) * 255  
            else:
                heatmap[i,j,1] = 255 - (depth_map[i,j]-4*r) / (6*r) * 255                                        
                heatmap[i,j,2] = (depth_map[i,j]-4*r) / (6*r) * 255  
    heatmap[:,:,0] *= 15
    return heatmap.astype(np.int64)


def drawHeatmap(depth_map, min_display=0, max_display=100):
    h = depth2Heatmap(depth_map, min_display=0, max_display=100)
    
    plt.figure(figsize=(20,10))
    plt.imshow(h)
    plt.savefig("heat.png")
    x = np.arange(70).reshape(1, -1)
    x = np.vstack((x,x,x))
    hb = depth2Heatmap(x)
    plt.figure()
    plt.imshow(hb)    

def read2Tensor(img_path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), transform=None):
    '''image path -> unet input (1 x 3 x H x W)'''

    img_t = read_image(img_path).to(torch.float32).unsqueeze(0).to(device)
    img_t = transform(img_t)  
    return img_t

def readLabel2Tensor(label_path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    '''image path -> loss input (1 x H x W)'''

    label_t = read_image(label_path).to(torch.float32).unsqueeze(0).to(device)
    transform = transforms.Compose([transforms.Resize( (120, 480) )])
    label_t = transform(label_t)  
    return label_t    

def pred2Img(pred):
    '''singleton unet output (1 x 1 x H x W) -> plottable grayscale image (H x W)'''
    pred = pred.squeeze(0)
    pred = pred.squeeze(0)
    pred = pred.to("cpu").to(torch.uint8)
    return pred

if __name__ == "__main__":
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name(torch.cuda.device_count()-1))

    gpu_device = torch.device("cuda")
    cpu_device = torch.device("cpu")
    dataset = KITTI_DEP(TRAIN_RGB_PATHS, TRAIN_DEP_PATHS, device=cpu_device, qmark=True, original=False)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    # dataset.example(519)
    print('len', len(dataset))
    tic = time.time()
    for i, data in enumerate(dataloader):
        print("data rgb", data['rgb'].shape)
        rgb = data['rgb'].to(gpu_device)
        label = data['label'].to(gpu_device)
        if i > 10:
            break
    print("time", time.time() - tic)    