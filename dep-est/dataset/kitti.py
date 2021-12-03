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
from PIL import Image
from utils import *

'''
@todo customized kitti transform
      more augmentation
      quantization
'''
class KITTI_SEM(Dataset):
    '''
    @note assume that label image's name and corresponding rgb image's name are the same
    just in different folders. e.g. labels/1.jpg <---> rgbs/1.jpg
    '''
    rgb_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize( (200, 640) ),     
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    label_preprocess = transforms.Compose([transforms.Resize( (200, 640) )])

    def __init__(self, rgb_dir_paths, label_dir_paths, transform=None, target_transform=None, 
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), original=False):
        self.rgb_dir_paths = rgb_dir_paths
        self.label_dir_paths = label_dir_paths
        
        self.data_pair_paths = []
        for rgb_dir_path, label_dir_path in zip(rgb_dir_paths, label_dir_paths):
            rgb_data_names = sorted(os.listdir(rgb_dir_path))
            label_data_names = sorted(os.listdir(label_dir_path))
            for rgb_data_name, label_data_name  in zip(rgb_data_names, label_data_names):
                img_path = os.path.join(rgb_dir_path, rgb_data_name)
                label_path = os.path.join(label_dir_path, label_data_name)  
                self.data_pair_paths.append((img_path, label_path))
                
        self.transform = transform if transform != None else self.rgb_preprocess
        self.target_transform = target_transform if target_transform != None else self.label_preprocess
        
        self.device = device
        self.qmark = False
        self.original = original

    def __len__(self):
        return len(self.data_pair_paths)

    def __getitem__(self, idx):
        img_path, label_path = self.data_pair_paths[idx]
   
        img = Image.open(img_path)
        label = read_image(label_path, ImageReadMode.GRAY) 
        
        img = self.transform(img)
        label = self.target_transform(label)

        img = img.to(self.device)  
        label = label.squeeze().to(torch.long).to(self.device)      

        if self.original:
            original_img = cv2.imread(img_path)
            original_img = cv2.resize(original_img, (1241, 376))    
            original_label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            original_label = cv2.resize(original_label, (1241, 376))    
        else:
            original_img = torch.empty(1)
            original_label = torch.empty(1)                     
            
        data = {}
        data['rgb'] = img
        data['label'] = label
        data['original_rgb'] = original_img
        data['original_label'] = original_label
        data['rgb_path'] = img_path
        data['label_path'] = label_path        
            
        return data
    
    def example(self, idx=10):
        data = self.__getitem__(idx)
        image = data['rgb']
        label = data['label']
        original_image = data['original_rgb']
        original_label = data['original_label']
        
        image = image.permute(1, 2, 0).to("cpu").to(torch.long)
        label = label.to("cpu").numpy()
        
        plt.figure(figsize=(20, 10))
        plt.imshow(image)
        plt.savefig(f"example image {idx}.png")
     
        plt.figure(figsize=(20, 10))
        plt.imshow(label)    
        plt.savefig(f"example original image {idx}.png")    

        plt.figure(figsize=(20, 10))
        plt.imshow(original_image)    
        plt.savefig(f"example original image {idx}.png")             

        plt.figure(figsize=(20, 10))
        plt.imshow(original_label)    
        plt.savefig(f"example original label {idx}.png")  


class KITTI_DEP(Dataset):
    '''
    @note assume that label image's name and corresponding rgb image's name are the same
    just in different folders. e.g. labels/1.jpg <---> rgbs/1.jpg
    '''
    rgb_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize( (200, 640) ),     
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    label_preprocess = transforms.Compose([#transforms.ToTensor(),
                                           transforms.GaussianBlur(15, sigma=4.0),
                                           transforms.Resize( (200, 640) )])

    def __init__(self, rgb_dir_paths, label_dir_paths, transform=None, target_transform=None, 
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), original=False):
        self.rgb_dir_paths = rgb_dir_paths
        self.label_dir_paths = label_dir_paths
        
        self.data_pair_paths = []
        for rgb_dir_path, label_dir_path in zip(rgb_dir_paths, label_dir_paths):
            rgb_data_names = sorted(os.listdir(rgb_dir_path))
            label_data_names = sorted(os.listdir(label_dir_path))
            for rgb_data_name, label_data_name  in zip(rgb_data_names, label_data_names):
                img_path = os.path.join(rgb_dir_path, rgb_data_name)
                label_path = os.path.join(label_dir_path, label_data_name)  
                self.data_pair_paths.append((img_path, label_path))
                
        self.transform = transform if transform != None else self.rgb_preprocess
        self.target_transform = target_transform if target_transform != None else self.label_preprocess
        
        self.device = device
        self.qmark = False
        self.original = original

    def __len__(self):
        return len(self.data_pair_paths)

    def __getitem__(self, idx):
        img_path, label_path = self.data_pair_paths[idx]
   
        img = Image.open(img_path)
        label = cv2.imread(label_path)
        label = torch.tensor(label[:,:,0]).unsqueeze(0)     
        
        img = self.transform(img)
        label = self.target_transform(label)

        img = img.to(self.device)  
        label = label.to(self.device).to(torch.float32)        

        if self.original:
            original_img = cv2.imread(img_path)     
            original_label = cv2.imread(label_path)[:,:,0]        
        else:
            original_img = torch.empty(1)
            original_label = torch.empty(1)                     
            
        data = {}
        data['rgb'] = img
        data['label'] = label
        data['original_rgb'] = original_img
        data['original_label'] = original_label
        data['rgb_path'] = img_path
        data['label_path'] = label_path        
            
        return data
    
    def example(self, idx=10):
        data = self.__getitem__(idx)
        image = data['rgb']
        label = data['label']
        original_image = data['original_rgb']
        original_label = data['original_label']
        
        image = image.permute(1, 2, 0).to("cpu").to(torch.uint8)
        label = label.permute(1, 2, 0).to("cpu").to(torch.uint8).squeeze(2).numpy()
        
        plt.figure(figsize=(20, 10))
        plt.imshow(image)
        plt.savefig(f"example image {idx}.png")
     
        plot_depth_map(label, np.ones_like(label), ".")

        plt.figure(figsize=(20, 10))
        plt.imshow(original_image)    
        plt.savefig(f"example original image {idx}.png")             

        h = depth2Heatmap(original_label)
        plt.figure(figsize=(20, 10))
        plt.imshow(h)    
        plt.savefig(f"example original label {idx}.png")  
