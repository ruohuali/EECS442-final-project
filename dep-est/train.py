import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import cv2
import os
import time
from copy import deepcopy
import gc

from UNet import *
from utils import *


def depMapGradMask(label):
    mask = torch.ones_like(label)
    mask[label == 0] = 0
    return mask

def checkMem():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved  
    a /= 1024 * 1024
    return a

def train(model, dataloader, regression, num_epoch=400, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)    
    loss_fn = nn.CrossEntropyLoss()
    best_record = np.inf
    hist = []
    for epoch in range(num_epoch):
        running_loss = 0        
        tic = time.time()
        for batch_idx, data in enumerate(dataloader):
            imgs = data['rgb'].to(device)
            labels = data['label'].to(device)
            
            if regression:
                labels = labels.squeeze(1).to(torch.int64)

            print("batch", batch_idx, "/", len(dataloader), end='\r')
            pred = model(imgs)
            loss = loss_fn(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()        
            
            running_loss += loss.item() / len(dataloader)
            
        toc = time.time()
        hist.append(running_loss)
            
        if epoch % 1 == 0:
            print("epoch", epoch)
            print("epoch", epoch, "takes", toc-tic)
            print("running loss", running_loss)
            print("-"*50)

        if best_record > running_loss:
            print("best record", best_record)
            best_record = running_loss
            best_model = deepcopy(model)
        
        if epoch % 50 == 47:
            model_path = "trained_model_dep"+str(epoch)+".pth"
            torch.save(best_model, model_path)
            # files.download(model_path)
          
            plt.figure()
            plt.plot(hist)
            plt.savefig(str(epoch)+".jpg")
            # testViz(best_model)
            
    return best_model
        
def testViz(model, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), num_example=5):
    test_img_names = sorted(os.listdir(TEST_RGB_PATH))
    test_label_names = sorted(os.listdir(TEST_DEP_PATH))
    for i in range(num_example):
        img_path = os.path.join(TEST_RGB_PATH, test_img_names[i*40])
        img_t = read2Tensor(img_path)
        img = cv2.imread(img_path)

        label_path = os.path.join(TEST_DEP_PATH, test_label_names[i*40])
        label = cv2.imread(label_path)
        label = label[:,:,0]
        label = cv2.resize(label, (480, 120))

        tic = time.time()
        pred = model(img_t)
        pred = pred2Img(pred)
        toc = time.time()
        print("inference takes", toc-tic)

        plt.figure(figsize=(20,10))
        plt.imshow(img) 
        plt.title("img"+str(i)) 
        plt.figure(figsize=(20,10))
        plt.imshow(label) 
        plt.title("label"+str(i))         
        plt.figure(figsize=(20,10))
        plt.imshow(pred)
        plt.title("pred"+str(i)) 

        if i+1 >= num_example:
            break


if __name__ == "__main__":
    model_device = torch.device("cuda")   
    data_device = device = torch.device("cpu")       

    dataset = KITTI_DEP(TRAIN_RGB_PATHS, TRAIN_DEP_PATHS, device=data_device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    unet = UNet(3, 85).to(model_device)
    model = train(unet, dataloader, True, device=model_device)