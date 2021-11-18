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
from loss import *
from PATH import *
from data import *


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

def manualCheckpoint(epoch, loss_hist, best_model, model_name, save_dir):
    print("="*30)
    print(f"epoch {epoch} loss {min(loss_hist)}")
    model_path = os.path.join(save_dir, model_name+".pth")
    torch.save(best_model, model_path)
    print(f"saved model to {model_path}")
    plot_path = os.path.join(save_dir, str(epoch)+".png")
    plt.figure()
    plt.plot(hist)
    plt.savefig(plot_path)
    print(f"plotted as {plot_path}")
    print("="*30)
    print()

def train(model, dataloader, regression, num_epoch=400, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)    
    
    l1_loss = nn.L1Loss()
    ssim = SSIM()

    best_record = np.inf
    hist = []
    for epoch in range(num_epoch):
        running_loss = 0        
        tic = time.time()
        for batch_idx, data in enumerate(dataloader):
            imgs = data['rgb'].to(device)
            labels = data['label'].to(device)
            
            if not regression:
                labels = labels.squeeze(1).to(torch.int64)

            print("batch", batch_idx, "/", len(dataloader), end='\r')
            pred = model(imgs)
            loss = 0.3 * l1(pred, labels) + 1 * (-ssim(pred, label))

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
            
        if epoch % 5 == 1:
            manualCheckpoint(epoch, hist, best_model, "trained_model"+str(epoch), "train-history")
            
    return best_model
        
def testViz(model, transform, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), num_example=5):
    test_img_names = sorted(os.listdir(TEST_RGB_PATH))
    test_label_names = sorted(os.listdir(TEST_DEP_PATH))
    for i in range(num_example):
        idx = len(test_img_names) // (num_example+1) * i

        img_path = os.path.join(TEST_RGB_PATH, test_img_names[idx])
        img_t = read2Tensor(img_path, transform)
        img = cv2.imread(img_path)

        label_path = os.path.join(TEST_DEP_PATH, test_label_names[idx])
        label = cv2.imread(label_path)
        label = label[:,:,0]
        label = cv2.resize(label, (320, 320))

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

    transform = transforms.Compose([transforms.Resize( (320, 320) ),
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    target_transform = transforms.Compose([transforms.Resize( (320, 320) )])

    dataset = DIODE(TRAIN_PATHS, transform=transform, target_transform=target_transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    unet = RegUNet(3, 32).to(model_device)
    model = train(unet, dataloader, True, device=model_device)
    testViz(model, transform)