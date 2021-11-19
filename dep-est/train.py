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
from pdb import set_trace

from models import *
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
    plt.plot(loss_hist)
    plt.savefig(plot_path)
    print(f"plotted as {plot_path}")
    print("="*30)
    print()

def train(model, dataloader, regression, num_epoch=400, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.train()

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)    

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

            print("batch", batch_idx, "/", len(dataloader), end='       \r')
            pred = model(imgs)
            loss = 0.2 * l1_loss(pred, labels) + 0.85 * (-ssim(pred, labels))

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
            
        if epoch % 20 == 19:
            manualCheckpoint(epoch, hist, best_model, "trained_model"+str(epoch), "train-history")
            
    return best_model
        
def testViz(model, dataset, save_dir, device=torch.device("cpu"), num_example=5):
    model = model.to(device)
    model.eval()

    for i in range(num_example):
        idx = len(dataset) // (num_example+1) * i

        data = dataset[idx]
        img_t = data['rgb'].unsqueeze(0).to(device)
        img = data['original_rgb']
        label_t = data['label']
        label = data['original_label']

        tic = time.time()
        pred = model(img_t)
        pred = regPred2Img(pred)
        toc = time.time()
        print("inference takes", toc-tic)

        displayInference(data, pred, save_dir, i, backend="cmap")
        print(np.unique(pred.numpy()))

if __name__ == "__main__":
    pass