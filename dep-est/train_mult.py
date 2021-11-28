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




def manualCheckpoint(epoch, loss_hist1, loss_hist2, best_model, model_name, save_dir):
    print("="*30)
    print(f"epoch {epoch} loss {min(loss_hist1)} {min(loss_hist2)}")
    model_path = os.path.join(save_dir, model_name+".pth")
    torch.save(best_model, model_path)
    print(f"saved model to {model_path}")
    plot_path = os.path.join(save_dir, str(epoch)+".png")
    plt.figure()
    plt.plot(loss_hist1, label="train")
    plt.plot(loss_hist2, label="val")
    plt.savefig(plot_path)
    plt.legend(loc="upper left")    
    print(f"plotted as {plot_path}")
    print("="*30)
    print()


def getLossReg(pred, labels, imgs):
    l2_loss = nn.MSELoss()
    ssim = SSIM(window_size=7)
    smooth_loss = SmoothnessLoss()

    loss = 1 * (1 - ssim(pred, labels)) + 0.5 * smooth_loss(pred, imgs) + 1 * l2_loss(pred, labels) 
    return loss


def getLossSeg(pred, labels, imgs):
    ce = nn.CrossEntropyLoss()

    loss = ce(pred, labels)
    return loss


def doEpochReg(dataloader, model, optimizer=None, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    running_loss = 0
    for batch_idx, data in enumerate(dataloader):
        imgs = data['rgb'].to(device)
        labels = data['label'].to(device)
        
        if optimizer != None:
            pred, _ = model(imgs)
            # loss = 1 * (1 - ssim(pred, labels)) + 1 * l2_loss(pred, labels) 
            loss = getLossReg(pred, labels, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()     
            print("train batch", batch_idx, "/", len(dataloader), end='       \r')

        else:
            with torch.no_grad():
                pred, _ = model(imgs)
                loss = getLossReg(pred, labels, imgs)
                print("val batch", batch_idx, "/", len(dataloader), end='       \r')

        running_loss += loss.item() / len(dataloader)

    return running_loss


def doEpochSeg(dataloader, model, optimizer=None, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    running_loss = 0
    for batch_idx, data in enumerate(dataloader):
        imgs = data['rgb'].to(device)
        labels = data['label'].to(device)
        
        if optimizer != None:
            _, pred = model(imgs)
            loss = getLossSeg(pred, labels, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()     
            print("train batch", batch_idx, "/", len(dataloader), end='       \r')

        else:
            with torch.no_grad():
                _, pred = model(imgs)
                loss = getLossSeg(pred, labels, imgs)
                print("val batch", batch_idx, "/", len(dataloader), end='       \r')

        running_loss += loss.item() / len(dataloader)

    return running_loss


def train(model, train_dataloader, val_dataloader, task, num_epoch=400, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.train()

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)    

    best_record = np.inf
    train_hist = []
    val_hist = []
    for epoch in range(num_epoch):
        tic = time.time()
        if task == "seg":
            running_train_loss = doEpochSeg(train_dataloader, model, optimizer=optimizer, device=device)
        elif task == "reg":
            running_train_loss = doEpochReg(train_dataloader, model, optimizer=optimizer, device=device)
        train_hist.append(running_train_loss)

        if task == "seg":
            running_val_loss = doEpochSeg(val_dataloader, model, optimizer=None, device=device)
        elif task == "reg":
            running_val_loss = doEpochReg(val_dataloader, model, optimizer=None, device=device)
        val_hist.append(running_val_loss)        
        toc = time.time()
            
        if epoch % 1 == 0:
            print("epoch", epoch)
            print("epoch", epoch, "takes", toc-tic)
            print("running train loss", running_train_loss)
            print("running val loss", running_val_loss)
            print("-"*50)
            if best_record > running_train_loss:
                print("best record", best_record)
                best_record = running_train_loss
                best_model = deepcopy(model)
            
        if epoch % 50 == 49:
            manualCheckpoint(epoch, train_hist, val_hist, best_model, "trained_model"+str(epoch), "train-history")
            
    return best_model


def trainMult(model, train_dep_dataloader, val_dep_dataloader, train_sem_dataloader, val_sem_dataloader, num_epoch=400, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.train()

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)    

    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()
    ssim = SSIM(window_size=7)
    smooth_loss = SmoothnessLoss()
    ce_loss = nn.CrossEntropyLoss()

    best_record = np.inf
    train_hist = []
    val_hist = []
    for epoch in range(num_epoch):
        running_train_loss = 0      
        running_val_loss = 0      
          
        tic = time.time()
        for batch_idx, data in enumerate(train_dataloader):
            imgs = data['rgb'].to(device)
            labels = data['label'].to(device)

            print("train batch", batch_idx, "/", len(train_dataloader), end='       \r')
            
            pred, _ = model(imgs)
            loss = 1 * (1 - ssim(pred, labels)) + 0.5 * smooth_loss(pred, imgs) + 1 * l2_loss(pred, labels) 
            # loss = 1 * (1 - ssim(pred, labels)) + 1 * l2_loss(pred, labels) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()        
            
            running_train_loss += loss.item() / len(train_dataloader)
        train_hist.append(running_train_loss)

        for batch_idx, data in enumerate(val_dataloader):
            with torch.no_grad():
                imgs = data['rgb'].to(device)
                labels = data['label'].to(device)

                print("val batch", batch_idx, "/", len(val_dataloader), end='       \r')
                pred, _ = model(imgs)
                loss1 = 1 * (1 - ssim(pred, labels)) + 0.1 * smooth_loss(pred, imgs) + 1 * l2_loss(pred, labels) 
                   
            running_val_loss += loss1.item() / len(val_dataloader)
        val_hist.append(running_val_loss)        
            
        toc = time.time()
            
        if epoch % 1 == 0:
            print("epoch", epoch)
            print("epoch", epoch, "takes", toc-tic)
            print("running train loss", running_train_loss)
            print("running val loss", running_val_loss)
            print("-"*50)
            if best_record > running_train_loss:
                print("best record", best_record)
                best_record = running_train_loss
                best_model = deepcopy(model)
            
        if epoch % 50 == 49:
            manualCheckpoint(epoch, train_hist, val_hist, best_model, "trained_model"+str(epoch), "train-history")
            
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
        with torch.no_grad():
            pred, _ = model(img_t)
            pred = regPred2Img(pred)
        toc = time.time()
        print("inference takes", toc-tic)
        print("unique", np.unique(pred.numpy()))
        displayInference(data, pred, save_dir, i, backend="DIODE")      


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
        with torch.no_grad():
            pred, _ = model(img_t)
            pred = regPred2Img(pred)
        toc = time.time()
        print("inference takes", toc-tic)
        print("unique", np.unique(pred.numpy()))
        displayInference(data, pred, save_dir, i, backend="DIODE")      


def testVizSeg(model, dataset, save_dir, device=torch.device("cpu"), num_example=5):
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
        with torch.no_grad():
            _, pred = model(img_t)
            pred = clsPred2Img(pred)
        toc = time.time()
        print("inference takes", toc-tic)
        print("unique", np.unique(pred))
        displayInference(data, pred, save_dir, i, backend="seg")  


if __name__ == "__main__":
    m = RegSegModel().cpu()
    dataset = KITTI_SEM(KITTI_SEM_TRAIN_RGB_PATHS, KITTI_SEM_TRAIN_LABEL_PATHS, device=torch.device("cpu"), original=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    testVizSeg(m, dataset, "train-history")