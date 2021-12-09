import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset, DataLoader, Subset
from torchinfo import summary

import matplotlib.pyplot as plt
import cv2
import os
import time
from copy import deepcopy
import gc
from pdb import set_trace
import argparse

# from models import *
# from PATH import *
# from data import *

from models.regseg_model import ProbedDualTaskSeg, ConvProbe
from models.unet import UNet
from utils import *
from dataset.kitti import KITTI_DEP, KITTI_SEM
from dataset.diode import DIODE
from dataset.data_path import KITTI_DEP_TRAIN_RGB_PATHS, KITTI_DEP_TRAIN_LABEL_PATHS, \
                           KITTI_SEM_TRAIN_RGB_PATHS, KITTI_SEM_TRAIN_LABEL_PATHS
from train_mult import *


def initTrainDIODE():
    model_device = torch.device("cuda")
    data_device = device = torch.device("cpu")

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize( (320, 320) ),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    target_transform = transforms.Compose([transforms.ToTensor(),
                                           transforms.Resize( (320, 320) )])

    dataset = DIODE(TRAIN_PATHS, transform=preprocess, target_transform=target_transform, device=data_device)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    test_dataset = DIODE(TEST_PATHS, transform=preprocess, target_transform=target_transform, device=data_device)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

    m = ProbedDualTaskSeg("deeplab").to(model_device)
    m.train()
    model = trainSingle(m, dataloader, test_dataloader, "reg", num_epoch=100, device=model_device)

    m.eval()
    test_dataset = DIODE(TEST_PATHS, transform=preprocess, target_transform=target_transform, device=data_device, original=True)
    testViz(model, test_dataset, "train-history")


def initTrainKITTIReg():
    model_device = torch.device("cuda")
    data_device = device = torch.device("cpu")

    dataset = KITTI_DEP(KITTI_DEP_TRAIN_RGB_PATHS, KITTI_DEP_TRAIN_RGB_PATHS, device=data_device)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, drop_last=True)
    test_dataset = KITTI_DEP(KITTI_TEST_RGB_PATHS, KITTI_TEST_LABEL_PATHS, device=data_device)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, drop_last=True)

    m = ProbedDualTaskSeg().to(model_device)
    m.train()

    model = trainSingle(m, dataloader, test_dataloader, "reg", num_epoch=100, device=model_device)

    test_dataset = KITTI_DEP(KITTI_TEST_RGB_PATHS, KITTI_TEST_LABEL_PATHS, device=data_device, original=True)
    m.eval()
    testViz(model, test_dataset, "train-history")


def initTrainKITTISeg():
    model_device = torch.device("cuda")
    data_device = device = torch.device("cpu")

    dataset = KITTI_SEM(KITTI_SEM_TRAIN_RGB_PATHS, KITTI_SEM_TRAIN_LABEL_PATHS, device=data_device)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, drop_last=True)
    test_dataset = KITTI_SEM(KITTI_SEM_TRAIN_RGB_PATHS, KITTI_SEM_TRAIN_LABEL_PATHS, device=data_device)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, drop_last=True)

    m = ProbedDualTaskSeg().to(model_device)
    m.train()
    model = trainSingle(m, dataloader, test_dataloader, "seg", num_epoch=100, device=model_device)

    test_dataset = KITTI_SEM(KITTI_SEM_TRAIN_RGB_PATHS, KITTI_SEM_TRAIN_LABEL_PATHS, device=data_device, original=True)
    m.eval()
    testViz(model, test_dataset, "train-history")


def initTrainKITTIDual():
    model_device = torch.device("cuda")
    data_device = device = torch.device("cpu")

    rgb_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize( (200, 640) ),
        # transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        # transforms.RandomHorizontalFlip(p=0.5),        
        # transforms.RandomAdjustSharpness(sharpness_factor=2),
        # transforms.RandomSolarize(threshold=180, p=0.5),                
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    reg_dataset = KITTI_DEP(KITTI_DEP_TRAIN_RGB_PATHS, KITTI_DEP_TRAIN_LABEL_PATHS, device=data_device, transform=rgb_preprocess)
    SPLIT = len(reg_dataset) // 10
    # SPLIT = 10
    train_reg_dataset = Subset(reg_dataset, np.arange(SPLIT, len(reg_dataset)))
    train_reg_dataloader = DataLoader(train_reg_dataset, batch_size=2, shuffle=True, num_workers=2, drop_last=True)
    test_reg_dataset = Subset(reg_dataset, np.arange(0, SPLIT))
    test_reg_dataloader = DataLoader(test_reg_dataset, batch_size=2, shuffle=False, num_workers=2, drop_last=True)

    seg_dataset = KITTI_SEM(KITTI_SEM_TRAIN_RGB_PATHS, KITTI_SEM_TRAIN_LABEL_PATHS, device=data_device, transform=rgb_preprocess)
    SPLIT = len(seg_dataset) // 10
    # SPLIT = 10
    train_seg_dataset = Subset(seg_dataset, np.arange(SPLIT, len(seg_dataset)))
    train_seg_dataloader = DataLoader(train_seg_dataset, batch_size=2, shuffle=True, num_workers=2, drop_last=True)
    test_seg_dataset = Subset(seg_dataset, np.arange(0, SPLIT))
    test_seg_dataloader = DataLoader(test_seg_dataset, batch_size=2, shuffle=False, num_workers=2, drop_last=True)

    m = ProbedDualTaskSeg().to(model_device)
    # set_trace()
    m.train()
    print("train dataloader len", len(train_reg_dataloader), len(train_seg_dataloader))
    model = trainDual(m, train_reg_dataloader, test_reg_dataloader, train_seg_dataloader, test_seg_dataloader, num_epoch=250, device=model_device)

    test_dataset = KITTI_DEP(KITTI_DEP_TRAIN_RGB_PATHS, KITTI_DEP_TRAIN_LABEL_PATHS, device=data_device, original=True)
    m.eval()
    testViz(model, test_dataset, "train-history")

    test_dataset = KITTI_SEM(KITTI_SEM_TRAIN_RGB_PATHS, KITTI_SEM_TRAIN_LABEL_PATHS, device=data_device, original=True)
    m.eval()
    testViz(model, test_dataset, "train-history")


def modelSummary():
    m = ConvProbe(21)
    m.eval()
    summary(m, input_size=(8, 21, 320, 320), device="cuda")


def testModelKITTIReg(model_path):
    model_device = torch.device("cuda")
    data_device = device = torch.device("cpu")

    rgb_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize( (200, 640) ),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    label_preprocess = transforms.Compose([#transforms.ToTensor(),
                                           transforms.GaussianBlur(15, sigma=3.0),
                                           transforms.Resize( (200, 640) )])

    reg_dataset = KITTI_DEP(KITTI_DEP_TRAIN_RGB_PATHS, KITTI_DEP_TRAIN_LABEL_PATHS, device=data_device, transform=rgb_preprocess, original=True)
    SPLIT = len(reg_dataset) // 10
    test_reg_dataset = Subset(reg_dataset, np.arange(0, SPLIT))

    model = torch.load(model_path)

    testVizReg(model, test_reg_dataset, "train-history", num_example=10)


def testModelKITTISeg(model_path):
    model_device = torch.device("cuda")
    data_device = device = torch.device("cpu")

    seg_dataset = KITTI_SEM(KITTI_SEM_TRAIN_RGB_PATHS, KITTI_SEM_TRAIN_LABEL_PATHS, device=data_device, original=True)
    test_seg_dataset = Subset(seg_dataset, np.arange(0, 25))

    model = torch.load(model_path)

    testVizSeg(model, test_seg_dataset, "train-history", num_example=10)


def showModelInference(model_path, img_path):
    model = torch.load(model_path)
    model.eval()
    model = model.cpu()
    reg_pred, seg_pred, comb_pred = model.showInference(img_path)
    plt.figure()
    plt.imshow(reg_pred)
    plt.figure()
    plt.imshow(seg_pred)
    plt.figure()
    plt.imshow(comb_pred)
    plt.show()






import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import cv2
import os
import time
from copy import deepcopy
from pdb import set_trace
from tqdm import tqdm

from utils import *
from loss import SSIM, SmoothnessLoss
from PATH import *


def showModelInference(model, img_path):
    model.eval()
    model = model.cpu()
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize( (200, 640) ),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ret = model.showInference(img_path, "train-history", preprocess, 1)
    return ret


def manualCheckpoint(epoch, loss_hist1, loss_hist2, best_model, model_name, save_dir):
    print("=" * 30)
    print(f"epoch {epoch} loss {min(loss_hist1)} {min(loss_hist2)}")
    model_path = os.path.join(save_dir, model_name + ".pth")
    torch.save(best_model, model_path)
    print(f"saved model to {model_path}")
    plot_path = os.path.join(save_dir, str(epoch) + ".png")
    plt.figure()
    plt.plot(loss_hist1, label="train")
    plt.plot(loss_hist2, label="val")
    plt.savefig(plot_path)
    plt.legend(loc="upper left")
    print(f"plotted as {plot_path}")
    print("plotting")    
    print("=" * 30)
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


def doEpochDual(reg_dataloader, seg_dataloader, model, optimizer=None,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    running_loss = 0
    running_reg_loss = 0
    running_seg_loss = 0
    for batch_idx, data in enumerate(tqdm(zip(reg_dataloader, seg_dataloader), total=min(len(reg_dataloader), len(seg_dataloader)), desc="progress")):
    # for batch_idx, data in enumerate(zip(reg_dataloader, seg_dataloader)):
        reg_data, seg_data = data

        reg_imgs = reg_data['rgb'].to(device)
        reg_labels = reg_data['label'].to(device)

        seg_imgs = seg_data['rgb'].to(device)
        seg_labels = seg_data['label'].to(device)

        if optimizer != None:
            reg_pred, _ = model(reg_imgs)
            reg_loss = getLossReg(reg_pred, reg_labels, reg_imgs)

            _, seg_pred = model(seg_imgs)
            seg_loss = getLossSeg(seg_pred, seg_labels, seg_imgs)

            loss = 1 * reg_loss + 1 * seg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print("train batch", batch_idx, "/", min(len(reg_dataloader), len(seg_dataloader)), end='       \r')

        else:
            with torch.no_grad():
                reg_pred, _ = model(reg_imgs)
                reg_loss = getLossReg(reg_pred, reg_labels, reg_imgs)

                _, seg_pred = model(seg_imgs)
                seg_loss = getLossSeg(seg_pred, seg_labels, seg_imgs)

                loss = 1 * reg_loss + 1 * seg_loss
                # print("val batch", batch_idx, "/", min(len(reg_dataloader), len(seg_dataloader)), end='       \r')

        running_reg_loss += reg_loss.item() / min(len(reg_dataloader), len(seg_dataloader))
        running_seg_loss += seg_loss.item() / min(len(reg_dataloader), len(seg_dataloader))
        running_loss += loss.item() / min(len(reg_dataloader), len(seg_dataloader))

    running_losses = {"reg": running_reg_loss, "seg": running_seg_loss, "total": running_loss}

    return running_losses


def trainSingle(model, train_dataloader, val_dataloader, task, num_epoch=400,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
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
        elif task == "dual":
            running_train_loss = doEpochDual(train_dataloader, model, optimizer=optimizer, device=device)
        train_hist.append(running_train_loss)

        if task == "seg":
            running_val_loss = doEpochSeg(val_dataloader, model, optimizer=None, device=device)
        elif task == "reg":
            running_val_loss = doEpochReg(val_dataloader, model, optimizer=None, device=device)
        elif task == "dual":
            running_val_loss = doEpochDual(val_dataloader, model, optimizer=None, device=device)
        val_hist.append(running_val_loss)
        toc = time.time()

        if epoch % 1 == 0:
            print("epoch", epoch, "takes", toc - tic)
            print("running train loss", running_train_loss)
            print("running val loss", running_val_loss)
            print("-" * 50)
            if best_record > running_train_loss:
                print("best record", best_record)
                best_record = running_train_loss
                best_model = deepcopy(model)

        if epoch % 50 == 49:
            manualCheckpoint(epoch, train_hist, val_hist, best_model, "trained_model" + str(epoch), "train-history")

    return best_model


def trainDual(model, train_reg_dataloader, val_reg_dataloader, train_seg_dataloader, val_seg_dataloader, num_epoch=400,
              device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.train()

    writer = SummaryWriter()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    best_record = np.inf
    train_hist = []
    val_hist = []
    for epoch in range(num_epoch):
        print("\n" * 5)
        print("-" * 50)
        print("epoch  ", epoch)
        tic = time.time()

        running_train_losses = doEpochDual(train_reg_dataloader, train_seg_dataloader, model, optimizer=optimizer,
                                         device=device)
        running_train_loss = running_train_losses['total']
        running_train_reg_loss = running_train_losses['reg']
        running_train_seg_loss = running_train_losses['seg']
        train_hist.append(running_train_loss)

        running_val_losses = doEpochDual(val_reg_dataloader, val_seg_dataloader, model, optimizer=None, device=device)
        running_val_loss = running_val_losses['total']
        running_val_reg_loss = running_val_losses['reg']
        running_val_seg_loss = running_val_losses['seg']
        val_hist.append(running_val_loss)

        toc = time.time()
        # writer.add_scalar("Loss/train_total", running_train_loss, epoch)
        # writer.add_scalar("Loss/train_reg", running_train_reg_loss, epoch)
        # writer.add_scalar("Loss/train_seg", running_train_seg_loss, epoch)
        # writer.add_scalar("Loss/val_total", running_val_loss, epoch)
        # writer.add_scalar("Loss/val_reg", running_val_reg_loss, epoch)
        # writer.add_scalar("Loss/val_seg", running_val_seg_loss, epoch)        
        writer.add_scalars("Loss/total", {"train": running_train_loss, "val": running_val_loss}, epoch)
        writer.add_scalars("Loss/reg", {"train": running_train_reg_loss, "val": running_val_reg_loss}, epoch)
        writer.add_scalars("Loss/seg", {"train": running_train_seg_loss, "val": running_val_seg_loss}, epoch)

        if epoch % 1 == 0:
            print("epoch", epoch, "takes", toc - tic)
            print("running train loss", running_train_loss)
            print("running val loss", running_val_loss)
            if best_record > running_train_loss:
                print("best record", best_record)
                best_record = running_train_loss
                best_model = deepcopy(model)
            print("-" * 50)

        if epoch % 25 == 24:
            manualCheckpoint(epoch, train_hist, val_hist, best_model, "trained_model" + str(epoch), "train-history")
        if epoch % 2 == 1:
            _, _, img_arr = showModelInference(best_model, "example1.png")
            print("written image")
            writer.add_image("example", img_arr, epoch, dataformats='HWC')

    return best_model


def testViz(model, dataset, save_dir, device=torch.device("cpu"), num_example=5):
    model = model.to(device)
    model.eval()

    for i in range(num_example):
        idx = len(dataset) // (num_example + 1) * i

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
        print("inference takes", toc - tic)
        print("unique", np.unique(pred.numpy()))
        displayInference(data, pred, save_dir, i, backend="DIODE")


def testVizReg(model, dataset, save_dir, device=torch.device("cpu"), num_example=5):
    model = model.to(device)
    model.eval()

    for i in range(num_example):
        idx = len(dataset) // (num_example + 1) * i

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
        print("inference takes", toc - tic)
        print("unique", np.unique(pred.numpy()))
        # displayInference(data, pred, save_dir, i, backend="DIODE")      
        displayInference(data, pred, save_dir, i, backend="cmap")


def testVizSeg(model, dataset, save_dir, device=torch.device("cpu"), num_example=5):
    model = model.to(device)
    model.eval()

    for i in range(num_example):
        idx = len(dataset) // (num_example + 1) * i

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
        print("inference takes", toc - tic)
        print("unique", np.unique(pred))
        displayInference(data, pred, save_dir, i, backend="seg")


def testVizRegSeg(model, dataset, save_dir, device=torch.device("cpu"), num_example=5):
    model = model.to(device)
    model.eval()

    for i in range(num_example):
        idx = len(dataset) // (num_example + 1) * i

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
        print("inference takes", toc - tic)
        print("unique", np.unique(pred))
        displayInference(data, pred, save_dir, i, backend="seg")






if __name__ == '__main__':
    # initTrainKITTIDual()
    # initTrainKITTISeg()
    # initTrainKITTIReg()
    # initTrain()
    # modelSummary()
    # testModelKITTISeg(os.path.join("train-history", "trained_model99.pth"))
    # testModelKITTIReg(os.path.join("train-history", "trained_model49.pth"))
    showModelInference(os.path.join("train-history", "trained_model49.pth"), "example1.png")