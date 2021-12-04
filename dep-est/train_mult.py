import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from copy import deepcopy
from pdb import set_trace
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from tqdm import tqdm

from loss import SSIM, SmoothnessLoss


def getModelInference(model, img_path):
    model.eval()
    model = model.cpu()
    ret = model.showInference(img_path)
    return ret


def manualCheckpoint(epoch, loss_hist1, loss_hist2, best_model, model_name, save_dir):
    print("=" * 30)
    print(f"epoch {epoch} loss {min(loss_hist1)} {min(loss_hist2)}")
    model_path = os.path.join(save_dir, model_name + ".pth")
    torch.save(best_model, model_path)
    print(f"saved model to {model_path}")
    # plot_path = os.path.join(save_dir, str(epoch) + ".png")
    # plt.figure()
    # plt.plot(loss_hist1, label="train")
    # plt.plot(loss_hist2, label="val")
    # plt.savefig(plot_path)
    # plt.legend(loc="upper left")
    # print(f"plotted as {plot_path}")
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


def doEpochDual(reg_dataloader, seg_dataloader, model, optimizer=None,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    running_loss = 0
    running_reg_loss = 0
    running_seg_loss = 0
    for batch_idx, data in enumerate(
            tqdm(zip(reg_dataloader, seg_dataloader), total=min(len(reg_dataloader), len(seg_dataloader)),
                 desc="progress")):
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


def trainDual(model, train_reg_dataloader, val_reg_dataloader, train_seg_dataloader, val_seg_dataloader, num_epoch=400,
              device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), save_dir="train-history",
              example_img_path=''):
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

        if epoch % 25 == 24 or num_epoch - epoch == 1:
            manualCheckpoint(epoch, train_hist, val_hist, best_model, "trained_model" + str(epoch), save_dir)
        if epoch % 10 == 9:
            if example_img_path != '':
                reg_pred, seg_pred, img_arr = getModelInference(best_model, example_img_path)
                writer.add_image("example/combine", img_arr, epoch, dataformats='HWC')
                writer.add_image("example/reg", reg_pred, epoch, dataformats='HWC')
                writer.add_image("example/seg", seg_pred, epoch, dataformats='HWC')
                print("written image")

    writer.close()
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


if __name__ == "__main__":
    pass
