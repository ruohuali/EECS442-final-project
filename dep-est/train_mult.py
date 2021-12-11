import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.nn as nn
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from loss import SSIM, SmoothnessLoss
from models.model_utils import showModelInference


def getModelInference(model, img_path):
    model.eval()
    model = model.cpu()
    ret = showModelInference(model, img_path)
    return ret


def manualCheckpoint(epoch, loss_hist1, loss_hist2, best_model, model_name, save_dir):
    print("=" * 30)
    print(f"epoch {epoch} loss {min(loss_hist1)} {min(loss_hist2)}")
    model_path = os.path.join(save_dir, model_name + ".pth")
    torch.save(best_model, model_path)
    print(f"saved model to {model_path}")
    print("=" * 30)


def getRegLoss(pred, labels, imgs):
    l2_loss = nn.MSELoss()
    ssim = SSIM(window_size=7)
    smooth_loss = SmoothnessLoss()

    loss = 1 * (1 - ssim(pred, labels)) + 0.5 * smooth_loss(pred, imgs) + 1 * l2_loss(pred, labels)
    return loss


def getSegLoss(pred, labels, imgs):
    # weights = []
    # for i in range(35):
    #     if i != 7 or not 24 <= i <= 33:
    #         w = 0.3
    #     elif 21 <= i <= 23:
    #         w = 0.1
    #     else:
    #         w = 1
    #     weights.append(w)
    # weights = torch.tensor(weights, device=pred.device)
    weights = torch.tensor([1. for _ in range(35)], device=pred.device)
    ce = nn.CrossEntropyLoss(weight=weights)

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

        if optimizer is not None:
            reg_pred, _ = model(reg_imgs)
            reg_loss = getRegLoss(reg_pred, reg_labels, reg_imgs)

            _, seg_pred = model(seg_imgs)
            seg_loss = getSegLoss(seg_pred, seg_labels, seg_imgs)

            loss = 1 * reg_loss + 1 * seg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print("train batch", batch_idx, "/", min(len(reg_dataloader), len(seg_dataloader)), end='       \r')

        else:
            with torch.no_grad():
                reg_pred, _ = model(reg_imgs)
                reg_loss = getRegLoss(reg_pred, reg_labels, reg_imgs)

                _, seg_pred = model(seg_imgs)
                seg_loss = getSegLoss(seg_pred, seg_labels, seg_imgs)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0)

    best_record = np.inf
    train_hist = []
    val_hist = []
    for epoch in range(num_epoch):
        print("\n" * 5)
        print("-" * 50)
        print("epoch  ", epoch)
        tic = time.time()

        model.train()
        running_train_losses = doEpochDual(train_reg_dataloader, train_seg_dataloader, model, optimizer=optimizer,
                                           device=device)
        running_train_loss = running_train_losses['total']
        running_train_reg_loss = running_train_losses['reg']
        running_train_seg_loss = running_train_losses['seg']
        train_hist.append(running_train_loss)

        model.eval()
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


if __name__ == "__main__":
    pass
