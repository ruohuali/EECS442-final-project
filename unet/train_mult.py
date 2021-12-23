import numpy as np
import os
import time
import torch
import torch.nn as nn
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from loss import SSIM, SmoothnessLoss
from models.model_utils import showSegModelInference


def getSegModelInference(model, img_path):
    model.eval()
    model = model.cpu()
    ret = showSegModelInference(model, img_path)
    return ret


def manualCheckpoint(epoch, loss_hist1, loss_hist2, best_model, model_name, save_dir):
    print("=" * 30)
    print(f"epoch {epoch} loss {min(loss_hist1)} {min(loss_hist2)}")
    model_path = os.path.join(save_dir, model_name + "_dict.pth")
    torch.save(best_model.state_dict(), model_path)
    print(f"saved model to {model_path}")
    print("=" * 30)


def getRegLoss(pred, labels, imgs):
    l2_loss = nn.MSELoss()
    ssim = SSIM(window_size=7)
    smooth_loss = SmoothnessLoss()

    loss = 1 * (1 - ssim(pred, labels)) + 0.5 * smooth_loss(pred, imgs) + 1 * l2_loss(pred, labels)
    return loss


def getSegLoss(pred, labels):
    ce = nn.CrossEntropyLoss()
    loss = ce(pred, labels)
    return loss


def doEpochSeg(seg_dataloader, model, optimizer=None,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    running_loss = 0
    for batch_idx, seg_data in enumerate(
            tqdm(seg_dataloader, total=len(seg_dataloader), desc="progress")):
        seg_imgs = seg_data['rgb'].to(device)
        seg_labels = seg_data['label'].to(device)

        if optimizer is not None:
            seg_pred = model(seg_imgs)
            seg_loss = getSegLoss(seg_pred, seg_labels)

            loss = seg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print("train batch", batch_idx, "/", min(len(reg_dataloader), len(seg_dataloader)), end='       \r')

        else:
            with torch.no_grad():
                seg_pred = model(seg_imgs)
                seg_loss = getSegLoss(seg_pred, seg_labels)

                loss = seg_loss
                # print("val batch", batch_idx, "/", min(len(reg_dataloader), len(seg_dataloader)), end='       \r')

        running_loss += loss.item() / len(seg_dataloader)

    running_losses = {"total": running_loss}

    return running_losses


def trainSeg(model, train_seg_dataloader, val_seg_dataloader, num_epoch=400,
              device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), save_dir="train-history",
              example_img_path=''):
    model.train()

    writer = SummaryWriter()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, weight_decay=1e-8, momentum=0.9)

    best_record = np.inf
    train_hist = []
    val_hist = []
    for epoch in range(num_epoch):
        print("\n" * 5)
        print("-" * 50)
        print("epoch  ", epoch)
        tic = time.time()

        model.train()
        running_train_losses = doEpochSeg(train_seg_dataloader, model, optimizer=optimizer, device=device)
        running_train_loss = running_train_losses['total']
        train_hist.append(running_train_loss)

        model.eval()
        running_val_losses = doEpochSeg(val_seg_dataloader, model, optimizer=None, device=device)
        running_val_loss = running_val_losses['total']
        val_hist.append(running_val_loss)

        toc = time.time()
        writer.add_scalars("Loss/total", {"train": running_train_loss, "val": running_val_loss}, epoch)

        if epoch % 1 == 0:
            print("epoch", epoch, "takes", toc - tic)
            print("running train loss", running_train_loss)
            print("running val loss", running_val_loss)
            if best_record > running_train_loss:
                print("best record", best_record, "-->", running_train_loss)
                best_record = running_train_loss
                best_model = deepcopy(model)
            print("-" * 50)

        if epoch % 25 == 24 or num_epoch - epoch == 1:
            manualCheckpoint(epoch, train_hist, val_hist, best_model, "trained_model" + str(epoch), save_dir)
        if epoch % 3 == 1:
            if example_img_path != '':
                seg_pred, original_img = getSegModelInference(best_model, example_img_path)
                writer.add_image("example/seg", seg_pred, epoch, dataformats='HWC')
                writer.add_image("example/img", original_img, epoch, dataformats='HWC')
                print("written image")

    writer.close()
    return best_model


if __name__ == "__main__":
    pass
