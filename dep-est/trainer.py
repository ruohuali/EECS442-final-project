import argparse
import cv2
import gc
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
from torch.utils.data import Dataset, DataLoader, Subset
from torchinfo import summary
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode

from dataset.data_path import KITTI_DEP_TRAIN_RGB_PATHS, KITTI_DEP_TRAIN_LABEL_PATHS, \
    KITTI_SEM_TRAIN_RGB_PATHS, KITTI_SEM_TRAIN_LABEL_PATHS
from dataset.diode import DIODE
from dataset.kitti import KITTI_DEP, KITTI_SEM
from models.regseg_model import ProbedDualTaskSeg, ConvProbe, \
                                DepthWiseSeparableConv2d, DepthWiseSeparableConvProbe, \
                                DualTaskSeg
from models.unet import DualTaskUNet
from train_mult import trainDual


def initTrainKITTIDual(save_dir, train_example_image_path):
    model_device = torch.device("cuda")
    data_device = device = torch.device("cpu")

    rgb_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((200, 640)),
        # transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        # transforms.RandomHorizontalFlip(p=0.5),        
        # transforms.RandomAdjustSharpness(sharpness_factor=2),
        # transforms.RandomSolarize(threshold=180, p=0.5),                
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    reg_dataset = KITTI_DEP(KITTI_DEP_TRAIN_RGB_PATHS, KITTI_DEP_TRAIN_LABEL_PATHS, device=data_device,
                            transform=rgb_preprocess)
    SPLIT = len(reg_dataset) // 10
    # SPLIT = 10
    train_reg_dataset = Subset(reg_dataset, np.arange(SPLIT, len(reg_dataset)))
    train_reg_dataloader = DataLoader(train_reg_dataset, batch_size=2, shuffle=True, num_workers=2, drop_last=True)
    test_reg_dataset = Subset(reg_dataset, np.arange(0, SPLIT))
    test_reg_dataloader = DataLoader(test_reg_dataset, batch_size=2, shuffle=False, num_workers=2, drop_last=True)

    seg_dataset = KITTI_SEM(KITTI_SEM_TRAIN_RGB_PATHS, KITTI_SEM_TRAIN_LABEL_PATHS, device=data_device,
                            transform=rgb_preprocess)
    SPLIT = len(seg_dataset) // 10
    # SPLIT = 10
    train_seg_dataset = Subset(seg_dataset, np.arange(SPLIT, len(seg_dataset)))
    train_seg_dataloader = DataLoader(train_seg_dataset, batch_size=2, shuffle=True, num_workers=2, drop_last=True)
    test_seg_dataset = Subset(seg_dataset, np.arange(0, SPLIT))
    test_seg_dataloader = DataLoader(test_seg_dataset, batch_size=2, shuffle=False, num_workers=2, drop_last=True)

    # m = RegSegModel(depthwise=True).to(model_device)
    m = DualTaskUNet().to(model_device)
    # set_trace()
    m.train()
    print("train dataloader lengths", len(train_reg_dataloader), len(train_seg_dataloader))
    model = trainDual(m, train_reg_dataloader, test_reg_dataloader, train_seg_dataloader, test_seg_dataloader,
                      num_epoch=50, device=model_device, save_dir=save_dir, example_img_path=train_example_image_path)

    return model


def modelSummary():
    m1 = DualTaskSeg()
    m2 = DualTaskUNet()
    summary(m1, input_size=(8, 3, 320, 320), device="cpu")
    print(1111)        
    summary(m2, input_size=(8, 3, 320, 320), device="cpu")
    print(2222)           


def showModelInference(model_path, img_path):
    model = torch.load(model_path)
    model.eval()
    model = model.cpu()
    reg_pred, seg_pred, comb_pred = model.showInference(img_path)
    # plt.figure()
    # plt.imshow(reg_pred)
    # plt.figure()
    # plt.imshow(seg_pred)
    # plt.figure()
    # plt.imshow(comb_pred)
    plt.show()


def main():
    """
        python3 trainer.py --job train --train_save_dir train-history --train_example_image_path example1.png
        python3 trainer.py --job infer --infer_image_path example1.png --infer_model_path train-history/trained_model49.pth
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=str, default='train')
    parser.add_argument('--train_save_dir', type=str, default='train-history')
    parser.add_argument('--train_example_image_path', type=str, default='')
    parser.add_argument('--infer_image_path', type=str)
    parser.add_argument('--infer_model_path', type=str)
    args = parser.parse_args()
    if args.job == "train":
        initTrainKITTIDual(args.train_save_dir, args.train_example_image_path)
    elif args.job == "infer":
        showModelInference(args.infer_model_path, args.infer_image_path)
    elif args.job == "model_summary":
        modelSummary()


if __name__ == '__main__':
    main()
