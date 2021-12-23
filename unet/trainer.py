import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchinfo import summary
from torchvision import transforms

from dataset.data_path import KITTI_SEM_TRAIN_RGB_PATHS, KITTI_SEM_TRAIN_LABEL_PATHS
from dataset.kitti import KITTI_SEM
from models.unet import UNet
from models.model_utils import showSegModelInference
from train_mult import trainSeg


def initTrainKITTISeg(save_dir, train_example_image_path):
    model_device = torch.device("cuda")
    data_device = torch.device("cpu")

    seg_dataset1 = KITTI_SEM(KITTI_SEM_TRAIN_RGB_PATHS, KITTI_SEM_TRAIN_LABEL_PATHS, is_val=False, device=data_device,
                             original=True)
    SPLIT = len(seg_dataset1) // 10
    train_seg_dataset = Subset(seg_dataset1, np.arange(SPLIT, len(seg_dataset1)))
    train_seg_dataloader = DataLoader(train_seg_dataset, batch_size=16, shuffle=True, num_workers=2, drop_last=True)

    seg_dataset2 = KITTI_SEM(KITTI_SEM_TRAIN_RGB_PATHS, KITTI_SEM_TRAIN_LABEL_PATHS, is_val=True, device=data_device,
                             original=True)
    test_seg_dataset = Subset(seg_dataset2, np.arange(0, SPLIT))
    test_seg_dataloader = DataLoader(test_seg_dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=True)

    m = UNet(3, 35, [24, 64, 128]).to(model_device)
    m.train()
    print("train dataloader lengths", len(train_seg_dataloader))
    model = trainSeg(m, train_seg_dataloader, test_seg_dataloader,
                     num_epoch=250, device=model_device, save_dir=save_dir, example_img_path=train_example_image_path)

    return model


def modelSummary():
    m1 = m2 = UNet(3, 35)
    summary(m1, input_size=(8, 3, 320, 320), device="cpu")
    summary(m2, input_size=(8, 3, 320, 320), device="cpu")


def showSegInference(model_path, img_path):
    model_dict = torch.load(model_path)
    model = UNet(3, 35)
    model.load_state_dict(model_dict)
    model.eval()
    model = model.cpu()
    showSegModelInference(model, img_path, display=True)
    plt.show()


def readArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=str, default='train')
    parser.add_argument('--train_save_dir', type=str, default='train-history')
    parser.add_argument('--train_example_image_path', type=str, default='')
    parser.add_argument('--infer_image_path', type=str)
    parser.add_argument('--infer_model_path', type=str)
    args = parser.parse_args()
    return args


def main():
    """
        python3 trainer.py --job train --train_save_dir train-history --train_example_image_path images/example1.png
        python3 trainer.py --job inferseg --infer_image_path images/red_car.png --infer_model_path train-history/trained_model199_dict.pth
    """
    args = readArgs()
    if args.job == "trainseg":
        initTrainKITTISeg(args.train_save_dir, args.train_example_image_path)
    elif args.job == "inferseg":
        showSegInference(args.infer_model_path, args.infer_image_path)
    elif args.job == "model_summary":
        modelSummary()


if __name__ == '__main__':
    main()
