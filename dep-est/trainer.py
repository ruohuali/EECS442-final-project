import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchinfo import summary
from torchvision import transforms

from dataset.data_path import KITTI_SEM_TRAIN_RGB_PATHS, KITTI_SEM_TRAIN_LABEL_PATHS
from dataset.kitti import KITTI_SEM
from models.unet import UNet, DualTaskUNet
from models.model_utils import showSegModelInference
from train_mult import trainDual, trainSeg


def initTrainKITTISeg(save_dir, train_example_image_path):
    model_device = torch.device("cuda")
    data_device = torch.device("cpu")

    rgb_preprocess = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Resize((120, 480)),
        transforms.ColorJitter(brightness=0.2, hue=0.2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    label_preprocess2 = transforms.Compose([transforms.Resize((120, 480))])

    seg_dataset = KITTI_SEM(KITTI_SEM_TRAIN_RGB_PATHS, KITTI_SEM_TRAIN_LABEL_PATHS, device=data_device,
                            transform=rgb_preprocess, target_transform=label_preprocess2)
    SPLIT = len(seg_dataset) // 10
    # SPLIT = 10
    train_seg_dataset = Subset(seg_dataset, np.arange(SPLIT, len(seg_dataset)))
    train_seg_dataloader = DataLoader(train_seg_dataset, batch_size=7, shuffle=True, num_workers=2, drop_last=True)
    test_seg_dataset = Subset(seg_dataset, np.arange(0, SPLIT))
    test_seg_dataloader = DataLoader(test_seg_dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=True)

    m = UNet(3, 35).to(model_device)
    m.train()
    print("train dataloader lengths", len(train_seg_dataloader))
    model = trainSeg(m, train_seg_dataloader, test_seg_dataloader,
                     num_epoch=250, device=model_device, save_dir=save_dir, example_img_path=train_example_image_path)

    return model


def modelSummary():
    m2 = DualTaskUNet()
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
        python3 trainer.py --job infer --infer_image_path images/example1.png --infer_model_path train-history/trained_model99_dict.pth
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
