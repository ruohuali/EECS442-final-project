import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.io import read_image
import cv2
import matplotlib.pyplot as plt
import os
import platform
from PIL import Image


def read2Tensor(img_path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), transform=None):
    '''image path -> unet input (1 x 3 x H x W)'''

    img_t = read_image(img_path).to(torch.float32).unsqueeze(0).to(device)
    img_t = transform(img_t)
    return img_t


def readLabel2Tensor(label_path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    '''image path -> loss input (1 x H x W)'''

    label_t = read_image(label_path).to(torch.float32).unsqueeze(0).to(device)
    transform = transforms.Compose([transforms.Resize((120, 480))])
    label_t = transform(label_t)
    return label_t


def clsPred2Img(pred):
    '''singleton unet output (1 x K x H x W) in [0, num class] -> plottable grayscale image (H x W)'''
    pred = pred.squeeze(0)
    pred = pred.argmax(0)
    pred = pred.squeeze(0)
    pred = pred.to("cpu").to(torch.long)
    return pred


def regPred2Img(pred):
    '''singleton unet output (1 x 1 x H x W) in [-1, 1] -> plottable grayscale image (H x W)'''
    pred = pred.squeeze(0)
    pred = pred.squeeze(0)
    # pred = (pred + 1) * 100
    pred = pred.to("cpu").to(torch.float32)
    return pred



def clearTemp():
    if platform.system() != "Windows":
        os.system("rm temp.png")
    else:
        raise NotImplementedError("write win cmd for removing temp file")


def showRegSegModelInference(model, img_path, preprocess=transforms.Compose([transforms.ToTensor(),
                                                                             transforms.CenterCrop((600, 352)),
                                                                             transforms.Resize((224, 224)),
                                                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                            std=[0.229, 0.224,
                                                                                                 0.225])]), display=True):
    """
    @func img_path -> 3 np arrays of results
    label assign refer
    https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    """

    img = Image.open(img_path)
    img_t = preprocess(img)
    img_t = img_t.unsqueeze(0)
    with torch.no_grad():
        reg_pred, seg_pred = model(img_t)
    reg_pred, seg_pred = regPred2Img(reg_pred), clsPred2Img(seg_pred)

    reg_pred_o, seg_pred_o = reg_pred.clone(), seg_pred.clone()
    img_o = np.array(img)
    img_o = cv2.resize(img_o, (reg_pred_o.shape[1], reg_pred_o.shape[0]))

    reg_pred[torch.logical_and(9 <= seg_pred, seg_pred <= 16)] = float('nan')
    reg_pred[torch.logical_and(0 <= seg_pred, seg_pred <= 6)] = float('nan')
    reg_pred[torch.logical_and(21 <= seg_pred, seg_pred <= 23)] = float('nan')

    ##
    plt.figure()
    plt.imshow(img)
    if not display:
        plt.close()

    ##
    cmap = plt.cm.jet
    cmap.set_bad(color="black")

    plt.figure()
    plt.imshow(reg_pred.numpy(), cmap=cmap, alpha=0.97)
    plt.imshow(img_o, alpha=0.6)
    plt.savefig("temp.png", bbox_inches='tight', pad_inches=0)
    if not display:
        plt.close()

    img_arr = cv2.imread("temp.png")
    clearTemp()

    ##
    plt.figure()
    plt.imshow(reg_pred_o.numpy(), cmap=cmap)
    plt.savefig("temp.png", bbox_inches='tight', pad_inches=0)
    if not display:
        plt.close()

    reg_pred_arr = cv2.imread("temp.png")
    clearTemp()

    ##
    plt.figure()
    plt.imshow(seg_pred.numpy())
    plt.savefig("temp.png", bbox_inches='tight', pad_inches=0)
    if not display:
        plt.close()

    seg_pred_arr = cv2.imread("temp.png")
    clearTemp()

    return reg_pred_arr, seg_pred_arr, img_arr


def regSegOutput2Img(regseg_output, np_img, display=False):
    reg_pred, seg_pred = regseg_output
    reg_pred, seg_pred = regPred2Img(reg_pred), clsPred2Img(seg_pred)

    reg_pred[torch.logical_and(9 <= seg_pred, seg_pred <= 16)] = float('nan')
    reg_pred[torch.logical_and(0 <= seg_pred, seg_pred <= 6)] = float('nan')
    reg_pred[torch.logical_and(21 <= seg_pred, seg_pred <= 23)] = float('nan')

    reg_pred = TF.resize(reg_pred.unsqueeze(0), np_img.shape[:-1]).squeeze()

    cmap = plt.cm.jet
    cmap.set_bad(color="black")

    plt.figure()
    plt.imshow(reg_pred.numpy(), cmap=cmap, alpha=0.97)
    plt.imshow(np_img, alpha=0.6)
    plt.savefig("temp.png", bbox_inches='tight', pad_inches=0)
    if not display:
        plt.close()

    img_arr = cv2.imread("temp.png")
    clearTemp()
    return img_arr


if __name__ == "__main__":
    pass
