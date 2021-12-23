import numpy as np
import torch
from torchvision import transforms
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


def segPred2Img(pred):
    '''singleton unet output (1 x K x H x W) in [0, num class] -> plottable grayscale image (H x W)'''
    pred = pred.squeeze(0)
    pred = pred.argmax(0)
    pred = pred.squeeze(0)
    pred = pred.to("cpu").to(torch.long)
    return pred


def clearTemp():
    if platform.system() != "Windows":
        os.system("rm temp.png")
    else:
        raise NotImplementedError("write win cmd for removing temp file")


def showSegModelInference(model, img_path, preprocess=transforms.Compose([transforms.Resize((120, 480)),
                                                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                               std=[0.229, 0.224, 0.225])]), display=True):
    """
    @func img_path -> 3 np arrays of results
    label assign refer
    https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    """

    img = Image.open(img_path)
    img_p = np.array(img)
    img_p = cv2.resize(img_p, (480, 120))
    img_t = read_image(img_path).to(torch.float32)
    img_t = preprocess(img_t)
    img_t = img_t.unsqueeze(0)
    with torch.no_grad():
        seg_pred = model(img_t)
    seg_pred = segPred2Img(seg_pred)

    ##
    plt.figure()
    plt.imshow(img_p)
    plt.savefig("temp.png", bbox_inches='tight', pad_inches=0)
    if not display:
        plt.close()

    origin_img_arr = cv2.imread("temp.png")
    clearTemp()

    ##
    cmap = plt.cm.jet
    cmap.set_bad(color="black")

    plt.figure()
    plt.imshow(seg_pred.numpy(), cmap=cmap)
    plt.savefig("temp.png", bbox_inches='tight', pad_inches=0)
    if not display:
        plt.close()

    seg_pred_arr = cv2.imread("temp.png")
    clearTemp()

    return seg_pred_arr, origin_img_arr


if __name__ == "__main__":
    pass
