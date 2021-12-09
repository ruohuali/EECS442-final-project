import numpy as np
import torch
from torchvision import transforms
from torchvision.io import read_image
import cv2
import matplotlib.pyplot as plt
import os
import platform
from PIL import Image


def depth2Cutoff(depth_map, cutoff):
    '''@param depth_map ~ (H x W) needs to be numpy array'''
    heatmap = np.ones((*depth_map.shape, 3))
    heatmap[depth_map < cutoff] *= 150
    heatmap[depth_map >= cutoff] *= 50
    return heatmap.astype(np.int64)


def depth2Heatmap(depth_map, min_display=1e-4, max_display=255):
    '''@param depth_map ~ (H x W) needs to be numpy array'''
    r = (np.max(depth_map) - np.min(depth_map)) / 10
    heatmap = np.zeros((*depth_map.shape, 3))
    for i in range(depth_map.shape[0]):
        for j in range(depth_map.shape[1]):
            if not min_display < depth_map[i, j] < max_display:
                continue
            if depth_map[i, j] < r:
                heatmap[i, j, 2] = 255 - depth_map[i, j] / r * 255
                heatmap[i, j, 0] = depth_map[i, j] / r * 255
            elif r < depth_map[i, j] < 5 * r:
                heatmap[i, j, 0] = 255 - (depth_map[i, j] - r) / (4 * r) * 255
                heatmap[i, j, 1] = (depth_map[i, j] - r) / (4 * r) * 255
            else:
                heatmap[i, j, 1] = 255 - (depth_map[i, j] - 4 * r) / (6 * r) * 255
                heatmap[i, j, 2] = (depth_map[i, j] - 4 * r) / (6 * r) * 255
                # heatmap[:,:,0] *= 15
    return heatmap.astype(np.int64)


def drawHeatmap(depth_map, min_display=0, max_display=100):
    h = depth2Heatmap(depth_map, min_display=0, max_display=100)

    plt.figure(figsize=(20, 10))
    plt.imshow(h)
    plt.savefig("heat.png")
    x = np.arange(70).reshape(1, -1)
    x = np.vstack((x, x, x))
    hb = depth2Heatmap(x)
    plt.figure()
    plt.imshow(hb)


def plot_depth_map(dm, validity_mask, save_path):
    validity_mask = validity_mask > 0
    MIN_DEPTH = 0.5
    MAX_DEPTH = min(300, np.percentile(dm, 99))
    dm = np.clip(dm, MIN_DEPTH, MAX_DEPTH)
    dm = np.log(dm, where=validity_mask)

    dm = np.ma.masked_where(~validity_mask, dm)
    dm = dm.squeeze()

    cmap = plt.cm.jet
    cmap.set_bad(color='black')
    plt.imshow(dm, cmap=cmap, vmax=np.log(MAX_DEPTH))
    plt.savefig(save_path)


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


def displayInference(data, pred, save_dir, i, backend="cmap"):
    image = data['rgb']
    label = data['label']
    original_image = data['original_rgb']
    original_label = data['original_label']
    print("inference data sizes", image.shape, label.shape, original_image.shape, original_label.shape)

    image = image.permute(1, 2, 0).to("cpu").to(torch.uint8)
    label = label.squeeze().to(torch.uint8).cpu().numpy()
    pred = pred.numpy()

    if backend == "cmap":
        cmap = plt.cm.jet
        cmap.set_bad(color="black")

        plt.figure()
        plt.imshow(image)
        plt.savefig(os.path.join(save_dir, "img" + str(i) + ".png"))

        plt.figure()
        plt.imshow(label, cmap=cmap)
        plt.savefig(os.path.join(save_dir, "label" + str(i) + ".png"))

        plt.figure()
        plt.imshow(original_image)
        plt.savefig(os.path.join(save_dir, "oimg" + str(i) + ".png"))

        plt.figure()
        plt.imshow(original_label, cmap=cmap)
        plt.savefig(os.path.join(save_dir, "olabel" + str(i) + ".png"))

        plt.figure()
        plt.imshow(pred, cmap=cmap)
        plt.savefig(os.path.join(save_dir, "pred" + str(i) + ".png"))
    elif backend == "heatmap":
        plt.figure()
        plt.imshow(image)
        plt.savefig(os.path.join(save_dir, "img" + str(i) + ".png"))

        plt.figure()
        h = depth2Heatmap(label)
        plt.imshow(h)
        plt.savefig(os.path.join(save_dir, "label" + str(i) + ".png"))

        plt.figure()
        plt.imshow(original_image)
        plt.savefig(os.path.join(save_dir, "oimg" + str(i) + ".png"))

        plt.figure()
        h = depth2Heatmap(original_label)
        plt.imshow(h)
        plt.savefig(os.path.join(save_dir, "olabel" + str(i) + ".png"))

        plt.figure()
        h = depth2Heatmap(pred)
        plt.imshow(h)
        plt.savefig(os.path.join(save_dir, "pred" + str(i) + ".png"))
    elif backend == "DIODE":
        plt.figure()
        plt.imshow(image)
        plt.savefig(os.path.join(save_dir, "img" + str(i) + ".png"))

        plot_depth_map(label, np.ones_like(label), os.path.join(save_dir, "label" + str(i) + ".png"))

        plt.figure()
        plt.imshow(original_image)
        plt.savefig(os.path.join(save_dir, "oimg" + str(i) + ".png"))

        plot_depth_map(original_label, np.ones_like(original_label), os.path.join(save_dir, "olabel" + str(i) + ".png"))

        plot_depth_map(pred, np.ones_like(pred), os.path.join(save_dir, "pred" + str(i) + ".png"))
    elif backend == "seg":
        plt.figure()
        plt.imshow(image)
        plt.savefig(os.path.join(save_dir, "img" + str(i) + ".png"))

        plt.figure()
        plt.imshow(label)
        plt.savefig(os.path.join(save_dir, "label" + str(i) + ".png"))

        plt.figure()
        plt.imshow(original_image)
        plt.savefig(os.path.join(save_dir, "oimg" + str(i) + ".png"))

        plt.figure()
        plt.imshow(original_label)
        plt.savefig(os.path.join(save_dir, "olabel" + str(i) + ".png"))

        plt.figure()
        plt.imshow(pred)
        plt.savefig(os.path.join(save_dir, "pred" + str(i) + ".png"))

def showModelInference(model, img_path, preprocess=transforms.Compose([transforms.ToTensor(),
                                                                 transforms.Resize((200, 640)),
                                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                      std=[0.229, 0.224, 0.225])])):
    """
    @func img_path -> 3 np arrays of results
    """

    def clearTemp():
        if platform.system() != "Windows":
            os.system("rm temp.png")
        else:
            raise NotImplementedError("write win cmd for removing temp file")

    img = Image.open(img_path)
    img_t = preprocess(img)
    img_t = img_t.unsqueeze(0)
    with torch.no_grad():
        reg_pred, seg_pred = model.forward(img_t)
    reg_pred, seg_pred = regPred2Img(reg_pred), clsPred2Img(seg_pred)

    reg_pred_o, seg_pred_o = reg_pred.clone(), seg_pred.clone()
    img_o = np.array(img)
    img_o = cv2.resize(img_o, (reg_pred_o.shape[1], reg_pred_o.shape[0]))

    reg_pred = reg_pred.max() - reg_pred
    reg_pred7 = reg_pred.clone()
    reg_pred7[seg_pred != 7] = 0
    reg_pred26 = reg_pred.clone()
    reg_pred26[seg_pred != 26] = 0
    reg_pred = reg_pred7 + reg_pred26
    reg_pred[reg_pred == 0] = float('nan')

    ##
    cmap = plt.cm.jet
    cmap.set_bad(color="black")

    plt.figure()
    plt.imshow(reg_pred.numpy(), cmap=cmap, alpha=0.97)
    plt.imshow(img_o, alpha=0.6)
    plt.savefig("temp.png", bbox_inches='tight', pad_inches=0)

    img_arr = cv2.imread("temp.png")
    clearTemp()

    ##
    plt.figure()
    plt.imshow(reg_pred.numpy(), cmap=cmap)
    plt.savefig("temp.png", bbox_inches='tight', pad_inches=0)

    reg_pred_arr = cv2.imread("temp.png")
    clearTemp()

    ##
    plt.figure()
    plt.imshow(seg_pred.numpy())
    plt.savefig("temp.png", bbox_inches='tight', pad_inches=0)

    seg_pred_arr = cv2.imread("temp.png")
    clearTemp()

    return reg_pred_arr, seg_pred_arr, img_arr


if __name__ == "__main__":
    pass
