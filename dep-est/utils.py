import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import time
from copy import deepcopy
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def depth2Heatmap(depth_map, min_display=0, max_display=100):
    r = (np.max(depth_map) - np.min(depth_map)) / 10
    heatmap = np.zeros((*depth_map.shape, 3))
    for i in range(depth_map.shape[0]):
        for j in range(depth_map.shape[1]):
            if not min_display < depth_map[i,j] < max_display:
                continue
            if depth_map[i,j] < r:
                heatmap[i,j,2] = 255 - depth_map[i,j] / r * 255    
                heatmap[i,j,0] = depth_map[i,j] / r * 255           
            elif r < depth_map[i,j] < 5*r:
                heatmap[i,j,0] = 255 - (depth_map[i,j]-r) / (4*r) * 255               
                heatmap[i,j,1] = (depth_map[i,j]-r) / (4*r) * 255  
            else:
                heatmap[i,j,1] = 255 - (depth_map[i,j]-4*r) / (6*r) * 255                                        
                heatmap[i,j,2] = (depth_map[i,j]-4*r) / (6*r) * 255  
    heatmap[:,:,0] *= 15
    return heatmap.astype(np.int64)


def drawHeatmap(depth_map, min_display=0, max_display=100):
    h = depth2Heatmap(depth_map, min_display=0, max_display=100)
    
    plt.figure(figsize=(20,10))
    plt.imshow(h)
    plt.savefig("heat.png")
    x = np.arange(70).reshape(1, -1)
    x = np.vstack((x,x,x))
    hb = depth2Heatmap(x)
    plt.figure()
    plt.imshow(hb)    

def read2Tensor(img_path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    img_t = read_image(img_path).to(torch.float32).unsqueeze(0).to(device)
    transform = transforms.Compose([transforms.Resize( (120, 480) ),
                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    img_t = transform(img_t)  
    return img_t

def readLabel2Tensor(label_path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    label_t = read_image(label_path).to(torch.float32).unsqueeze(0).to(device)
    transform = transforms.Compose([transforms.Resize( (120, 480) )])
    label_t = transform(label_t)  
    return label_t    

def pred2Img(pred):
    pred = pred.squeeze(0)
    pred = pred.squeeze(0)
    pred = pred.to("cpu").to(torch.uint8)
    return pred

class KITTI_DEP(Dataset):
    '''
    @note assume that label image's name and corresponding rgb image's name are the same
    just in different folders. e.g. labels/1.jpg <---> rgbs/1.jpg
    '''
    def __init__(self, rgb_dir_paths, label_dir_paths, transform=transforms.Compose([transforms.Resize( (80, 320) ),
                                          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
                          target_transform=transforms.Compose([transforms.Resize( (80, 320) )]),
                          device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), qmark=True, original=False):
        self.rgb_dir_paths = rgb_dir_paths
        self.label_dir_paths = label_dir_paths
        
        self.data_pair_paths = []
        for rgb_dir_path, label_dir_path in zip(rgb_dir_paths, label_dir_paths):
            data_names = sorted(os.listdir(label_dir_path))
            for data_name in data_names:
                img_path = os.path.join(rgb_dir_path, data_name)
                label_path = os.path.join(label_dir_path, data_name)   
                self.data_pair_paths.append((img_path, label_path))
        
        self.transform = transform
        self.target_transform = target_transform
        
        self.device = device
        self.qmark = qmark
        self.original = original

    def __len__(self):
        return len(self.data_pair_paths)

    def __getitem__(self, idx):
        img_path, label_path = self.data_pair_paths[idx]
           
        if self.qmark:
            img =  cv2.imread(img_path)
            img = transforms.ToTensor()(img)

            label = cv2.imread(label_path)
            label = label[:,:,0]
            label = label.reshape(1, *label.shape)
            label = torch.tensor(label).to(torch.float32)
        else:
            img = read_image(img_path)
            img = img.to(torch.float32)
                      
            label = read_image(label_path).to(torch.float32)
        img = img.to(self.device)  
        label = label.to(self.device)
        
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        if self.original:
            original_img = cv2.imread(img_path)     
            original_label = cv2.imread(label_path)[:,:,0]        
        else:
            original_img = torch.empty(1)
            original_label = torch.empty(1)                     
            
        data = {}
        data['rgb'] = img
        data['label'] = label
        data['original_rbg'] = original_img
        data['original_label'] = original_label
            
        return data
    
    def example(self, idx=10):
        data = self.__getitem__(idx)
        image = data['rgb']
        label = data['label']
        original_image = data['original_rbg']
        original_label = data['original_label']
        print("data", image.shape, label.shape, original_image.shape, original_label.shape)           
        
        image = image.permute(1, 2, 0).to("cpu").to(torch.uint8)
        label = label.permute(1, 2, 0).to("cpu").to(torch.uint8).squeeze(2)
        
        plt.figure(figsize=(20, 10))
        plt.imshow(image)
        plt.savefig(f"example image {idx}.png")

        h = depth2Heatmap(original_label)

        plt.figure(figsize=(20, 10))
        plt.imshow(h)    
        plt.savefig(f"example label {idx}.png")

        plt.figure(figsize=(20, 10))
        plt.imshow(original_image)    
        plt.savefig(f"example original image {idx}.png")


# DATA_PATH = "../../data/data_depth_self_compiled"
# TRAIN_PATH = os.path.join(DATA_PATH, "training")
# TRAIN_RGB_PATH = os.path.join(TRAIN_PATH, "rgb")
# TRAIN_LABEL_PATH = os.path.join(TRAIN_PATH, "label")
# TRAIN_RGB_PATHS = [os.path.join(TRAIN_RGB_PATH, "0001"),
#                 os.path.join(TRAIN_RGB_PATH, "0002"),
#                 os.path.join(TRAIN_RGB_PATH, "0009"),
#                 os.path.join(TRAIN_RGB_PATH, "0011"),
#                 os.path.join(TRAIN_RGB_PATH, "0017"),
#                 os.path.join(TRAIN_RGB_PATH, "0018"),
#                 os.path.join(TRAIN_RGB_PATH, "0048"),
#                 os.path.join(TRAIN_RGB_PATH, "0051")]

# TRAIN_DEP_PATHS = [os.path.join(TRAIN_LABEL_PATH, "0001"),
#                 os.path.join(TRAIN_LABEL_PATH, "0002"),
#                 os.path.join(TRAIN_LABEL_PATH, "0009"),
#                 os.path.join(TRAIN_LABEL_PATH, "0011"),
#                 os.path.join(TRAIN_LABEL_PATH, "0017"),
#                 os.path.join(TRAIN_LABEL_PATH, "0018"),
#                 os.path.join(TRAIN_LABEL_PATH, "0048"),
#                 os.path.join(TRAIN_LABEL_PATH, "0051")]

# TEST_PATH = os.path.join(DATA_PATH, "testing")
# TEST_RGB_PATH = os.path.join(TEST_PATH, "rgb")
# TEST_DEP_PATH = os.path.join(TEST_PATH, "label")


DATA_PATH = "/home/ruohuali/Desktop/depth-estimation/data_depth_selection/depth_selection/"
TRAIN_PATH = os.path.join(DATA_PATH, "test_depth_completion_anonymous")
TRAIN_RGB_PATH = os.path.join(TRAIN_PATH, "image")
TRAIN_DEP_PATH = os.path.join(TRAIN_PATH, "velodyne_raw")
TRAIN_RGB_PATHS = [TRAIN_RGB_PATH]
TRAIN_DEP_PATHS = [TRAIN_DEP_PATH]                


if __name__ == "__main__":
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name(torch.cuda.device_count()-1))

    dataset = KITTI_DEP(TRAIN_RGB_PATHS, TRAIN_DEP_PATHS, device=torch.device("cuda:0"), qmark=True, original=False)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    # dataset.example(519)
    print('len', len(dataset))
    tic = time.time()
    for i, data in enumerate(dataloader):
        print("data rgb", data['rgb'].shape)
        if i > 3:
            break
    print("time", time.time() - tic)    