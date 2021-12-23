import torch
from PIL import Image
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF

from .data_utils import *

'''
@todo customized kitti transform
      more augmentation
      quantization
'''


class HorizontalFlip(object):
    def __init__(self, p):
        self.p = p
        self.flip = TF.hflip

    def __call__(self, x):
        """
        @param imgs: tensor (3 x H x W)
        @param labels: tensor (H x W)
        @return: flipped stuff
        """
        imgs, labels = x
        rand_num = torch.rand(1)[0]
        if rand_num > self.p:
            imgs = self.flip(imgs)
            labels = self.flip(labels)
        return imgs, labels


class PIL2NormalizedTensor(object):
    def __init__(self):
        pass

    def __call__(self, x):
        """
        @param imgs: PIL image
        @param labels: tensor (H x W)
        @return: tensor (3 x H x W) in [0, 1]
        """
        imgs, labels = x
        imgs = np.array(imgs)
        imgs = imgs.astype(np.float)
        imgs /= 255
        imgs = np.transpose(imgs, (2, 0, 1))
        imgs = torch.tensor(imgs).to(torch.float)
        labels = labels.squeeze().to(torch.long)
        return imgs, labels


class RandomColorChange(object):
    def __init__(self, color_jittor_p, blur_p, solarize_p, sharpen_p):
        self.color_jittor = transforms.ColorJitter(*color_jittor_p)
        self.blur_p = blur_p
        self.blur = transforms.GaussianBlur(kernel_size=(7, 7), sigma=(7, 7))
        self.solarize = transforms.RandomSolarize(threshold=192.0, p=solarize_p)
        self.sharpen = transforms.RandomAdjustSharpness(sharpness_factor=2, p=sharpen_p)

    def __call__(self, x):
        """
        @param imgs: tensor (3 x H x W)
        @param labels: tensor (H x W)
        @return: flipped stuff
        """
        imgs, labels = x
        imgs = self.color_jittor(imgs)
        rand_num = torch.rand(1)[0]
        if rand_num < self.blur_p:
            imgs = self.blur(imgs)
        imgs = self.solarize(imgs)
        imgs = self.sharpen(imgs)

        return imgs, labels


class Resize(object):
    def __init__(self, w, h):
        self.resize = transforms.Resize((w, h))

    def __call__(self, x):
        """
        @param imgs: tensor (3 x H x W)
        @param labels: tensor (H x W)
        @return: (3 x h x w), (h, w)
        """
        imgs, labels = x
        imgs = self.resize(imgs)
        labels = labels
        labels = self.resize(labels)
        return imgs, labels


class Standardize(object):
    def __init__(self):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, x):
        """
        @param imgs: tensor (3 x H x W)
        @param labels: tensor (H x W)
        @return: imgs with mean = 0, std = 1
        """
        imgs, labels = x
        imgs = self.normalize(imgs)
        return imgs, labels


def readKITTIData2Tensor(img_path, label_path, val=False):
    img = Image.open(img_path).convert('RGB')
    label = read_image(label_path, ImageReadMode.GRAY)
    if not val:
        preprocess = transforms.Compose([
            HorizontalFlip(0.5),
            RandomColorChange((0.2, 0.2, 0.2, 0.2), 0.2, 0., 0.2),
            Resize(120, 360),
            PIL2NormalizedTensor(),
            Standardize()
        ])
    else:
        preprocess = transforms.Compose([
            Resize(120, 360),
            PIL2NormalizedTensor(),
            Standardize()
        ])

    img, label = preprocess((img, label))

    return img, label


def readSingleImage2Tensor(img_path):
    img = Image.open(img_path).convert('RGB')
    label = torch.zeros(1, 500, 500)
    preprocess = transforms.Compose([
        Resize(120, 360),
        PIL2NormalizedTensor(),
        Standardize()
    ])

    img, label = preprocess((img, label))
    img = img.unsqueeze(0)
    return img


class KITTI_SEM(Dataset):
    '''
    @note assume that label image's name and corresponding rgb image's name are the same
    just in different folders. e.g. labels/1.jpg <---> rgbs/1.jpg
    '''
    def __init__(self, rgb_dir_paths, label_dir_paths, is_val=False,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), original=False):
        self.rgb_dir_paths = rgb_dir_paths
        self.label_dir_paths = label_dir_paths

        self.data_pair_paths = []
        for rgb_dir_path, label_dir_path in zip(rgb_dir_paths, label_dir_paths):
            rgb_data_names = sorted(os.listdir(rgb_dir_path))
            label_data_names = sorted(os.listdir(label_dir_path))
            for rgb_data_name, label_data_name in zip(rgb_data_names, label_data_names):
                img_path = os.path.join(rgb_dir_path, rgb_data_name)
                label_path = os.path.join(label_dir_path, label_data_name)
                self.data_pair_paths.append((img_path, label_path))

        self.is_val = is_val
        self.device = device
        self.original = original

    def toMode(self, is_val):
        self.is_val = is_val

    def __len__(self):
        return len(self.data_pair_paths)

    def __getitem__(self, idx):
        img_path, label_path = self.data_pair_paths[idx]
        img, label = readKITTIData2Tensor(img_path, label_path, val=self.is_val)

        if self.original:
            original_img = cv2.imread(img_path)
            original_img = cv2.resize(original_img, (1241, 376))
            original_label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            original_label = cv2.resize(original_label, (1241, 376))
        else:
            original_img = torch.empty(1)
            original_label = torch.empty(1)

        data = {}
        data['rgb'] = img
        data['label'] = label
        data['original_rgb'] = original_img
        data['original_label'] = original_label
        data['rgb_path'] = img_path
        data['label_path'] = label_path

        return data

    def example(self, idx=10):
        data = self.__getitem__(idx)
        image = data['rgb']
        label = data['label']
        image *= 255
        original_image = data['original_rgb']
        original_label = data['original_label']

        image = image.permute(1, 2, 0).to("cpu").to(torch.long).numpy()
        label = label.to("cpu").numpy()
        color_label = image.copy()
        print("shapes", image.shape, label.shape, original_image.shape, original_label.shape, color_label.shape)

        plt.figure(figsize=(20, 10))
        plt.imshow(image)

        plt.figure(figsize=(20, 10))
        plt.imshow(label)

        plt.figure(figsize=(20, 10))
        plt.imshow(original_image)

        plt.figure(figsize=(20, 10))
        plt.imshow(original_label)
