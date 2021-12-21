from PIL import Image

from .data_utils import *

'''
@todo customized kitti transform
      more augmentation
      quantization
'''

class KITTITransform(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class KITTI_SEM(Dataset):
    '''
    @note assume that label image's name and corresponding rgb image's name are the same
    just in different folders. e.g. labels/1.jpg <---> rgbs/1.jpg
    '''
    rgb_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((200, 640)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    label_preprocess = transforms.Compose([transforms.Resize((200, 640))])

    def __init__(self, rgb_dir_paths, label_dir_paths, transform=None, target_transform=None,
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

        self.transform = transform if transform != None else self.rgb_preprocess
        self.target_transform = target_transform if target_transform != None else self.label_preprocess

        self.device = device
        self.qmark = False
        self.original = original

    def __len__(self):
        return len(self.data_pair_paths)

    def __getitem__(self, idx):
        img_path, label_path = self.data_pair_paths[idx]

        # img = Image.open(img_path)
        # label = read_image(label_path, ImageReadMode.GRAY)
        img = read_image(img_path).to(torch.float32)
        label = read_image(label_path, ImageReadMode.GRAY).to(torch.long)

        img = self.transform(img)
        label = self.target_transform(label)

        img = img.to(self.device)
        label = label.squeeze().to(self.device)

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
        original_image = data['original_rgb']
        original_label = data['original_label']

        image = image.permute(1, 2, 0).to("cpu").to(torch.long).numpy()
        label = label.to("cpu").numpy()
        # color_label = np.ones((*label.shape, 3))
        # color_label[:, :, 0] = 10
        # color_label[:, :, 1:] *= 100
        color_label = image.copy()
        color_label[:, :, 1] = label * 2
        color_label[:, :, 2] = 255 - label
        color_label[:, :, 0] = label
        print("shapes", image.shape, label.shape, original_image.shape, original_label.shape, color_label.shape)
        print(np.unique(image))

        plt.figure(figsize=(20, 10))
        print("color", image[200, 200])
        plt.imshow(image)

        plt.figure(figsize=(20, 10))
        plt.imshow(label)

        plt.figure(figsize=(20, 10))
        plt.imshow(color_label)

        plt.figure(figsize=(20, 10))
        plt.imshow(original_image)

        plt.figure(figsize=(20, 10))
        plt.imshow(original_label)

    def func(self):
        pass


class KITTI_DEP(Dataset):
    '''
    @note assume that label image's name and corresponding rgb image's name are the same
    just in different folders. e.g. labels/1.jpg <---> rgbs/1.jpg
    '''
    rgb_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((200, 640)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    label_preprocess = transforms.Compose([  # transforms.ToTensor(),
        transforms.GaussianBlur(15, sigma=4.0),
        transforms.Resize((200, 640))])

    def __init__(self, rgb_dir_paths, label_dir_paths, transform=None, target_transform=None,
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

        self.transform = transform if transform != None else self.rgb_preprocess
        self.target_transform = target_transform if target_transform != None else self.label_preprocess

        self.device = device
        self.qmark = False
        self.original = original

    def __len__(self):
        return len(self.data_pair_paths)

    def __getitem__(self, idx):
        img_path, label_path = self.data_pair_paths[idx]

        img = Image.open(img_path)
        label = cv2.imread(label_path)
        label = torch.tensor(label[:, :, 0]).unsqueeze(0)
        # label = label / torch.max(label)

        img = self.transform(img)
        label = self.target_transform(label)

        img = img.to(self.device)
        label = label.to(self.device).to(torch.float32)

        if self.original:
            original_img = cv2.imread(img_path)
            original_label = cv2.imread(label_path)[:, :, 0]
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
        original_image = data['original_rgb']
        original_label = data['original_label']

        image = image.permute(1, 2, 0).to("cpu").to(torch.uint8)
        label = label.permute(1, 2, 0).to("cpu").to(torch.uint8).squeeze(2).numpy()

        plt.figure(figsize=(20, 10))
        plt.imshow(image)
        plt.savefig(f"example image {idx}.png")

        plot_depth_map(label, np.ones_like(label), ".")

        plt.figure(figsize=(20, 10))
        plt.imshow(original_image)
        plt.savefig(f"example original image {idx}.png")

        h = depth2Heatmap(original_label)
        plt.figure(figsize=(20, 10))
        plt.imshow(h)
        plt.savefig(f"example original label {idx}.png")
