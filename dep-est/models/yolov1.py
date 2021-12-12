import os
# os.system("pip install git+https://github.com/deepvision-class/starter-code")
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import cv2
import copy
import time
import shutil
import os

# parameters for plotting
plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# data type and device for torch.tensor
# unpack as argument to torch functions, like so: **to_float
to_float = {'dtype': torch.float, 'device': 'cpu'}
to_float_cuda = {'dtype': torch.float, 'device': 'cuda'}
to_double = {'dtype': torch.double, 'device': 'cpu'}
to_double_cuda = {'dtype': torch.double, 'device': 'cuda'}
to_long = {'dtype': torch.long, 'device': 'cpu'}
to_long_cuda = {'dtype': torch.long, 'device': 'cuda'}


def get_pascal_voc2007_data(image_root, split='train'):
    """
    Use torchvision.datasets
    https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.VOCDetection
    """
    from torchvision import datasets

    train_dataset = datasets.VOCDetection(image_root, year='2007', image_set=split,
                                          download=True)

    return train_dataset


def pascal_voc2007_loader(dataset, batch_size, num_workers=0):
    """
    Data loader for Pascal VOC 2007.
    https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """
    from torch.utils.data import DataLoader
    # turn off shuffle so we can index the original image
    train_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=False, pin_memory=True,
                              num_workers=num_workers,
                              collate_fn=voc_collate_fn)
    return train_loader


class_to_idx = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
                'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10,
                'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15,
                'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19
                }
idx_to_class = {i: c for c, i in class_to_idx.items()}

from torchvision import transforms


def voc_collate_fn(batch_lst, reshape_size=224):
    preprocess = transforms.Compose([
        transforms.Resize((reshape_size, reshape_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    batch_size = len(batch_lst)

    img_batch = torch.zeros(batch_size, 3, reshape_size, reshape_size)

    max_num_box = max(len(batch_lst[i][1]['annotation']['object']) \
                      for i in range(batch_size))

    box_batch = torch.Tensor(batch_size, max_num_box, 5).fill_(-1.)
    w_list = []
    h_list = []
    img_id_list = []

    for i in range(batch_size):
        img, ann = batch_lst[i]
        w_list.append(img.size[0])  # image width
        h_list.append(img.size[1])  # image height
        img_id_list.append(ann['annotation']['filename'])
        img_batch[i] = preprocess(img)
        all_bbox = ann['annotation']['object']
        if type(all_bbox) == dict:  # inconsistency in the annotation file
            all_bbox = [all_bbox]
        for bbox_idx, one_bbox in enumerate(all_bbox):
            bbox = one_bbox['bndbox']
            obj_cls = one_bbox['name']
            box_batch[i][bbox_idx] = torch.Tensor([float(bbox['xmin']), float(bbox['ymin']),
                                                   float(bbox['xmax']), float(bbox['ymax']), class_to_idx[obj_cls]])

    h_batch = torch.tensor(h_list)
    w_batch = torch.tensor(w_list)

    return img_batch, box_batch, w_batch, h_batch, img_id_list


def coord_trans(bbox, w_pixel, h_pixel, w_amap=7, h_amap=7, mode='a2p'):
    """
    Coordinate transformation function. It converts the box coordinate from
    the image coordinate system to the activation map coordinate system and vice versa.
    In our case, the input image will have a few hundred pixels in
    width/height while the activation map is of size 7x7.

    Input:
    - bbox: Could be either bbox, anchor, or proposal, of shape Bx*x4
    - w_pixel: Number of pixels in the width side of the original image, of shape B
    - h_pixel: Number of pixels in the height side of the original image, of shape B
    - w_amap: Number of pixels in the width side of the activation map, scalar
    - h_amap: Number of pixels in the height side of the activation map, scalar
    - mode: Whether transfer from the original image to activation map ('p2a') or
            the opposite ('a2p')

    Output:
    - resized_bbox: Resized box coordinates, of the same shape as the input bbox
    """

    assert mode in ('p2a', 'a2p'), 'invalid coordinate transformation mode!'
    assert bbox.shape[-1] >= 4, 'the transformation is applied to the first 4 values of dim -1'

    if bbox.shape[0] == 0:  # corner cases
        return bbox

    resized_bbox = bbox.clone()
    # could still work if the first dim of bbox is not batch size
    # in that case, w_pixel and h_pixel will be scalars
    resized_bbox = resized_bbox.view(bbox.shape[0], -1, bbox.shape[-1])
    invalid_bbox_mask = (resized_bbox == -1)  # indicating invalid bbox

    if mode == 'p2a':
        # pixel to activation
        width_ratio = w_pixel * 1. / w_amap
        height_ratio = h_pixel * 1. / h_amap
        resized_bbox[:, :, [0, 2]] /= width_ratio.view(-1, 1, 1)
        resized_bbox[:, :, [1, 3]] /= height_ratio.view(-1, 1, 1)
    else:
        # activation to pixel
        width_ratio = w_pixel * 1. / w_amap
        height_ratio = h_pixel * 1. / h_amap
        resized_bbox[:, :, [0, 2]] *= width_ratio.view(-1, 1, 1)
        resized_bbox[:, :, [1, 3]] *= height_ratio.view(-1, 1, 1)

    resized_bbox.masked_fill_(invalid_bbox_mask, -1)
    resized_bbox.resize_as_(bbox)
    return resized_bbox


"""# Data Visualizer
This function will help us visualize boxes on top of images.
"""


def data_visualizer(img, idx_to_class, bbox=None, pred=None):
    """
    Data visualizer on the original image. Support both GT box input and proposal
    input.

    Input:
    - img: PIL Image input
    - idx_to_class: Mapping from the index (0-19) to the class name
    - bbox: GT bbox (in red, optional), a tensor of shape Nx5, where N is
            the number of GT boxes, 5 indicates (x_tl, y_tl, x_br, y_br, class)
    - pred: Predicted bbox (in green, optional), a tensor of shape N'x6, where
            N' is the number of predicted boxes, 6 indicates
            (x_tl, y_tl, x_br, y_br, class, object confidence score)
    """

    img_copy = np.array(img).astype('uint8')

    if bbox is not None:
        for bbox_idx in range(bbox.shape[0]):
            one_bbox = bbox[bbox_idx][:4]
            cv2.rectangle(img_copy, (one_bbox[0], one_bbox[1]), (one_bbox[2],
                                                                 one_bbox[3]), (255, 0, 0), 2)
            if bbox.shape[1] > 4:  # if class info provided
                obj_cls = idx_to_class[bbox[bbox_idx][4].item()]
                cv2.putText(img_copy, '%s' % (obj_cls),
                            (one_bbox[0], one_bbox[1] + 15),
                            cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)

    if pred is not None:
        for bbox_idx in range(pred.shape[0]):
            one_bbox = pred[bbox_idx][:4]
            first_arg = (int(one_bbox[0]), int(one_bbox[1]))
            second_arg = (int(one_bbox[2]), int(one_bbox[3]))
            # cv2.rectangle(img_copy, (one_bbox[0], one_bbox[1]), (one_bbox[2], one_bbox[3]), (0, 255, 0), 2)
            cv2.rectangle(img_copy, first_arg, second_arg, (0, 255, bbox_idx * 255), 2)

            if pred.shape[1] > 4:  # if class and conf score info provided
                obj_cls = idx_to_class[pred[bbox_idx][4].item()]
                conf_score = pred[bbox_idx][5].item()
                # cv2.putText(img_copy, '%s, %.2f' % (obj_cls, conf_score),
                #             (one_bbox[0], one_bbox[1]+15),
                #             cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)
                cv2.putText(img_copy, '%s, %.2f' % (obj_cls, conf_score),
                            (int(one_bbox[0]), int(one_bbox[1]) + 15),
                            cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)

    plt.imshow(img_copy)
    plt.axis('off')
    plt.show()


"""# (a) Detector Backbone Network
Here, we use a [MobileNet v2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/) for image feature extraction.
"""


class FeatureExtractor(nn.Module):
    """
    Image feature extraction with MobileNet.
    """

    def __init__(self, reshape_size=224, pooling=False, verbose=False):
        super().__init__()

        from torchvision import models
        from torchsummary import summary

        # self.mobilenet = models.mobilenet_v2(pretrained=True)
        # self.mobilenet = nn.Sequential(*list(self.mobilenet.children())[:-1]) # Remove the last classifier

        self.mobilenet = models.mobilenet_v3_small(pretrained=True, process=True)  # modified
        self.mobilenet = self.mobilenet.features

        # average pooling
        if pooling:
            self.mobilenet.add_module('LastAvgPool',
                                      nn.AvgPool2d(math.ceil(reshape_size / 32.)))  # input: N x 1280 x 7 x 7

        # for i in self.mobilenet.named_parameters():
        #   i[1].requires_grad = True # fine-tune all parameters
        for param in self.mobilenet.parameters():
            param.requires_grad = False  # modified

        if verbose:
            summary(self.mobilenet.cuda(), (3, reshape_size, reshape_size))

    def forward(self, img, verbose=False):
        """
        Inputs:
        - img: Batch of resized images, of shape Nx3x224x224

        Outputs:
        - feat: Image feature, of shape Nx1280 (pooled) or Nx1280x7x7
        """
        num_img = img.shape[0]

        img_prepro = img

        feat = []
        process_batch = 500
        for b in range(math.ceil(num_img / process_batch)):
            feat.append(self.mobilenet(img_prepro[b * process_batch:(b + 1) * process_batch]
                                       ).squeeze(-1).squeeze(-1))  # forward and squeeze
        feat = torch.cat(feat)

        if verbose:
            print('Output feature shape: ', feat.shape)

        return feat


"""Now, let's see what's inside MobileNet v2. Assume we have a 3x224x224 image input."""

"""# (b) Activation and Proposal

## Activation Grid Generator
After passing the input image through the backbone network, we have a convolutional feature map of shape $(D, 7, 7)$ which we interpret as a $7$x$7$ grid of $D-$dimensional features. At each point in this grid, we predict a set of $A$ bounding boxes. The total number of bounding boxes we predict for a single image are thus $A$x$7$x$7$. 

Centered at each position of the $7$x$7$ activation grid, we predict $A$ bounding boxes, where $A = 2$ in our case. In order to place bounding boxes centered at each position of the $7$x$7$ grid of backbone features, we need to know the spatial position of the center of each cell in the grid of features.

This function will compute these center coordinates for us.
"""


def GenerateGrid(batch_size, w_amap=7, h_amap=7, dtype=torch.float32, device='cuda'):
    """
    Return a grid cell given a batch size (center coordinates).

    Inputs:
    - batch_size, B
    - w_amap: or W', width of the activation map (number of grids in the horizontal dimension)
    - h_amap: or H', height of the activation map (number of grids in the vertical dimension)
    - W' and H' are always 7 in our case while w and h might vary.

    Outputs:
    grid: A float32 tensor of shape (B, H', W', 2) giving the (x, y) coordinates
          of the centers of each feature for a feature map of shape (B, D, H', W')
    """
    w_range = torch.arange(0, w_amap, dtype=dtype, device=device) + 0.5
    h_range = torch.arange(0, h_amap, dtype=dtype, device=device) + 0.5

    w_grid_idx = w_range.unsqueeze(0).repeat(h_amap, 1)
    h_grid_idx = h_range.unsqueeze(1).repeat(1, w_amap)
    grid = torch.stack([w_grid_idx, h_grid_idx], dim=-1)
    grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    return grid


"""## (b) TODO: Proposal Generator

Now, consider a grid with center, width and height $(x_c^g,y_c^g)$.
The Prediction Network that you will finish implementing later will predict *offsets* $(t^x, t^y, t^w, t^h)$; applying this transformation yields the *bounding box or proposal* with center x, center y, width and height $(x_c^p,y_c^p,w^p,h^p)$. To convert the offsets to the actual bounding box/proposal parameters, use the below mentioned transformation formulas.

Notice here we parametrize the boxes in form `(x, y, w, h)`, where `x, y` represent the center coordinates, and `w, h` represent the width and height respectively. This is in contrast to the `(xtl, ytl, xbr, ybr)` parametrization 
for the GT bounding boxes. While writing functions and performing operations,
be mindful of which of these forms your bounding boxes are of. In this case,
even though we apply the transformations to `(x, y, w, h)` parametrized form, we expect the final propsals to be in `(xtl, ytl, xbr, ybr)` form, so you will need to convert from `(x, y, w, h)` to `(xtl, ytl, xbr, ybr)` format before returning proposals. Below are the transformation formulations to help you get bounding boxes or proposals. Here we assume that the shape of the activation map is $7$ x $7$.

We assume that $t^x$ and $t^y$ are both in the range $-0.5\leq t^x,t^y\leq 0.5$, while $t^w$ and $t^h$ are real numbers in the range $(0, 1)$. Then we have:
- $x_c^p = x_c^g + t^x$
- $y_c^p = y_c^g + t^y$
- $w^p = (t^w)*7$
- $h^p = (t^h)*7$


### Training
During training, we compute the ground-truth transformation $(\hat{t^x}, \hat{t^y}, \hat{t^w}, \hat{t^h})$ that would, with the help of the grid center coordinates, yield the bounding box $(x_c^p,y_c^p,w^p,h^p)$ which we expect to match the ground-truth box $(x_c^{gt},y_c^{gt},w^{gt},h^{gt})$. We then apply a regression loss that penalizes differences between the predicted transform $(t^x, t^y, t^w, t^h)$ and the ground-truth transform.

"""


def GenerateProposal(grids, offsets):
    """
    Proposal generator.

    Inputs:
    - grids: Activation grids, of shape (B, H', W', 2). Grid centers are
    represented by their coordinates in the activation map.
    - offsets: Transformations obtained from the Prediction Network
      of shape (B, A, H', W', 4) that will be used to generate proposals region
      proposals. The transformation offsets[b, a, h, w] = (tx, ty, tw, th) will be
      applied to the grids[b, a, h, w].
      Assume that tx and ty are in the range
      (-0.5, 0.5) and h,w are normalized and thus in the range (0, 1).

    Outputs:
    - proposals: Region proposals of shape (B, A, H', W', 4), represented by the
      coordinates of their top-left and bottom-right corners. Using the
      transform offsets[b, a, h, w] and girds[b, a, h, w] should give the proposals.
      The expected parametrization of the proposals is (xtl, ytl, xbr, ybr).

    CAUTION:
      Notice that the offsets here are parametrized as (x, y, w, h).
      The proposals you return need to be of the form (xtl, ytl, xbr, ybr).
    """
    proposals = None

    #############################################################################
    # TODO: Given grid coordinates and the proposed offset for each bounding    #
    # box, compute the proposal coordinates using the transformation formulas   #
    # above.                                                                    #
    #############################################################################
    # 1. Follow the formulas above to convert the grid centers into proposals.

    # 2. Convert the proposals into (xtl, ytl, xbr, ybr) coordinate format as
    # mentioned in the header and in the cell above that.
    B, H, W, _ = grids.shape
    _, A, _, _, _ = offsets.shape

    grids = torch.unsqueeze(grids, 1)

    xcp = grids[:, :, :, :, 0] + offsets[:, :, :, :, 0]
    ycp = grids[:, :, :, :, 1] + offsets[:, :, :, :, 1]
    wp = offsets[:, :, :, :, 2] * 7 / 2
    hp = offsets[:, :, :, :, 3] * 7 / 2
    proposals = torch.stack((xcp - wp, ycp - hp, xcp + wp, ycp + hp), 4)
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return proposals


"""# (c-e) Prediction Networks

## (c) TODO: Intersection Over Union (IoU)
You will implement the IoU function. Carefully read the expected inputs and outputs. This function will also be used later in the ObjectClassification module to calculate the IoU between the predicted bounding boxes/proposals and the ground truth bounding boxes. 
**NOTE**: Keep in mind the parametrization of the input and output bounding boxes. (Whether it is (x, y, w, h) or (xtl, ytl, xbr, ybr)). The definition of IoU can be found in the [lecture slides](https://www.eecs.umich.edu/courses/eecs442-ahowens/fa21/slides/lec12-object.pdf) (slides 46-47).
"""


def IoU(proposals, bboxes):
    """
    Compute intersection over union between sets of bounding boxes.

    Inputs:
    - proposals: Proposals of shape (B, A, H', W', 4). These should be parametrized
      as (xtl, ytl, xbr, ybr).
    - bboxes: Ground-truth boxes from the DataLoader of shape (B, N, 5).
      Each ground-truth box is represented as tuple (x_tl, y_tl, x_br, y_br, class).
      If image i has fewer than N boxes, then bboxes[i] will be padded with extra
      rows of -1.

    Outputs:
    - iou_mat: IoU matrix of shape (B, A*H'*W', N) where iou_mat[b, i, n] gives
      the IoU between one element of proposals[b] and bboxes[b, n].

    For this implementation you DO NOT need to filter invalid proposals or boxes;
    in particular you don't need any special handling for bboxxes that are padded
    with -1.
    """
    iou_mat = None

    #############################################################################
    # TODO: Compute the Intersection over Union (IoU) on proposals and GT boxes.#
    # No need to filter invalid proposals/bboxes (i.e., allow region area <= 0).#
    # However, you need to make sure to compute the IoU correctly (it should be #
    # 0 in those cases.                                                         #
    # You need to ensure your implementation is efficient (no for loops).       #
    # HINT:                                                                     #
    # IoU = Area of Intersection / Area of Union, where                         #
    # Area of Union = Area of Proposal + Area of BBox - Area of Intersection    #
    # and the Area of Intersection can be computed using the top-left corner and#
    # bottom-right corner of proposal and bbox. Think about their relationships.#
    #############################################################################
    B, A, H, W, p_ori = proposals.shape
    B, N, bb_ori = bboxes.shape
    p = torch.unsqueeze(torch.reshape(proposals, [B, A * H * W, p_ori]), dim=2)
    bb = torch.unsqueeze(bboxes, dim=1)

    xtl = p[:, :, :, 0]
    ytl = p[:, :, :, 1]
    xbr = p[:, :, :, 2]
    ybr = p[:, :, :, 3]
    x_tl = bb[:, :, :, 0]
    y_tl = bb[:, :, :, 1]
    x_br = bb[:, :, :, 2]
    y_br = bb[:, :, :, 3]

    inter_xtl = torch.max(xtl, x_tl)
    inter_ytl = torch.max(ytl, y_tl)
    inter_xbr = torch.min(xbr, x_br)
    inter_ybr = torch.min(ybr, y_br)
    zero = torch.zeros(inter_xtl.shape, **to_float_cuda)
    intersection = torch.max(zero, inter_xbr - inter_xtl) * torch.max(zero, inter_ybr - inter_ytl)
    union = abs((x_br - x_tl) * (y_tl - y_br)) + abs((xbr - xtl) * (ytl - ybr)) - intersection

    iou_mat = intersection / union
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return iou_mat


class PredictionNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_bboxes=2, num_classes=20, drop_ratio=0.3):
        super().__init__()

        assert (num_classes != 0 and num_bboxes != 0)
        self.num_classes = num_classes
        self.num_bboxes = num_bboxes

        # Here we set up a network that will predict outputs for all bounding boxes.
        # This network has a 1x1 convolution layer with `hidden_dim` filters,
        # followed by a Dropout layer with `p=drop_ratio`, a Leaky ReLU
        # nonlinearity, and finally another 1x1 convolution layer to predict all
        # outputs. The network is stored in `self.net`, and has shape
        # (B, 5*A+C, 7, 7), where the 5 predictions are in the order
        # (x, y, w, h, conf_score), with A = `self.num_bboxes`
        # and C = `self.num_classes`.

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.drop_ratio = drop_ratio
        out_dim = 5 * self.num_bboxes + self.num_classes

        layers = [
            torch.nn.Conv2d(self.in_dim, self.hidden_dim, kernel_size=1),
            torch.nn.Dropout(p=self.drop_ratio),
            torch.nn.LeakyReLU(negative_slope=0.2),

            torch.nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),  # modified
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.BatchNorm2d(self.hidden_dim),

            torch.nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.BatchNorm2d(self.hidden_dim),

            torch.nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.BatchNorm2d(self.hidden_dim),

            torch.nn.Conv2d(self.hidden_dim, out_dim, kernel_size=1),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, features):
        """
        Run the forward pass of the network to predict outputs given features
        from the backbone network.

        Inputs:
        - features: Tensor of shape (B, in_dim, 7, 7) giving image features computed
          by the backbone network.

        Outputs:
        - bbox_xywh: Tensor of shape (B, A, 4, H, W) giving predicted offsets for
          all bounding boxes.
        - conf_scores: Tensor of shape (B, A, H, W) giving predicted classification
          scores for all bounding boxes.
        - cls_scores: Tensor of shape (B, C, H, W) giving classification scores for
          each spatial position.
        """
        bbox_xywh, conf_scores, cls_scores = None, None, None

        ###########################################################################
        # TODO: Implement the forward pass of the PredictionNetwork.              #
        # - Use features to predict bbox_xywh (offsets), conf_scores, and         #
        # class_scores.                                                           #
        # - Make sure conf_scores is between 0 and 1 by squashing the             #
        # network output with a sigmoid.                                          #
        # - The first two elements t^x and t^y of offsets should be between -0.5  #
        # and 0.5. You can achieve this by squashing with a sigmoid               #
        # and subtracting 0.5.                                                    #
        # - The last two elements of bbox_xywh w and h should be normalized,      #
        # i.e. squashed with a sigmoid between 0 and 1.                           #
        #                                                                         #
        # Note: In the 5A+C dimension, the first 5*A would be bounding box        #
        # offsets, and next C will be class scores.                               #
        ###########################################################################
        outputs = self.net(features)
        B, AC, H, W = outputs.shape
        C = self.num_classes
        A = self.num_bboxes

        bbox_xywh_index = torch.tensor([i for i in range(0, 5 * A) if i % 5 != 4], **to_long_cuda)
        bbox_xywh = torch.reshape(torch.index_select(outputs, 1, bbox_xywh_index), (B, A, 4, H, W))
        sub = torch.zeros(bbox_xywh.shape, **to_float_cuda)
        sub[:, :, :2] += 0.5
        bbox_xywh = torch.sigmoid(bbox_xywh).clone() - sub
        conf_score_index = torch.arange(4, 5 * A, 5, **to_long_cuda)
        conf_scores = torch.sigmoid(torch.index_select(outputs, 1, conf_score_index))
        cls_scores = outputs[:, -C:, :, :]
        ###########################################################################
        #                              END OF YOUR CODE                           #
        ###########################################################################

        # You can uncomment these lines when training for a few iterations to
        # check if your offets are within the expected bounds.

        # print("Checking offset bounds in Prediction Network...")
        # assert bbox_xywh[:, :, 0:2].max() <= 0.5 and bbox_xywh[:, :, 0:2].min() >= -0.5, 'invalid offsets (x, y) values'
        # assert bbox_xywh[:, :, 2:4].max() <= 1 and bbox_xywh[:, :, 2:4].min() >= 0, 'invalid offsets (w, h) values'
        # print("Check passed!")

        return bbox_xywh, conf_scores, cls_scores


"""You can uncomment the assert statements above the return in the previous function when training for a few iterations to see if the offsets you predict are within the required bounds. Once you are sure of that, comment them again and continue training.

## Activated (positive) and negative bounding boxes
During training, after we calculate the bbox_xywh (offset) values for all bounding boxes, we need to match the ground-truth boxes against the predicted bounding boxes to determine the classification labels for the bounding boxes -- which boxes should be classified as containing an object and which should be classified as background? Based on this, we will caluclate the 'expected' or 'Ground Truth' targets for offsets and class, which will be used by you to caluclate the loss. We have written this part for you.

Read and digest the input/output definition carefully. You are highly recommended to skim through the code as well, the ground targets are core to training an accurate model.
"""


def ReferenceOnActivatedBboxes(bboxes, gt_bboxes, grid, iou_mat, pos_thresh=0.7, neg_thresh=0.3):
    """
    Determine the activated (positive) and negative bboxes for model training.

    A grid cell is responsible for predicting a GT box if the center of
    the box falls into that cell.
    Implementation details: First compute manhattan distance between grid cell centers
    (BxH’xW’) and GT box centers (BxN). This gives us a matrix of shape Bx(H'xW')xN and
    perform torch.min(dim=1)[1] on it gives us the indexes indicating activated grids
    responsible for GT boxes (convert to x and y). Among all the bboxes associated with
    the activate grids, the bbox with the largest IoU with the GT box is responsible to
    predict (regress to) the GT box.
    Note: One bbox might match multiple GT boxes.

    Main steps include:
    i) Decide activated and negative bboxes based on the IoU matrix.
    ii) Compute GT confidence score/offsets/object class on the positive proposals.
    iii) Compute GT confidence score on the negative proposals.

    Inputs:
    - bboxes: Bounding boxes, of shape BxAxH’xW’x4
    - gt_bboxes: GT boxes of shape BxNx5, where N is the number of PADDED GT boxes,
              5 indicates (x_{lr}^{gt}, y_{lr}^{gt}, x_{rb}^{gt}, y_{rb}^{gt}) and class index
    - grid (float): A cell grid of shape BxH'xW'x2 where 2 indicate the (x, y) coord
    - iou_mat: IoU matrix of shape Bx(AxH’xW’)xN
    - pos_thresh: Positive threshold value
    - neg_thresh: Negative threshold value

    Outputs:
    - activated_anc_ind: Index on activated bboxes, of shape M, where M indicates the
                         number of activated bboxes
    - negative_anc_ind: Index on negative bboxes, of shape M
    - GT_conf_scores: GT IoU confidence scores on activated bboxes, of shape M
    - GT_offsets: GT offsets on activated bboxes, of shape Mx4. They are denoted as
                  \hat{t^x}, \hat{t^y}, \hat{t^w}, \hat{t^h} in the formulation earlier.
    - GT_class: GT class category on activated bboxes, essentially indexed from gt_bboxes[:, :, 4],
                of shape M
    - activated_anc_coord: Coordinates on activated bboxes (mainly for visualization purposes)
    - negative_anc_coord: Coordinates on negative bboxes (mainly for visualization purposes)
    """

    B, A, h_amap, w_amap, _ = bboxes.shape
    N = gt_bboxes.shape[1]

    # activated/positive bboxes
    max_iou_per_anc, max_iou_per_anc_ind = iou_mat.max(dim=-1)

    bbox_mask = (gt_bboxes[:, :, 0] != -1)  # BxN, indicate invalid boxes
    bbox_centers = (gt_bboxes[:, :, 2:4] - gt_bboxes[:, :, :2]) / 2. + gt_bboxes[:, :, :2]  # BxNx2

    mah_dist = torch.abs(grid.view(B, -1, 2).unsqueeze(2) - bbox_centers.unsqueeze(1)).sum(dim=-1)  # Bx(H'xW')xN
    min_mah_dist = mah_dist.min(dim=1, keepdim=True)[0]  # Bx1xN
    grid_mask = (mah_dist == min_mah_dist).unsqueeze(1)  # Bx1x(H'xW')xN

    reshaped_iou_mat = iou_mat.view(B, A, -1, N)
    anc_with_largest_iou = reshaped_iou_mat.max(dim=1, keepdim=True)[0]  # Bx1x(H’xW’)xN
    anc_mask = (anc_with_largest_iou == reshaped_iou_mat)  # BxAx(H’xW’)xN
    activated_anc_mask = (grid_mask & anc_mask).view(B, -1, N)
    activated_anc_mask &= bbox_mask.unsqueeze(1)

    # one bbox could match multiple GT boxes
    activated_anc_ind = torch.nonzero(activated_anc_mask.view(-1)).squeeze(-1)
    GT_conf_scores = iou_mat.view(-1)[activated_anc_ind]
    gt_bboxes = gt_bboxes.view(B, 1, N, 5).repeat(1, A * h_amap * w_amap, 1, 1).view(-1, 5)[activated_anc_ind]
    GT_class = gt_bboxes[:, 4].long()
    gt_bboxes = gt_bboxes[:, :4]
    activated_anc_ind = (activated_anc_ind.float() / activated_anc_mask.shape[-1]).long()

    print('number of pos proposals: ', activated_anc_ind.shape[0])

    activated_anc_coord = bboxes.reshape(-1, 4)[activated_anc_ind]

    activated_grid_coord = grid.repeat(1, A, 1, 1, 1).reshape(-1, 2)[activated_anc_ind]

    # GT offsets

    # bbox are x_tl, y_tl, x_br, y_br
    # offsets are t_x, t_y, t_w, t_h

    # Grid: (B, H, W, 2) -> This will be used to calculate center offsets
    # w, h offsets are not offsets but normalized w,h themselves.

    wh_offsets = torch.sqrt((gt_bboxes[:, 2:4] - gt_bboxes[:, :2]) / 7.)
    assert torch.max(
        (gt_bboxes[:, 2:4] - gt_bboxes[:, :2]) / 7.) <= 1, "w and h targets not normalised, should be between 0 and 1"

    xy_offsets = (gt_bboxes[:, :2] + gt_bboxes[:, 2:4]) / (2.) - activated_grid_coord

    assert torch.max(torch.abs(xy_offsets)) <= 0.5, \
        "x and y offsets should be between -0.5 and 0.5! Got {}".format( \
            torch.max(torch.abs(xy_offsets)))

    GT_offsets = torch.cat((xy_offsets, wh_offsets), dim=-1)

    # negative bboxes
    negative_anc_mask = (max_iou_per_anc < neg_thresh)  # Bx(AxH’xW’)
    negative_anc_ind = torch.nonzero(negative_anc_mask.view(-1)).squeeze(-1)
    negative_anc_ind = negative_anc_ind[torch.randint(0, negative_anc_ind.shape[0], (activated_anc_ind.shape[0],))]
    negative_anc_coord = bboxes.reshape(-1, 4)[negative_anc_ind.view(-1)]

    # activated_anc_coord and negative_anc_coord are mainly for visualization purposes
    return activated_anc_ind, negative_anc_ind, GT_conf_scores, GT_offsets, GT_class, \
           activated_anc_coord, negative_anc_coord


"""## (e) Loss Function
The confidence score regression loss is for both activated/negative bounding boxes / proposals while the bounding box regression loss and the object classification loss are for activated bounding boxes only. These are implemented for you. Please go through and understand their inputs and outputs carefully as they are key to the implementation of the forward pass of the object detection module.

### Confidence score regression
"""


def ConfScoreRegression(conf_scores, GT_conf_scores):
    """
    Use sum-squared error as in YOLO

    Inputs:
    - conf_scores: Predicted confidence scores
    - GT_conf_scores: GT confidence scores

    Outputs:
    - conf_score_loss
    """
    # the target conf_scores for negative samples are zeros
    GT_conf_scores = torch.cat((torch.ones_like(GT_conf_scores), \
                                torch.zeros_like(GT_conf_scores)), dim=0).view(-1, 1)

    conf_score_loss = torch.sum((conf_scores - GT_conf_scores) ** 2) * 1. / GT_conf_scores.shape[0]
    return conf_score_loss


"""### Bounding box regression"""


def BboxRegression(offsets, GT_offsets):
    """"
    Use sum-squared error as in YOLO
    For both xy and wh.
    NOTE: In YOLOv1, the authors use sqrt(w) and sqrt(h) for normalized w and h
    (read paper for more details) and thus both offsets and GT_offsets will
    be having (x, y, sqrt(w), sqrt(h)) parametrization of the coodinates.


    Inputs:
    - offsets: Predicted box offsets
    - GT_offsets: GT box offsets

    Outputs:
    - bbox_reg_loss
    """

    bbox_reg_loss = torch.sum((offsets - GT_offsets) ** 2) * 1. / GT_offsets.shape[0]
    return bbox_reg_loss


"""### Object classification"""


def ObjectClassification(class_prob, GT_class, batch_size, anc_per_img, activated_anc_ind):
    """"
    Use softmax loss

    Inputs:
    - class_prob: Predicted class logits
    - GT_class: GT box class label
    - batch_size: the batch size to compute loss over
    - anc_per_img: anchor indices for each image
    - activated_anc_ind: indices for positive anchors

    Outputs:
    - object_cls_loss, the classification loss for object detection
    """
    # average within sample and then average across batch
    # such that the class pred would not bias towards dense popular objects like `person`

    all_loss = F.cross_entropy(class_prob, GT_class, reduction='none')  # , reduction='sum') * 1. / batch_size
    object_cls_loss = 0
    for idx in range(batch_size):
        anc_ind_in_img = (activated_anc_ind >= idx * anc_per_img) & (activated_anc_ind < (idx + 1) * anc_per_img)
        object_cls_loss += all_loss[anc_ind_in_img].sum() * 1. / torch.sum(anc_ind_in_img)
    object_cls_loss /= batch_size
    # object_cls_loss = F.cross_entropy(class_prob, GT_class, reduction='sum') * 1. / batch_size

    return object_cls_loss


"""# (f) Object Detector Code

## (f) TODO: Object detection module

We will now combine everything into the `SingleStageDetector` class:

We have implemented the `forward` function of the detector for you. This implements the training-time forward pass: it receives the input images and the ground-truth bounding boxes, and returns the total loss for the minibatch.

Below are the key steps in implementing the forward pass:   


```
i)   Image feature extraction using Detector Backbone Network  
ii)  Grid List generation using Grid Generator  
iii) Compute offsets, conf_scores, cls_scores through the Prediction Network.  
iv)  Calculate the proposals or actual bounding boxes using offsets stored in 
     offsets and grid_list by passing these into the GenerateProposal function 
     you wrote earlier.  
v)   Compute IoU between grid_list and proposals and then determine activated
     negative bounding boxes/proposals, and GT_conf_scores, GT_offsets GT_class   
     using the ReferenceOnActivatedBboxes function.
vi)  The loss function for BboxRegression which expects the parametrization as   
     (x, y, sqrt(w), sqrt(h)) for the offsets and the GT_offsets also have  
     sqrt(w), sqrt(h) in the offsets as part of GT_offsets. So before the next step,   
     convert the bbox_xywh parametrization form (x, y, w, h) to (x, y, sqrt(w), sqrt(h)).   
vii) Extract the confidence scores corresponding to the positive and negative   
     activated bounding box indices, classification scores for positive box indices,   
     offsets using positive box indices.  HINT: Set `neg_thresh=0.2` in ReferenceOnActivatedBboxes.
     HINT: You can use the provided helper methods self._extract_bbox_data and   
     self._extract_class_scores to extract information for positive and   
     negative bounding boxes / proposals specified by activated_anc_ind and negative_anc_ind.
viii) Compute the total_loss which is formulated as:   
      total_loss = w_conf * conf_loss + w_reg * reg_loss + w_cls * cls_loss,   
      where conf_loss is determined by ConfScoreRegression, w_reg by BboxRegression,  
      and w_cls by ObjectClassification. 
```
"""


class SingleStageDetector(nn.Module):
    def __init__(self):
        super().__init__()

        self.feat_extractor = FeatureExtractor()
        self.num_classes = 20
        self.num_bboxes = 2
        # self.pred_network = PredictionNetwork(1280, num_bboxes=2, \
        #                                       num_classes=self.num_classes)

        self.pred_network = PredictionNetwork(576, num_bboxes=2, \
                                              num_classes=self.num_classes)  # modified

    def forward(self, images, bboxes):
        """
        Training-time forward pass for the single-stage detector.

        Inputs:
        - images: Input images, of shape (B, 3, 224, 224)
        - bboxes: GT bounding boxes of shape (B, N, 5) (padded)

        Outputs:
        - total_loss: Torch scalar giving the total loss for the batch.
        """
        # weights to multiple to each loss term
        w_conf = 1  # for conf_scores
        w_reg = 1  # for offsets
        w_cls = 1  # for class_prob

        total_loss = None

        # 1. Feature extraction
        features = self.feat_extractor(images)

        # 2. Grid generator
        grid_list = GenerateGrid(images.shape[0])

        # 3. Prediction Network
        bbox_xywh, conf_scores, cls_scores = self.pred_network(features)
        # (B, A, 4, H, W), (B, A, H, W), (B, C, H, W)

        B, A, _, H, W = bbox_xywh.shape
        bbox_xywh = bbox_xywh.permute(0, 1, 3, 4, 2)  # (B, A, H, W, 4)

        assert bbox_xywh.max() <= 1 and bbox_xywh.min() >= -0.5, 'invalid offsets values'

        # 4. Calculate the proposals
        proposals = GenerateProposal(grid_list, bbox_xywh)

        # 5. Compute IoU
        iou_mat = IoU(proposals, bboxes)

        # 7. Using the activated_anc_ind, select the activated conf_scores, bbox_xywh, cls_scores
        activated_anc_ind, negative_anc_ind, GT_conf_scores, GT_offsets, GT_class, _, _ \
            = ReferenceOnActivatedBboxes(bbox_xywh, bboxes, grid_list, iou_mat, neg_thresh=0.3)

        conf_scores = conf_scores.view(B, A, 1, H, W)
        pos = self._extract_bbox_data(conf_scores, activated_anc_ind)
        neg = self._extract_bbox_data(conf_scores, negative_anc_ind)
        conf_scores = torch.cat([pos, neg], dim=0)

        # 6. The loss function
        bbox_xywh[:, :, :, :, 2:4] = torch.sqrt(bbox_xywh[:, :, :, :, 2:4])

        # assert bbox_xywh[:, :, :, :, :2].max() <= 0.5 and bbox_xywh[:, :, :, :, :2].min() >= -0.5, 'invalid offsets values'
        # assert bbox_xywh[:, :, :, :, :2:4].max() <= 1 and bbox_xywh[:, :, :, :, 2:4].min() >= 0, 'invalid offsets values'

        offsets = self._extract_bbox_data(bbox_xywh.permute(0, 1, 4, 2, 3), activated_anc_ind)
        cls_scores = self._extract_class_scores(cls_scores, activated_anc_ind)
        anc_per_img = torch.prod(torch.tensor(bbox_xywh.shape[1:-1]))  # use as argument in ObjectClassification
        ###########################################################################
        # TODO: Compute conf_loss, reg_loss, cls_loss, total_loss using the       #
        # functions defined in part (e).                                           #
        # - total_loss is the sum of the three other losses.                      #
        ###########################################################################
        # 8. Compute losses
        conf_loss = ConfScoreRegression(conf_scores, GT_conf_scores)
        reg_loss = BboxRegression(offsets, GT_offsets)
        cls_loss = ObjectClassification(cls_scores, GT_class, B, anc_per_img, activated_anc_ind)
        total_loss = conf_loss + reg_loss + cls_loss
        ###########################################################################
        #                            END OF YOUR CODE                             #
        ###########################################################################

        print('(weighted) conf loss: {:.4f}, reg loss: {:.4f}, cls loss: {:.4f}'.format(conf_loss, reg_loss, cls_loss))

        return total_loss

    def inference(self):
        raise NotImplementedError

    def combinedInference(self):
        raise NotImplementedError

    def _extract_bbox_data(self, bbox_data, bbox_idx):
        """
        Inputs:
        - bbox_data: Tensor of shape (B, A, D, H, W) giving a vector of length
          D for each of A bboxes at each point in an H x W grid.
        - bbox_idx: int64 Tensor of shape (M,) giving bbox indices to extract

        Returns:
        - extracted_bboxes: Tensor of shape (M, D) giving bbox data for each
          of the bboxes specified by bbox_idx.
        """
        B, A, D, H, W = bbox_data.shape
        bbox_data = bbox_data.permute(0, 1, 3, 4, 2).contiguous().view(-1, D)
        extracted_bboxes = bbox_data[bbox_idx]
        return extracted_bboxes

    def _extract_class_scores(self, all_scores, bbox_idx):
        """
        Inputs:
        - all_scores: Tensor of shape (B, C, H, W) giving classification scores for
          C classes at each point in an H x W grid.
        - bbox_idx: int64 Tensor of shape (M,) giving the indices of bboxes at
          which to extract classification scores

        Returns:
        - extracted_scores: Tensor of shape (M, C) giving the classification scores
          for each of the bboxes specified by bbox_idx.
        """
        B, C, H, W = all_scores.shape
        A = self.num_bboxes
        all_scores = all_scores.contiguous().permute(0, 2, 3, 1).contiguous()
        all_scores = all_scores.view(B, 1, H, W, C).expand(B, A, H, W, C)
        all_scores = all_scores.reshape(B * A * H * W, C)
        extracted_scores = all_scores[bbox_idx]
        return extracted_scores


"""## Object detection solver
The `DetectionSolver` object runs the training loop to train an single stage detector.
"""


def DetectionSolver(detector, train_loader, learning_rate=3e-3,
                    lr_decay=1, num_epochs=20, **kwargs):
    """
    Run optimization to train the model.
    """

    # ship model to GPU
    detector.to(**to_float_cuda)

    # optimizer setup
    from torch import optim
    # optimizer = optim.Adam(
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, detector.parameters()),
        learning_rate)  # leave betas and eps by default
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                               lambda epoch: lr_decay ** epoch)

    # sample minibatch data
    loss_history = []
    detector.train()
    for i in range(num_epochs):
        start_t = time.time()
        for iter_num, data_batch in enumerate(train_loader):
            images, boxes, w_batch, h_batch, _ = data_batch
            resized_boxes = coord_trans(boxes, w_batch, h_batch, mode='p2a')
            # print(resized_boxes)
            images = images.to(**to_float_cuda)
            resized_boxes = resized_boxes.to(**to_float_cuda)

            loss = detector(images, resized_boxes)
            optimizer.zero_grad()
            loss.backward()
            loss_history.append(loss.item())
            optimizer.step()

            print('(Iter {} / {})'.format(iter_num, len(train_loader)))

        end_t = time.time()
        print('(Epoch {} / {}) loss: {:.4f} time per epoch: {:.1f}s'.format(
            i, num_epochs, loss.item(), end_t - start_t))
        print('\n\n\n')

        lr_scheduler.step()

    # plot the training losses
    plt.plot(loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training loss history')
    plt.show()

    return loss_history


"""# (g) Train the Object Detector


## (h) TODO: Non-Maximum Suppression (NMS)
The definition of NMS and instructions on how to compute NMS can be found in the [lecture slides](https://www.eecs.umich.edu/courses/eecs442-ahowens/fa20/lec/lec12-object.pdf) (p47-48)
"""


def nms(boxes, scores, iou_threshold=0.5, topk=None):
    """
    Non-maximum suppression removes overlapping bounding boxes.

    Inputs:
    - boxes: top-left and bottom-right coordinate values of the bounding boxes
      to perform NMS on, of shape Nx4
    - scores: scores for each one of the boxes, of shape N
    - iou_threshold: discards all overlapping boxes with IoU > iou_threshold; float
    - topk: If this is not None, then return only the topk highest-scoring boxes.
      Otherwise if this is None, then return all boxes that pass NMS.

    Outputs:
    - keep: torch.long tensor with the indices of the elements that have been
      kept by NMS, sorted in decreasing order of scores; of shape [num_kept_boxes]
    """

    if (not boxes.numel()) or (not scores.numel()):
        return torch.zeros(0, dtype=torch.long)

    keep = None

    def cal_iou(prev, new):
        xtl = prev[:, 0]
        ytl = prev[:, 1]
        xbr = prev[:, 2]
        ybr = prev[:, 3]
        x_tl = new[:, 0]
        y_tl = new[:, 1]
        x_br = new[:, 2]
        y_br = new[:, 3]
        inter_xtl = torch.max(xtl, x_tl)
        inter_ytl = torch.max(ytl, y_tl)
        inter_xbr = torch.min(xbr, x_br)
        inter_ybr = torch.min(ybr, y_br)
        zero = torch.zeros(inter_xtl.shape)
        intersection = torch.max(zero, inter_xbr - inter_xtl) * torch.max(zero, inter_ybr - inter_ytl)
        union = abs((x_br - x_tl) * (y_tl - y_br)) + abs((xbr - xtl) * (ytl - ybr)) - intersection
        return intersection / union

    sort = torch.argsort(scores, descending=True)
    for i in sort:
        i = torch.unsqueeze(i, 0)
        if keep == None: keep = torch.Tensor([i[0].item()]).long()
        prev = torch.index_select(boxes, 0, keep)
        iou = cal_iou(prev, torch.index_select(boxes, 0, i))
        if torch.sum(iou > iou_threshold) > 0: continue
        keep = torch.cat((keep, i))
        if topk != None and keep.size(0) >= topk: break
    #############################################################################
    #                               END OF YOUR CODE                            #
    #############################################################################
    return keep


"""## (i) Inference

Now, implement the inference part of module `SingleStageDetector`.
"""


def detector_inference(self, images, thresh=0.5, nms_thresh=0.7):
    """"
    Inference-time forward pass for the single stage detector.

    Inputs:
    - images: Input images
    - thresh: Threshold value on confidence scores
    - nms_thresh: Threshold value on NMS

    Outputs:
    - final_propsals: Keeped proposals after confidence score thresholding and NMS,
                      a list of B (*x4) tensors
    - final_conf_scores: Corresponding confidence scores, a list of B (*x1) tensors
    - final_class: Corresponding class predictions, a list of B  (*x1) tensors
    """
    final_proposals, final_conf_scores, final_class = [], [], []

    # Predicting the final proposal coordinates `final_proposals`,         
    # confidence scores `final_conf_scores`, and the class index `final_class`.  

    # The overall steps are similar to the forward pass but now we do not need  
    # to decide the activated nor negative bounding boxes.                         
    # We threshold the conf_scores based on the threshold value `thresh`.  
    # Then, apply NMS (torchvision.ops.nms) to the filtered proposals given the  
    # threshold `nms_thresh`.                                                    
    # The class index is determined by the class with the maximal probability.   
    # Note that `final_propsals`, `final_conf_scores`, and `final_class` are all 
    # lists of B 2-D tensors.

    with torch.no_grad():
        # Feature extraction
        features = self.feat_extractor(images)
        features1 = feat_ext(images)
        x = torch.ones(1, 3, 224, 224, device=torch.device("cuda"))
        y1 = self.feat_extractor(x)
        y2 = feat_ext(x)
        print("check same", (y1 - y2).sum())
        print("feat same", (features1 - features).sum())

        # Grid  Generator
        grid_list = GenerateGrid(images.shape[0])

        # Prediction Network
        offsets, conf_scores, class_scores = self.pred_network(features)
        B, A, _, w_amap, h_amap = offsets.shape  # B, A, 4, H, W
        C = self.num_classes
        conf_scores = conf_scores.view(B, -1)  # B, A*H*W
        offsets = offsets.permute(0, 1, 3, 4, 2)  # B, A, H, W, 4
        class_scores = class_scores.permute(0, 2, 3, 1).reshape(B, -1, C)  # B, H*W, C

        most_conf_class_score, most_conf_class_idx = class_scores.max(dim=-1)

        # Proposal generator
        proposals = GenerateProposal(grid_list, offsets).view(B, -1, 4)  # Bx(AxH'xW')x4

        # Thresholding and NMS
        for i in range(B):
            score_mask = torch.nonzero((conf_scores[i] > thresh)).squeeze(1)  # (AxH'xW')
            prop_before_nms = proposals[i, score_mask]
            scores_before_nms = conf_scores[i, score_mask]
            class_idx_before_nms = most_conf_class_idx[i, score_mask % (h_amap * w_amap)]
            # class_prob_before_nms = most_conf_class_prob[i, score_mask/A]

            prop_keep = torchvision.ops.nms(prop_before_nms, scores_before_nms, nms_thresh).to(images.device)
            final_proposals.append(prop_before_nms[prop_keep])
            final_conf_scores.append(scores_before_nms[prop_keep].unsqueeze(-1))
            final_class.append(class_idx_before_nms[prop_keep].unsqueeze(-1))

    return final_proposals, final_conf_scores, final_class


SingleStageDetector.inference = detector_inference


def combined_detector_inference(self, images, feat, thresh=0.8, nms_thresh=0.05):
    """"
    Inference-time forward pass for the single stage detector.

    Inputs:
    - images: Input images
    - thresh: Threshold value on confidence scores
    - nms_thresh: Threshold value on NMS

    Outputs:
    - final_propsals: Keeped proposals after confidence score thresholding and NMS,
                      a list of B (*x4) tensors
    - final_conf_scores: Corresponding confidence scores, a list of B (*x1) tensors
    - final_class: Corresponding class predictions, a list of B  (*x1) tensors
    """
    final_proposals, final_conf_scores, final_class = [], [], []

    # Predicting the final proposal coordinates `final_proposals`,
    # confidence scores `final_conf_scores`, and the class index `final_class`.

    # The overall steps are similar to the forward pass but now we do not need
    # to decide the activated nor negative bounding boxes.
    # We threshold the conf_scores based on the threshold value `thresh`.
    # Then, apply NMS (torchvision.ops.nms) to the filtered proposals given the
    # threshold `nms_thresh`.
    # The class index is determined by the class with the maximal probability.
    # Note that `final_propsals`, `final_conf_scores`, and `final_class` are all
    # lists of B 2-D tensors.

    with torch.no_grad():
        # Feature extraction
        # features = self.feat_extractor(images)
        features = feat

        # Grid  Generator
        grid_list = GenerateGrid(images.shape[0])

        # Prediction Network
        offsets, conf_scores, class_scores = self.pred_network(features)
        B, A, _, w_amap, h_amap = offsets.shape  # B, A, 4, H, W
        C = self.num_classes
        conf_scores = conf_scores.view(B, -1)  # B, A*H*W
        offsets = offsets.permute(0, 1, 3, 4, 2)  # B, A, H, W, 4
        class_scores = class_scores.permute(0, 2, 3, 1).reshape(B, -1, C)  # B, H*W, C

        most_conf_class_score, most_conf_class_idx = class_scores.max(dim=-1)

        # Proposal generator
        proposals = GenerateProposal(grid_list, offsets).view(B, -1, 4)  # Bx(AxH'xW')x4

        # Thresholding and NMS
        for i in range(B):
            score_mask = torch.nonzero((conf_scores[i] > thresh)).squeeze(1)  # (AxH'xW')
            prop_before_nms = proposals[i, score_mask]
            scores_before_nms = conf_scores[i, score_mask]
            class_idx_before_nms = most_conf_class_idx[i, score_mask % (h_amap * w_amap)]
            # class_prob_before_nms = most_conf_class_prob[i, score_mask/A]

            prop_keep = torchvision.ops.nms(prop_before_nms, scores_before_nms, nms_thresh).to(images.device)
            final_proposals.append(prop_before_nms[prop_keep])
            final_conf_scores.append(scores_before_nms[prop_keep].unsqueeze(-1))
            final_class.append(class_idx_before_nms[prop_keep].unsqueeze(-1))

    return final_proposals, final_conf_scores, final_class


SingleStageDetector.combinedInference = combined_detector_inference


def DetectionInference(detector, data_loader, dataset, idx_to_class, thresh=0.8, nms_thresh=0.3, output_dir=None):
    # ship model to GPU
    detector.to(**to_float_cuda)

    detector.eval()
    start_t = time.time()

    if output_dir is not None:
        det_dir = 'mAP/input/detection-results'
        gt_dir = 'mAP/input/ground-truth'
        if os.path.exists(det_dir):
            shutil.rmtree(det_dir)
        os.mkdir(det_dir)
        if os.path.exists(gt_dir):
            shutil.rmtree(gt_dir)
        os.mkdir(gt_dir)

    for iter_num, data_batch in enumerate(data_loader):
        images, boxes, w_batch, h_batch, img_ids = data_batch
        images = images.to(**to_float_cuda)

        final_proposals, final_conf_scores, final_class = detector.inference(images, thresh=thresh,
                                                                             nms_thresh=nms_thresh)

        # clamp on the proposal coordinates
        batch_size = len(images)
        for idx in range(batch_size):
            torch.clamp_(final_proposals[idx][:, 0::2], min=0, max=w_batch[idx])
            torch.clamp_(final_proposals[idx][:, 1::2], min=0, max=h_batch[idx])

            # visualization
            # get the original image
            # hack to get the original image so we don't have to load from local again...
            i = batch_size * iter_num + idx
            img, _ = dataset.__getitem__(i)

            valid_box = sum([1 if j != -1 else 0 for j in boxes[idx][:, 0]])
            final_all = torch.cat((final_proposals[idx], \
                                   final_class[idx].float(), final_conf_scores[idx]), dim=-1).cpu()
            resized_proposals = coord_trans(final_all, w_batch[idx], h_batch[idx])

            # drive.mount('/content/drive')
            # write results to file for evaluation (use mAP API https://github.com/Cartucho/mAP for now...)
            if output_dir is not None:
                file_name = img_ids[idx].replace('.jpg', '.txt')
                with open(os.path.join(det_dir, file_name), 'w') as f_det, \
                        open(os.path.join(gt_dir, file_name), 'w') as f_gt:
                    print(
                        '{}: {} GT bboxes and {} proposals'.format(img_ids[idx], valid_box, resized_proposals.shape[0]))
                    for b in boxes[idx][:valid_box]:
                        f_gt.write(
                            '{} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(idx_to_class[b[4].item()], b[0], b[1], b[2],
                                                                      b[3]))
                    for b in resized_proposals:
                        f_det.write(
                            '{} {:.6f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(idx_to_class[b[4].item()], b[5], b[0],
                                                                             b[1], b[2], b[3]))
            else:
                data_visualizer(img, idx_to_class, boxes[idx][:valid_box], resized_proposals)

    end_t = time.time()
    print('Total inference time: {:.1f}s'.format(end_t - start_t))


def SingleDetectionInference(detector, img, img2draw=None, feat=None, idx_to_class=idx_to_class, thresh=0.8,
                             nms_thresh=0.05):
    # ship model to GPU
    detector.to(**to_float_cuda)

    detector.eval()
    start_t = time.time()

    img_h, img_w, _ = img.shape

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print(type(img))
    images = preprocess(img).unsqueeze(0)

    images = images.to(**to_float_cuda)

    if feat is None:
        final_proposals, final_conf_scores, final_class = detector.inference(images, thresh=thresh,
                                                                             nms_thresh=nms_thresh)
    else:
        final_proposals, final_conf_scores, final_class = detector.combinedInference(images, feat, thresh=thresh,
                                                                                     nms_thresh=nms_thresh)

    # clamp on the proposal coordinates
    torch.clamp_(final_proposals[0][:, 0::2], min=0, max=img_w)
    torch.clamp_(final_proposals[0][:, 1::2], min=0, max=img_h)

    # valid_box = sum([1 if j != -1 else 0 for j in boxes[:, 0]])
    final_all = torch.cat((final_proposals[0], \
                           final_class[0].float(), final_conf_scores[0]), dim=-1).cpu()
    resized_proposals = coord_trans(final_all, torch.tensor([img_w]), torch.tensor([img_h]))

    if img2draw is None:
        img = np.array(img)
    else:
        img = img2draw
    img = cv2.resize(img, (img_w, img_h))
    print(img.shape, idx_to_class, resized_proposals.shape)
    data_visualizer(img, idx_to_class, None, resized_proposals)

    end_t = time.time()
    print('Total inference time: {:.1f}s'.format(end_t - start_t))


def yoloOutput2ImgWrapper(detector, img, idx_to_class=idx_to_class, thresh=0.8, nms_thresh=0.05):
    # ship model to GPU
    detector.to(**to_float_cuda)

    detector.eval()
    start_t = time.time()

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    images = preprocess(img).unsqueeze(0)

    images = images.to(**to_float_cuda)

    yolo_output = detector.inference(images, thresh=thresh, nms_thresh=nms_thresh)
    yoloOutput2Img(yolo_output, img)


print("finished")


def yoloOutput2Img(yolo_output, img, idx_to_class=idx_to_class):
    final_proposals, final_conf_scores, final_class = yolo_output
    img_h, img_w, _ = img.shape
    # clamp on the proposal coordinates
    torch.clamp_(final_proposals[0][:, 0::2], min=0, max=img_w)
    torch.clamp_(final_proposals[0][:, 1::2], min=0, max=img_h)

    # valid_box = sum([1 if j != -1 else 0 for j in boxes[:, 0]])
    final_all = torch.cat((final_proposals[0], \
                           final_class[0].float(), final_conf_scores[0]), dim=-1).cpu()
    resized_proposals = coord_trans(final_all, torch.tensor([img_w]), torch.tensor([img_h]))

    img = cv2.resize(img, (img_w, img_h))
    data_visualizer(img, idx_to_class, None, resized_proposals)


if __name__ == "__main__":
    detector_dict = torch.load("../train-history/yolo_detector.pt")
    detector = SingleStageDetector()
    detector.load_state_dict(detector_dict)
    from PIL import Image

    img = Image.open("../example2.png").convert('RGB')
    img = np.array(img)
    yoloOutput2ImgWrapper(detector, img)
