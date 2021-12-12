import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision



m = torch.load("train-history/trained_model74.pth")
m = m.cpu()
img = torch.ones(1, 3, 320, 320)
feat = torch.ones(1, 576, 7, 7)
y = m.combinedInference(img, feat)