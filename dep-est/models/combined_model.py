import time

from PIL import Image

from yolov1 import *
from unet import *
from model_utils import *


class TriTaskModel(nn.Module):
    def __init__(self, det_model, regseg_model):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.det_head = det_model
        self.regseg_head = regseg_model
        self.preprocess = transforms.Compose([transforms.ToTensor(),
                                              transforms.Resize((224, 224)),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])

    def forward(self, np_img):
        self.feature_extractor.eval()
        self.det_head.eval()
        self.regseg_head.eval()
        with torch.no_grad():
            img = self.preprocess(np_img).unsqueeze(0)
            img = img.cpu()
            feat = self.feature_extractor(img)
            yolo_output = self.det_head.combinedInference(img, feat, thresh=0.6, nms_thresh=0.005)
            regseg_output = self.regseg_head.combinedInference(img, feat)
            img_arr = regSegOutput2Img(regseg_output, np_img)
            yoloOutput2Img(yolo_output, img_arr)


def getTriTaskModel(det_path, regseg_path):
    det_model = SingleStageDetector()
    det_dict = torch.load(det_path)
    det_model.load_state_dict(det_dict)
    det_model = det_model.cpu()

    regseg_model = DualTaskUNet()
    regseg_model_dict = torch.load(regseg_path)
    regseg_model.load_state_dict(regseg_model_dict)
    regseg_model = regseg_model.cuda()

    model = TriTaskModel(det_model, regseg_model).cpu()
    return model


def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')


if __name__ == "__main__":
    import os

    model = getTriTaskModel("trained-models/yolo_detector3.pt", "trained-models/report.pth")
    img = Image.open("images/example1.png").convert('RGB')
    img = np.array(img)
    tic = time.time()
    model(img)
    toc = time.time()
    print("once takes ", toc - tic)
    print(1 / (toc - tic),  "fps")
