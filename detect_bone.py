import cv2
import torch
from torch import nn
from torchvision import models
from data_utils.tools import trans_square
from bone_filter_utils import bone_filter
import os
from PIL import Image
import numpy as np
import config
from torchvision.transforms import Compose, ToTensor, Resize, InterpolationMode

data_transform = Compose([
    Resize(size=(224, 224), interpolation=InterpolationMode.NEAREST),
    ToTensor()
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    # 1.加载yolov5模型
    yolov5_model = torch.hub.load(r"D:\projects\yolov5-master", "custom",
                                  r"D:\projects\hand_test\best.pt", source="local")
    yolov5_model.conf = 0.7
    yolov5_model.eval()
    # 2.加载9个分类模型
    cls_models = {}
    for i, name in enumerate(config.CATEGORTY):
        model_name = config.arthrosis[name][0]
        if model_name in cls_models:
            continue

        net = models.resnet18()
        net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        net.fc = nn.Linear(512, config.arthrosis[name][1], bias=True)
        net.load_state_dict(torch.load("params/{}_best.pth".format(model_name), map_location=DEVICE))
        net.eval()
        cls_models[model_name] = net
    print("加载模型成功")
    return yolov5_model, cls_models


def detect_img(yolov5_model, cls_models, img_path, sex):
    print("开始检测！")
    info = ''
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(tileGridSize=(5, 5))
    dst1 = clahe.apply(gray)
    results = yolov5_model(dst1)
    # print(results)
    out = results.xyxy[0]
    # print(out)
    out = bone_filter(out)
    out = out.cpu().numpy()
    # print(out)

    score_all = 0
    bone_results = {}
    for i, name in enumerate(config.CATEGORTY):
        x1 = int(out[i][0])
        y1 = int(out[i][1])
        x2 = int(out[i][2])
        y2 = int(out[i][3])
        # 裁剪关节并保存
        img_ori = img[y1:y2, x1:x2]
        if not os.path.exists("captures"):
            os.makedirs("captures")
        save_path = "./captures/{}.png".format(name)
        cv2.imwrite(save_path, img_ori)

        # 把每个关节分别传到对应的分类模型中侦测，得到每个关节的等级
        im = Image.open(save_path)
        im = trans_square(im)
        im = im.convert("L")

        im = np.array(im)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
        im = clahe.apply(im)
        im = Image.fromarray(im)
        im = data_transform(im)

        # CHW --> N CHW
        im = torch.unsqueeze(im, dim=0)
        cls_net = cls_models[config.arthrosis[name][0]]
        cls_out = cls_net(im)
        # 获取关节分类等级
        bone_index = int(cls_out.argmax(dim=1))
        score = config.SCORE[sex][name][bone_index]
        score_all += score
        # print(name+"的等级是："+str(bone_index+1))
        info += (f"{name}的等级是：{str(bone_index + 1)}"+'\n')
    s = f"骨龄：{config.calcBoneAge(score_all, sex)}"
    info += s
    return str(info)


if __name__ == '__main__':
    yolov5_model, cls_models = load_model()
    info = detect_img(yolov5_model, cls_models, "images/1525.png", "girl")
    print(info)
