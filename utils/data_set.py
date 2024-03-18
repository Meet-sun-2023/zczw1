# 创建Dataset
# 123

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage, InterpolationMode
from PIL import Image
from utils.tools import trans_square


data_transforms = Compose([
    Resize(size=(224, 224), interpolation=InterpolationMode.NEAREST),
    ToTensor()
])


class MyData(Dataset):
    def __init__(self, file_path, mode):
        self.data_list = list()
        self.mode = mode
        if self.mode == 'train':
            file_path = os.path.join(file_path, "train.txt")
        elif self.mode == 'val':
            file_path = os.path.join(file_path, "val.txt")

        with open(file_path, 'r') as f:
            for line in f:
                items = line.strip().split()
                image_path = items[0]
                label = int(items[1])
                self.data_list.append([image_path, label])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_path, label = self.data_list[idx]
        im = Image.open(image_path)
        im = trans_square(im)
        im = im.convert('L')
        im = data_transforms(im)
        label = torch.tensor([label]).squeeze()
        return im, label


if __name__ == '__main__':
    mydata = MyData(r"D:\data\arthrosis\DIP", "train")
    print(mydata[0])
    # train_loader = DataLoader(mydata, batch_size=10, shuffle=True)
    # for x, y in train_loader:
    #     print(x.shape)
    #     print(y.shape)
