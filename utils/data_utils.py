# --1.去雾

# --2.数据少，作数据增样（手会摆动，因此做旋转操作-----+-45度之内）

import os
import random
from PIL import Image
import cv2
import tqdm


# 数据预处理(去雾)
def opt_img(img_path):
    img = cv2.imread(img_path, 0)
    clahe = cv2.createCLAHE(tileGridSize=(3, 3))
    dst = clahe.apply(img)
    cv2.imwrite(img_path, dst)


# 数据增样
def img_rotate(img_path, flag=5):
    img = Image.open(img_path)
    for i in range(flag):
        rota = random.randint(-15, 15)
        dst = img.rotate(rota)
        file_path_name, _ = img_path.split(".")
        dst.save(file_path_name+f"{i}.png")


def img_pre(pic_path_folder):
    print("开始图片预处理")
    for pic_folder in tqdm.tqdm(os.listdir(pic_path_folder), desc="folder", total=len(os.listdir(pic_path_folder))):
        # data_path = os.path.join(pic_path_folder, pic_folder)
        data_path = pic_path_folder + "/" + pic_folder
        num_class = len(os.listdir(data_path))
        for folder in os.listdir(data_path):
            if os.path.isfile(os.path.join(data_path, folder)):
                continue
            img_lists = os.listdir(os.path.join(data_path, folder))
            for index, img in enumerate(img_lists):
                # 去雾
                opt_img(os.path.join(data_path, folder, img))
                # 增样
                img_rotate(os.path.join(data_path, folder, img))
    print("图像预处理完成")


def save_file(list, path, name):
    myfile = os.path.join(path, name)
    if os.path.exists(myfile):
        os.remove(myfile)
    with open(myfile, "w") as f:
        f.writelines(list)


# 划分数据集 train.txt 和 val.txt
def part_data(path):
    for pic_folder in os.listdir(path):
        data_path = path + "/" + pic_folder
        num_class = len(os.listdir(data_path))
        train_list = []
        val_list = []
        train_ration = 0.9
        for folder in os.listdir(data_path):
            if os.path.isfile(os.path.join(data_path, folder)):
                continue
            train_nums = len(os.listdir(os.path.join(data_path, folder))) * train_ration
            img_lists = os.listdir(os.path.join(data_path, folder))
            random.shuffle(img_lists)
            for index, img in enumerate(img_lists):
                if index < train_nums:
                    train_list.append(os.path.join(data_path, folder, img)+" "+str(int(folder)-1)+'\n')
                else:
                    val_list.append(os.path.join(data_path, folder, img)+" "+str(int(folder) - 1)+'\n')
        random.shuffle(train_list)
        random.shuffle(val_list)
        save_file(train_list, data_path, 'train.txt')
        save_file(val_list, data_path, 'val.txt')


if __name__ == '__main__':
    pic_path_folder = "D:/data/arthrosis"
    part_data(pic_path_folder)
