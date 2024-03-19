import glob
import os.path
import torch
from torch import nn
from torchvision import models
from torch.utils.data import DataLoader
from utils.data_set import MyData

from torch.utils.tensorboard import SummaryWriter

# 13个关节对应的分类模型
arthrosis = {'MCPFirst': ['MCPFirst', 11],  # 第一手指掌骨
             'DIPFirst': ['DIPFirst', 11],  # 第一手指远节指骨
             'PIPFirst': ['PIPFirst', 12],  # 第一手指近节指骨
             'MIP': ['MIP', 12],  # 中节指骨（除了拇指剩下四只手指）（第一手指【拇指】是没有中节指骨的））
             'Radius': ['Radius', 14],  # 桡骨
             'Ulna': ['Ulna', 12],  # 尺骨
             'PIP': ['PIP', 12],  # 近节指骨（除了拇指剩下四只手指）
             'DIP': ['DIP', 11],  # 远节指骨（除了拇指剩下四只手指）
             'MCP': ['MCP', 10]}  # 掌骨（除了拇指剩下四只手指）

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summaryWriter = SummaryWriter("logs")


def create_model(cls):
    net = models.resnet18()
    net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    net.fc = nn.Linear(512, cls)
    return net.to(DEVICE)


def train(category):
    train_txt = os.path.join('D:/data/arthrosis', category)
    val_txt = os.path.join('D:/data/arthrosis', category)
    train_dataset = MyData(train_txt, 'train')
    val_dataset = MyData(val_txt, 'val')
    train_loader = DataLoader(train_dataset, batch_size=60, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=60, shuffle=True)
    model = create_model(arthrosis[category][1])

    # 加载预训练权重
    if os.path.exists("params/{}_best.pth".format(category)):
        model.load_state_dict(torch.load("params/{}_best.pth".format(category), map_location=DEVICE))
    # print(model)
    loss_func = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters())
    print('{}_开始训练 ...'.format(category))
    model.train()
    best_acc = 0
    for epoch in range(80):
        train_losses = 0
        for i, (img, label) in enumerate(train_loader):
            img, label = img.to(DEVICE), label.to(DEVICE)
            out = model(img)
            loss = loss_func(out, label)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_losses += loss.item()

        train_avg_loss = train_losses/len(train_loader)
        print("epoch==>", epoch, "train_avg_loss==>", train_avg_loss)

        # 训练期间验证
        model.eval()
        all_acc = 0
        val_losses = 0
        for i, (img, label) in enumerate(val_loader):
            img, label = img.to(DEVICE), label.to(DEVICE)
            out = model(img)
            loss = loss_func(out, label)
            acc = torch.mean(torch.eq(out.argmax(dim=1), label).float())
            val_losses += loss.item()
            all_acc += acc.item()
        val_avg_losses = val_losses/len(val_loader)
        val_avg_acc = all_acc/len(val_loader)
        print("val_avg_losses==>", val_avg_losses, "val_acc==>", val_avg_acc)
        if val_avg_acc > best_acc:
            best_acc = val_avg_acc
            torch.save(model.state_dict(), f"params/{category}_best.pth")
            print("成功保存模型！")
        # 收集损失
        summaryWriter.add_scalars(category+"/"+"loss", {"train_avg_loss": train_avg_loss,
                                                        category+"/"+"val_avg_losses": val_avg_losses})
        summaryWriter.add_scalar(category+'/'+"val_avg_acc", val_avg_acc, epoch)


def run():
    for item in arthrosis:
        train(item)


if __name__ == '__main__':
    run()
