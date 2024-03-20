import torch


# box yolo侦测出来的框
# cls 要筛选的类别数
# flag 要找到的关节的索引列表
def filter(box, cls, flag):
    box = box[torch.where(box[:, 5] == cls)]
    index = box[:, 0].argsort()
    return box[index][flag]


def bone_filter(box):
    if box.shape[0] != 21:
        print("ERROR! Wrong numbers of bone...")
        return
    else:
        distalphalanx = filter(box, 6, [0, 2, 4])
        middlephalanx = filter(box, 5, [0, 2])
        proximalphalanx = filter(box, 4, [0, 2, 4])
        mcp = filter(box, 3, [0, 2])
        mcpfirst = filter(box, 2, [0])
        radius = filter(box, 1, [0])
        ulna = filter(box, 0, [0])

        return torch.cat([distalphalanx, middlephalanx, proximalphalanx, mcp, mcpfirst, radius, ulna])


