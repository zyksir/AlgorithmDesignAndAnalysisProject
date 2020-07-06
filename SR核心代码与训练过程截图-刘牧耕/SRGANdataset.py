# 接下来就要定义数据集，并对其进行一系列的操作
import random
import os
import numpy as np
import glob

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import cut_train


# 设定预训练PyTorch模型的归一化参数
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# 最后，定义数据集操作方法：
class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        print("dataloader")
        hr_height, hr_width = hr_shape  # hr_shape=(128, 128)
        # 通过源图像分别生成低、高分辨率图像，4倍
        self.lr_transform = transforms.Compose(  # 见注释8
            [
                transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        # 将文件夹中的图片进行按文件名升序排列，从000001.jpg到202599.jpg
        self.files = sorted(glob.glob(root + "/*.*"))


    def __getitem__(self, index):  # 定义时未调用，每次读取图像时调用，见注释9
        img = Image.open(self.files[index % len(self.files)])
        LR_img = self.lr_transform(img)
        HR_img = self.hr_transform(img)

        return {"lr": LR_img, "hr": HR_img}

    # 定义dataloader和每次读取图像时均调用
    def __len__(self):
        return len(self.files)
