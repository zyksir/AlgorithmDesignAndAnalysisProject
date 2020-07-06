# 定义生成器、判别器和特征提取器
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math


# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()  # in_features: 64
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
        )

    def forward(self, x):
        return x + self.conv_block(x)


# 定义生成器
class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        super(GeneratorResNet, self).__init__()

        # 第一个卷积层，见注释１
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4), nn.PReLU())

        # 初始化残差块
        res_blocks = []

        # 生成16个残差块，见注释２
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # 第二个卷积层，见注释３
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64, 0.8))

        # 上采样层，见注释４
        upsampling = []
        for out_features in range(2):  # why range(2) ?
            upsampling += [
                # nn.Upsample(scale_factor=2),
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=2),  # scale_factor
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # 第三个卷积层，见注释４
        self.conv3 = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4), nn.Tanh())

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.res_blocks(out1)
        out3 = self.conv2(out2)
        out4 = torch.add(out1, out3)
        out5 = self.upsampling(out4)
        out = self.conv3(out5)
        return out


# 定义判别器
class Discriminator(nn.Module):
    def discriminator_block(self, in_filters, out_filters, first_block=False):
        layers = []  # 每次layers都会重置为空
        layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
        if not first_block:  # 当first_block为False时，layers的第二层添加一个BN层
            layers.append(nn.BatchNorm2d(out_filters))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(out_filters))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def __init__(self, input_shape):  # input_shape: (3, 128, 128)
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)  # patch_h: 8 patch_w: 8

        layers = []  # 见注释６
        in_filters = in_channels  # in_filters = 3
        for i, out_filters in enumerate([64, 128, 256, 512]):
            first_block = (i == 0)
            layers.extend(self.discriminator_block(in_filters, out_filters, first_block))
            in_filters = out_filters

        # 见注释７
        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)


# 定义特征提取器
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        # 取了VGG19的前19层，见注释８
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:19])

    def forward(self, img):
        return self.feature_extractor(img)
