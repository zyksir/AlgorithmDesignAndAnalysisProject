import os
import itertools
import sys
import glob
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable

from model import ResidualBlock,GeneratorResNet,Discriminator,FeatureExtractor
from dataset import ImageDataset

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4" # 设置系统可见的GPU从１号开始
os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="DIV2K_train_HR", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=20, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=128, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=128, help="high res. image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
opt = parser.parse_known_args()[0]
# opt = parser.parse_args()
print(opt)

hr_shape = (opt.hr_height, opt.hr_width)

# 初始化生成器、判别器和特征提取器：
generator = GeneratorResNet()
discriminator = Discriminator((opt.channels, *hr_shape))
feature_extractor = FeatureExtractor()

# 将特征提取器设为评估模式：
feature_extractor.eval()

# # 从第2次循环开始，载入训练得到的生成器和判别器模型：
# if opt.epoch != 0:
#     generator.load_state_dict(torch.load("saved_models/generator_%d.pth"))
#     discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth"))

# 设置优化器
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# 设置损失函数，MSELoss和L1Loss
criterion_GAN = torch.nn.BCELoss()
criterion_content = torch.nn.MSELoss()

# self.files[index % len(self.files)]

dataloader = DataLoader(
    ImageDataset("./data/%s" % opt.dataset_name, hr_shape),
    batch_size=opt.batch_size,  # batch_size = 4
    shuffle=True,
    num_workers=opt.n_cpu  # num_workers = 8
)

# 开始训练
# 定义Tensor类型
# Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
Tensor=torch.Tensor

# Pre-train generator using raw MSE loss
print("Generator pre-training")
total_loss=0
for epoch in range(5):
    total_loss = 0
    for i, imgs in enumerate(dataloader):
        # Generate data
        print("epoch",epoch,":",i)
        imgs_lr = Variable(imgs["lr"].type(Tensor))  # torch.Size([4,3,32,32])
        imgs_hr = Variable(imgs["hr"].type(Tensor))  # torch.Size([4,3,128,128])

        gen_hr = generator(imgs_lr)
        ######### Train generator #########
        print(gen_hr.shape,imgs_hr.shape)
        generator_content_loss = criterion_content(gen_hr,imgs_hr)
        total_loss+=generator_content_loss.item()
        print("begin loss")
        optimizer_G.zero_grad()
        generator_content_loss.backward()
        optimizer_G.step()

        ######### Status and display #########
        sys.stdout.write(
            "[Epoch %d/%d] [Batch %d/%d] [loss: %f] "
            % (epoch, 5, i, len(dataloader), generator_content_loss.item()) + '\n'
        )  # 相当于print()
    sys.stdout.write('\r[%d/%d] Generator_MSE_Loss: %.4f\n' % (epoch, 5, total_loss))


print("Begin training")
for epoch in range(opt.epoch,opt.n_epochs):
    total_loss = 0
    for i, imgs in enumerate(dataloader):
        # 定义低、高分辨率图像对，imgs为字典
        imgs_lr = Variable(imgs["lr"].type(Tensor))  # torch.Size([4,3,32,32])
        imgs_hr = Variable(imgs["hr"].type(Tensor))  # torch.Size([4,3,128,128])

        # 生成地面真值,真为1，假为0
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        # torch.Size([4,1,8,8])
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        # torch.Size([4,1,8,8])

        # ------------------
        #  训练生成器
        # ------------------

        print("generator")
        # 利用生成器从低分辨率图像生成高分辨率图像，见注释10
        gen_hr = generator(imgs_lr)  # gen_hr: (4,3,128,128)

        # 对抗损失，见注释11
        # 第一次循环: tensor(0.9380, device='cuda:0', grad_fn=<MseLossBackward>)
        loss_GAN = criterion_GAN(nn.Sigmoid()(discriminator(gen_hr)), valid)

        # 内容损失，见注释12
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)
        loss_content = criterion_content(gen_features, real_features.detach())*0.001+criterion_content(gen_hr,imgs_hr)

        # 生成器的总损失
        loss_G = loss_content + 1e-3 * loss_GAN
        total_loss +=loss_G.item()

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  训练判别器
        # ---------------------

        print("discriminator")
        # 真假图像的损失
        loss_real = criterion_GAN(nn.Sigmoid()(discriminator(imgs_hr)), valid)
        loss_fake = criterion_GAN(nn.Sigmoid()(discriminator(gen_hr.detach())), fake)

        # 判别器的总损失
        loss_D = loss_real + loss_fake
        total_loss+=loss_D.item()

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  输出记录
        # --------------

        sys.stdout.write(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item()) + '\n'
        )  # 相当于print()

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            # 保存上采样和SRGAN输出的图像
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
            img_grid = torch.cat((imgs_lr, gen_hr), -1)
            save_image(img_grid, "images/%d.png" % batches_done, normalize=False)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # 每10个epoch保存一次模型
        sys.stdout.write(
            "[Epoch %d/%d] [loss: %f]"
            % (epoch, opt.n_epochs, total_loss) + '\n'
        )  # 相当于print()
        torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
        torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch)
