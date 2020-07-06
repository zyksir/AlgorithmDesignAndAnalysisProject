import evaluate
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import torch
HR_address="/Users/liumugeng/Desktop/my SRCNN/myDIV2K/DIV2K_valid_HR/0801.png"
train_HR_address="/Users/liumugeng/Desktop/my SRCNN/myDIV2K/DIV2K_valid_HR_result/0801.png"
bicubi_address="/Users/liumugeng/Desktop/my SRCNN/myDIV2K/DIV2K_valid_LR_mildx4/0801x4m.png"

trans=transforms.ToTensor()

HR = Image.open(HR_address).convert('RGB')

train_HR = Image.open(train_HR_address).convert('RGB')
# train_HR=torch.tensor(train_HR)
bicubi = Image.open(bicubi_address).convert('RGB')
# bicubi=torch.tensor(bicubi)

print("HR:train ",evaluate.psnr(np.asarray(HR),np.asarray(train_HR)))
print("HR:HR",evaluate.psnr(np.asarray(HR),np.asarray(HR)))
print("HR:bicubic",evaluate.psnr(np.asarray(HR),np.asarray(bicubi)))

HR=cv2.imread(HR_address)
train_HR=cv2.imread(train_HR_address)
bicubi=cv2.imread(bicubi_address)

print("HR:train ",evaluate.psnr(HR,train_HR))
print("HR:HR",evaluate.psnr(HR,HR))
print("HR:bicubic",evaluate.psnr(HR,bicubi))