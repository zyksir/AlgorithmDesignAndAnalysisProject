from model import GeneratorResNet,Discriminator
import torchvision.transforms as transforms
import argparse
import torch
from PIL import Image
import os


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="DIV2K_valid_LR_mild", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=128, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=128, help="high res. image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between model checkpoints")
opt = parser.parse_known_args()[0]
# opt = parser.parse_args()
print(opt)

generator = GeneratorResNet()

generator.load_state_dict(torch.load("saved_models/generator_19.pth"))

imagelist=sorted(os.listdir("./data/"+opt.dataset_name))

for i in range(len(imagelist)):
    img = Image.open("./data/" + opt.dataset_name+'/'+imagelist[i]).convert('RGB')
    img=transforms.ToTensor()(img).unsqueeze(dim=0)
    gen_hr=generator(img).squeeze(dim=0)
    unloader = transforms.ToPILImage()
    gen_hr = unloader(gen_hr).convert('RGB')
    gen_hr.save("./data/DIV2K_valid_myHR_mygan/"+imagelist[i])