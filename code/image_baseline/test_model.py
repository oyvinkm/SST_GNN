#!/usr/bin/env python3
import torch
import glob
import os
import sys
import numpy as np

from loguru import logger
from PIL import Image
from matplotlib.image import imsave
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.transforms.functional import pil_to_tensor as PTT
from torchvision.transforms.functional import to_pil_image as TPI
from VAE import VAE

logger.remove(0)
logger.add(sys.stderr, level='INFO')

if torch.backends.mps.is_available():
  device = 'mps'
elif torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'
logger.success(f'Device : {device}')




def get_numeric_part(filename):
    return int(''.join(filter(str.isdigit, filename)))

class MyDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x
        
    def __len__(self):
        return len(self.subset)

net = VAE(channel_in = 3, z = 32, device = device).to(device)
model_path = 'model_chkpoints/model_VAE_2024_01_16-08.17.pt'

if model_path is not None:
  state_dict = torch.load(model_path)
  net.load_state_dict(state_dict['state_dict'])
  logger.success(f'Model loaded')

transform = T.Compose([T.Resize((128, 512)),
                      #T.Grayscale(),
                      T.ToTensor()])

PATH = 'data/cylinder_flow/image_traj/root/traj_1'
data = MyDataset([Image.open(image) for image in sorted(glob.glob(f"{PATH}/*.png"), key=get_numeric_part)], transform=transform)
logger.success(f'Length of data : {len(data)}')
testLoader = DataLoader(data, batch_size=1, shuffle=False)
print(len(testLoader))
print(next(iter(testLoader)).shape)

_, out = net(next(iter(testLoader)).to(device), Train = False)
print(out.shape)
save_folder = 'test_img'
if not os.path.isdir(save_folder):
   os.mkdir(save_folder)
for i, img in enumerate(testLoader):
   _, out = net(img.to(device))
   img = out[0].permute(1,2,0).cpu().detach().numpy()
   imsave(os.path.join(save_folder, f'{i}.png'), img)

logger.success(f'Done Predicting and saving images')
def make_gif(data_folder, skip = 2):
    images = sorted(glob.glob(f"{data_folder}/*.png"), key=get_numeric_part)
    frames = [Image.open(image) for i, image in enumerate(images) if i % skip == 0]
    frame_one = frames[0]
    frame_one.save("pred_gif_01_16_08.17.gif", format="GIF", append_images=frames,
               save_all=True, duration=50, loop=0)
logger.success(f'Creating Gif')
make_gif(save_folder, skip = 10)
logger.success(f'Gif created')