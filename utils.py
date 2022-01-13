from PIL import Image
import os
import json
import random
import torchvision.transforms.functional as FT
import numpy as np
import torch
import math
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966]).to(device)
imagenet_mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
imagenet_std = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
imagenet_mean_cuda = torch.FloatTensor([0.485, 0.456, 0.406]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
imagenet_std_cuda = torch.FloatTensor([0.229, 0.224, 0.225]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)



def convert_image(img, source, target):
   
    assert source in {'pil', '[0, 1]', '[-1, 1]'}, "Cannot convert from source format %s!" % source
    assert target in {'pil', '[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm',
                      'y-channel'}, "Cannot convert to target format %s!" % target

    # Convert from source to [0, 1]
    if source == 'pil':
        img = FT.to_tensor(img)

    elif source == '[0, 1]':
        pass  # already in [0, 1]

    elif source == '[-1, 1]':
        img = (img + 1.) / 2.

    # Convert from [0, 1] to target
    if target == 'pil':
        img = FT.to_pil_image(img)

    elif target == '[0, 255]':
        img = 255. * img

    elif target == '[0, 1]':
        pass  # already in [0, 1]

    elif target == '[-1, 1]':
        img = 2. * img - 1.

    elif target == 'imagenet-norm':
        if img.ndimension() == 3:
            img = (img - imagenet_mean) / imagenet_std
        elif img.ndimension() == 4:
            img = (img - imagenet_mean_cuda) / imagenet_std_cuda

    elif target == 'y-channel':
     
        img = torch.matmul(255. * img.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :], rgb_weights) / 255. + 16.

    return img

class ImageTransforms(object):


    def __init__(self, split, crop_size, scaling_factor, lr_img_type, hr_img_type):
       
        self.split = split.lower()
        self.crop_size = crop_size
        self.scaling_factor = scaling_factor
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type

        assert self.split in {'train', 'val', 'test'}

    def __call__(self, img):
       
        if img.width <= self.crop_size or img.height <= self.crop_size:
            img = padding(img, self.crop_size)
        
        # Crop
        if self.split == 'train':
            if img.width - self.crop_size <= 1:
                left = 0
            else:
                left = random.randint(1, img.width - self.crop_size)
            if img.height - self.crop_size <= 1:
                top = 0
            else:
                top = random.randint(1, img.height - self.crop_size)
            right = left + self.crop_size
            bottom = top + self.crop_size

            img = img.crop((left, top, right, bottom))
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomGrayscale(p=0.1),
                
            ])
            hr_img = transform(img)

            lr_img, hr_img = self.downsample(hr_img)
        elif self.split == 'val':
            x_remainder = img.width % self.scaling_factor
            y_remainder = img.height % self.scaling_factor
            left = x_remainder // 2
            top = y_remainder // 2
            right = left + (img.width - x_remainder)
            bottom = top + (img.height - y_remainder)
            hr_img = img.crop((left, top, right, bottom))

            lr_img, hr_img = self.downsample(hr_img)
        else:
            lr_img = convert_image(img, source='pil', target=self.lr_img_type)
            hr_img = []

        return lr_img, hr_img

    def downsample(self, hr_img):
      
        lr_img = hr_img.resize((int(hr_img.width / self.scaling_factor), int(hr_img.height / self.scaling_factor)),
                               Image.BICUBIC)
        # Convert the LR and HR image to the required type
        lr_img = convert_image(lr_img, source='pil', target=self.lr_img_type)
        hr_img = convert_image(hr_img, source='pil', target=self.hr_img_type)
        return lr_img, hr_img

class AverageMeter(object):
 
    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def clip_gradient(optimizer, grad_clip):
  
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def padding(img, crop_size):
    w, h = img.size
    max_h = np.max([h, crop_size])
    max_w = np.max([w, crop_size])
    hp = int((max_h - h) / 2)
    wp = int((max_w - w) / 2)
    pad = [wp, hp, wp, hp]
    return FT.pad(img, pad, 0, 'constant')
