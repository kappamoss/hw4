import os
from PIL import Image
from torch.utils.data import Dataset
from utils import ImageTransforms


class SRDataset(Dataset):


    def __init__(self, split, crop_size, scaling_factor, lr_img_type, hr_img_type):

        
        self.split = split.lower()
        self.crop_size = int(crop_size)
        self.scaling_factor = int(scaling_factor)
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type
        self.images = []

        assert self.split in {'train', 'val', 'test'}
       
        assert lr_img_type in {'[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm'}
        assert hr_img_type in {'[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm'}

        if self.split == 'train':
            assert self.crop_size % self.scaling_factor == 0, "Crop dimensions are not perfectly divisible by scaling factor! This will lead to a mismatch in the dimensions of the original HR patches and their super-resolved (SR) versions!"

        if self.split == 'test':
            for dir_path, dir_names, file_names in os.walk('./dataset/testing_lr_images'):
                for f in file_names:
                    self.images.append(os.path.join(dir_path, f))
        else:
            for dir_path, dir_names, file_names in os.walk(os.path.join('./dataset', self.split)):
                for f in file_names:
                    self.images.append(os.path.join(dir_path, f))

        self.transform = ImageTransforms(split=self.split,
                                         crop_size=self.crop_size,
                                         scaling_factor=self.scaling_factor,
                                         lr_img_type=self.lr_img_type,
                                         hr_img_type=self.hr_img_type)

    def __getitem__(self, i):
       
        img = Image.open(self.images[i], mode='r')
        img = img.convert('RGB')
     
        lr_img, hr_img = self.transform(img)
        if self.split == 'test':
            return lr_img, hr_img, self.images[i]
        else:
            return lr_img, hr_img

    def __len__(self):
        return len(self.images)
