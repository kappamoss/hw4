from utils import *
from skimage.metrics import peak_signal_noise_ratio
from datasets import SRDataset
import os
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data_path = 'dataset/val'
images = []

srresnet_checkpoint = 'models/best_checkpoint_srresnet.pth.tar'


srresnet = torch.load(srresnet_checkpoint)['model'].to(device)
srresnet.eval()
model = srresnet

val_dataset = SRDataset(split='val', crop_size=0, scaling_factor=3, lr_img_type='imagenet-norm',
                        hr_img_type='[-1, 1]')

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4,
                                         pin_memory=True)

PSNRs = AverageMeter()

with torch.no_grad():

    for i, (lr_imgs, hr_imgs) in enumerate(val_loader):
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)

        sr_imgs = model(lr_imgs)

        sr_imgs_y = convert_image(sr_imgs, source='[-1, 1]', target='y-channel').squeeze(0) 
        hr_imgs_y = convert_image(hr_imgs, source='[-1, 1]', target='y-channel').squeeze(0) 
        psnr = peak_signal_noise_ratio(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(),data_range=255.)
      
        PSNRs.update(psnr, lr_imgs.size(0))
        
print('PSNR - {psnrs.avg:.3f}'.format(psnrs=PSNRs))
