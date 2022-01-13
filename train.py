import os
import time
import torch.backends.cudnn as cudnn
from skimage.metrics import peak_signal_noise_ratio
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from datasets import SRDataset
from model import SRResNet
from utils import *

crop_size = 291 
scaling_factor = 3  

large_kernel_size = 9 
small_kernel_size = 3  
n_channels = 64  
n_blocks = 16  
checkpoint_dir = 'models'
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)

epoch = 0
checkpoint = checkpoint_dir  
batch_size = 32  
start_epoch = 0 
iterations = 1e5  
workers = 4 
lr = 1e-3  
grad_clip = True  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()

cudnn.benchmark = True


def main():
    """
    Training.
    """
    global start_epoch, epoch, checkpoint

    
    print('Find checkpoint')
    checkpoint = torch.load('./models/best_checkpoint_srresnet.pth.tar')
    start_epoch = checkpoint['epoch'] + 1
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']

    model = model.to(device)
    criterion = nn.MSELoss().to(device)

    train_dataset = SRDataset(split='train',
                              crop_size=crop_size,
                              scaling_factor=scaling_factor,
                              lr_img_type='imagenet-norm',
                              hr_img_type='[-1, 1]')

    val_dataset = SRDataset(split='val',
                            crop_size=0,
                            scaling_factor=scaling_factor,
                            lr_img_type='imagenet-norm',
                            hr_img_type='[-1, 1]')
    print('train_loader loading')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                               pin_memory=True)  
    print('val_loader loading')

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=workers,
                                             pin_memory=True)
    
    epochs = int(iterations // len(train_loader) + 1)

    max_psnr = 0
    for epoch in range(start_epoch, epochs):
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)
        psnr = val(val_loader=val_loader,
                   model=model,
                   epoch=epoch)
        if psnr.avg > max_psnr:
            max_psnr = psnr.avg
            torch.save({'epoch': epoch, 
                        'model': model, 
                        'optimizer': optimizer},os.path.join(checkpoint_dir, 'best_checkpoint_srresnet.pth.tar'))

        torch.save({'epoch': epoch, 
                    'model': model, 
                    'optimizer': optimizer},os.path.join(checkpoint_dir, 'checkpoint_srresnet.pth.tar'))


def train(train_loader, model, criterion, optimizer, epoch):
   
    epochs = str(int(iterations // len(train_loader) + 1))

    model.train() 

    batch_time = AverageMeter()  
    data_time = AverageMeter()
    losses = AverageMeter()  
    PSNRs = AverageMeter()  

    start = time.time()

    for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
        data_time.update(time.time() - start)

        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)

        sr_imgs = model(lr_imgs)

        sr_imgs_y = convert_image(sr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)
        hr_imgs_y = convert_image(hr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)
        psnr = peak_signal_noise_ratio(hr_imgs_y.cpu().detach().numpy(), sr_imgs_y.cpu().detach().numpy(), data_range=255.)
        PSNRs.update(psnr, lr_imgs.size(0))

        
        loss = criterion(sr_imgs, hr_imgs)  

        optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        optimizer.step()

        losses.update(loss.item(), lr_imgs.size(0))

        batch_time.update(time.time() - start)

        start = time.time()

    writer.add_scalar('Loss/train', losses.avg, epoch)
    writer.add_scalar('PSNR/train', PSNRs.avg, epoch)
    print(
        f'Epoch: {epoch} / ' + epochs + ' -- '
        f'Batch Time: {batch_time.avg:.3f} -- '
        f'Loss: {losses.avg:.4f} -- '
        f'PSNR: {PSNRs.avg:.4f}'
    )
    del lr_imgs, hr_imgs, sr_imgs


def val(val_loader, model, epoch):
    model.eval()
    PSNRs = AverageMeter()
    with torch.no_grad():
        for i, (lr_imgs, hr_imgs) in enumerate(val_loader):
            
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            
            sr_imgs = model(lr_imgs)
            
           
            sr_imgs_y = convert_image(sr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)
            hr_imgs_y = convert_image(hr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)
            psnr = peak_signal_noise_ratio(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(), data_range=255.)
            
            PSNRs.update(psnr, lr_imgs.size(0))
    print(f'Epoch: {epoch}, PSNR: {PSNRs.avg}')
    writer.add_scalar('PSNR/val', PSNRs.avg, epoch)
    del lr_imgs, hr_imgs, sr_imgs

    return PSNRs


if __name__ == '__main__':
    main()
