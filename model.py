import torch
import torchvision
import math
from torch import nn

class ConvolutionalBlock(nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=None):
   
        super(ConvolutionalBlock, self).__init__()

        if activation is not None:
            activation = activation.lower()
            assert activation in {'prelu', 'leakyrelu', 'tanh'}

      
        layers = list()

        layers.append(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2))

        if batch_norm is True:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        if activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation == 'tanh':
            layers.append(nn.Tanh())

        
        self.conv_block = nn.Sequential(*layers)

    def forward(self, input):
       
        output = self.conv_block(input)  

        return output


class SubPixelConvolutionalBlock(nn.Module):
    

    def __init__(self, kernel_size=3, n_channels=64, scaling_factor=2):
       
        super(SubPixelConvolutionalBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels * (scaling_factor ** 2),
                              kernel_size=kernel_size, padding=kernel_size // 2)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)
        self.prelu = nn.PReLU()

    def forward(self, input):
        
        output = self.conv(input) 
        output = self.pixel_shuffle(output)  
        output = self.prelu(output)

        return output


class ResidualBlock(nn.Module):

    def __init__(self, kernel_size=3, n_channels=64):
       
        super(ResidualBlock, self).__init__()

        self.conv_block1 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation='PReLu')

        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation=None)

    def forward(self, input):
       
        residual = input  
        output = self.conv_block1(input)  
        output = self.conv_block2(output)  
        output = output + residual 

        return output


class SRResNet(nn.Module):
 

    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=3):
   
        super(SRResNet, self).__init__()

        scaling_factor = int(scaling_factor)

        self.conv_block1 = ConvolutionalBlock(in_channels=3, out_channels=n_channels, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='PReLu')

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(kernel_size=small_kernel_size, n_channels=n_channels) for _ in range(n_blocks)])

        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels,
                                              kernel_size=small_kernel_size,
                                              batch_norm=True, activation=None)

        self.subpixel_convolutional_blocks = SubPixelConvolutionalBlock(kernel_size=small_kernel_size,
                                                                        n_channels=n_channels,
                                                                        scaling_factor=scaling_factor)

        self.conv_block3 = ConvolutionalBlock(in_channels=n_channels, out_channels=3, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='Tanh')

    def forward(self, lr_imgs):
  
        output = self.conv_block1(lr_imgs)  
        residual = output 
        output = self.residual_blocks(output) 
        output = self.conv_block2(output) 
        output = output + residual 
        output = self.subpixel_convolutional_blocks(output)
        sr_imgs = self.conv_block3(output) 

        return sr_imgs
