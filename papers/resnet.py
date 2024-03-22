import numpy as np
import pandas as pd
import matplotlib as plt
import mlx
import mlx.nn as nn
import mlx.core as mx


"""
Notes:
    * BatchNorm after each convolution and before activation
    * No dropout
    * Image is resized with its shorter size sampled in [256, 480] for scale augmentation 
    * Conv3-64 = 64 3x3 filters, resulting in 64 output features
    * SGD with batch size of 256
Options:
    1. Define each layer, then code a class that takes everything and chains them together
    2. Define the blocks and forward in the class
    3. NanoGPT it - component (layer), block, resnet
Where do I pass in the hyperparameters? ResNetConfig()?
Define:
    * No. layers: 34
    * Input size: 224x224 crop sampled from an image, per-pixel mean subtracted, color augementation
    * Output size: 
    * No. I/O features of each block: depends
    * Kernel size (convolution filter): usually depends, but all (3x3) for ResNet
    * Convolutional layer hyperparameters - filter (3x3), stride, ?
    * Weight decay: 1e-4
    * Momentum: 9e-1
    * Learning rate: starts at 0.1, divided by 10 as error plateaus 
"""

class ResBlock(nn.Module):
    """
    Take the input, normal forward, recast input onto original
    y = F(x) + x where F(x) is the mapping function (wx + b)
    """
    ### 

    def __init__(self, no_blocks, no_layers, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        # No weights and biases because lazy eval i can call resblock.parameters() later?
        self.no_blocks = no_blocks
        self.no_layers = no_layers
        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.kernel_size = 3 # It's a square
        self.stride = stride
    
    def forward(self, x):
         
        return out


class ResNet (nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d() # Fill in the hyperparams later
            
    def relu(self, x):
        return nn.relu(x)
    
