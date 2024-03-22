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
    * 
Options:
    1. Define each layer, then code a class that takes everything and chains them together
    2. Define the blocks and forward in the class

Where do I pass in the hyperparameters? ResNetConfig()?
Define:
    * No. layers: 34
    * Input size: 224x224 crop sampled from an image, per-pixel mean subtracted, color augementation
    * Output size: 
    * Dimensions of the each block
    * Convolution filter: (3x3)
    * Convolutional layer hyperparameters - filter (3x3), step, ?
    * Weight decay: 1e-4
    * Momentum: 9e-1
    * Learning rate: starts at 0.1, divided by 10 as error plateaus 
"""

class ResBlock(nn.Module):
    def __init__(self):
       self.weights = weights
       self.filter = frame

    def relu(self, x):
        return nn.relu(x)
    
