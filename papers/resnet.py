import time
import numpy as np
import pandas as pd
import matplotlib as plt
import mlx
import mlx.nn as nn
import mlx.core as mx
from mlx.optimizers import SGD

from datasets import load_dataset

# If the dataset is gated/private, make sure you have run huggingface-cli login
dataset = load_dataset("imagenet-1k")


"""
Notes:
    * BatchNorm after each convolution and before activation
    * No dropout
    * Image is resized with its shorter size sampled in [256, 480] for scale augmentation 
    * Conv3-64 = 64 3x3 filters, resulting in 64 output features
    * SGD with batch size of 256
    * Works with 3.12 but not 3.9
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
    # i can write a block for each block, but the better option is to create layers based on the input parameters 
    # the no_blocks is variable but the interior structure of each layer is the same, except the no. channels
    # 
    expansion = 1 # Determines the ratio between IO channelsin a resblock, controls the dimesionality of the feature maps

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None): # Block as described in the paper
        super().__init__()
        """
        in_channels: int; add however many IO channels the block specifies (64, 64, 128, 256, 512) 
        out_channels: int; 
        """
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, bias=True)
        self.bn1 = nn.BatchNorm(num_features=out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, bias=True)
        self.bn2 = nn.BatchNorm(num_features=out_channels)

        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        cache = x

        # First layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
       
        # Second layer
        x = self.conv2(x)
        x = self.bn2(x)
        
        # Downsample?
        if self.downsample is not None:
            cache = self.downsample(x)

        x = mx.add(x, cache)
        out = self.relu(x)

        return out


class ResNet (nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        """
        block: ResBlock
        layers = list of how many blocks are in each layer [3, 4, 6, 3]
        num_classes: no. classes, for classification
        """
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm(num_features=64)
        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d((1, 1))
        self.fc = nn.Linear(input_dims=512*block.expansion, output_dims=num_classes)

    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                    nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),  
                    nn.BatchNorm(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        out  = self.fc(x)

        return out

"""
Work for tmr:
    * figure out how to train the model; what's batch iterate and what do i do
    * use gpu
    * actually train
    * save weights
    * load weights
"""

# Data preparation
train_images = dataset['train']['image']
train_labels = dataset['train']['label']
val_images = dataset['validation']['image']
val_labels = dataset['validation']['label']
test_images = dataset['test']['image']
test_labels = dataset['test']['label']


# Hyperparameters
lr = 1e-1 # Learning rate
momentum = 9e-1
dr = 1e-4 # Decay rate 
no_epochs = 5
batch_size = 256
layers = [3, 4, 6, 3]


# Initialize network
resnet34 = ResNet(ResBlock, layers) 
mx.eval(resnet34.parameters())


# Optimizers, functions
def loss_fn(model, X, y):
    return mx.mean(nn.losses.cross_entropy(model(X), y))

def eval_fn(model, X, y):
    return mx.mean(mx.argmax(model(X), axis=1) == y)

loss_and_grad_fn = nn.value_and_grad(resnet34, loss_fn)
optimizer = SGD(lr, momentum, dr) 

def predict(model, image):
    image = mx.array(image)
    image = image.reshape(1, -1)
    predictions = model(image)
    predicted_class = mx.argmax(predictions, axis=1)
    return predicted_class.item()

# Training loop
for i in range(no_epochs):
    tic = time.perf_counter()
    for X, y in batch_iterate(batch_size, train_images, train_labels):
        loss, grads = loss_and_grad_fn(model, X, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
    accuracy = eval_fn(model, test_images, test_labels)
	toc = time.perf_counter()
	print(
		f"Epoch {e}: Test accuracy {accuracy.item():.3f},"
		f" Time {toc - tic:.3f} (s)"
	)

resnet34.save_weights('resnet34') 

