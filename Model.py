#!/usr/bin/env python
# coding: utf-8




import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as f
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict
import glob 
from PIL import Image
import numpy as np 
from torch.distributions import uniform



# creating the cutout algorithm
## get the uniformly random mask size and the center point 
def size_center(s, height, width):
    '''
    Arguments:
    s: the random uniform boundaries for cutout sizes and the center points
    height: the height of the input images (boundary of the size respect to the height)
    width: the width of the input images (boundary of the size respect to the width)
    
    return:
    size: uniformly random mask size
    x: uniformly random center point (location) respect to height 
    y: uniformly random center point (location) respect to width
    '''
    ### uniformly random mask size
    size = uniform.Uniform(0,s)
    size = size.sample()

    ### uniformly random center point
    x = uniform.Uniform(0,height)
    x = x.sample()
    y = uniform.Uniform(0,width)
    y = y.sample()

    return size, x, y   

## get the mask boundaries in the single image
def mask_boundary(size, x, y, num_patches, height, width, s):
    '''
    Arguments: 
    size: the mask size
    x: the center point (location) respect to height 
    y: the center point (location) respect to width
    num_patches: the number of patches in each image
    height: the height of the input images (boundary of the size respect to the height)
    width: the width of the input images (boundary of the size respect to the width)
    s: the random uniform boundaries for cutout sizes and the center points
    
    return:
    x_left_margin_list: all x location left margins in each image
    x_right_margin_list: all x location right margins in each image
    y_top_margin_list: all y location top margins in each image
    y_bottom_margin_list: all y location bottom margins in each image
    '''
    ### initial the lists for all margins 
    x_left_margin_list = []
    x_right_margin_list = []
    y_top_margin_list = []
    y_bottom_margin_list = []
    
    for i in range(num_patches):
        ### clamp (set the boundaries for our squared mask)
        x_left_margin = torch.round(torch.clamp(x - size//2,0,height))
        x_right_margin = torch.round(torch.clamp(x + size//2,0,height))
        y_top_margin = torch.round(torch.clamp(y - size//2,0,width))
        y_bottom_margin = torch.round(torch.clamp(y + size//2,0,width))
        ### change all the above data type to the integer
        x_left_margin = x_left_margin.to(torch.int)
        x_right_margin = x_right_margin.to(torch.int)
        y_top_margin = y_top_margin.to(torch.int)
        y_bottom_margin = y_bottom_margin.to(torch.int)
        ### append each time margins into margin lists
        x_left_margin_list.append(x_left_margin)
        x_right_margin_list.append(x_right_margin)
        y_top_margin_list.append(y_top_margin)
        y_bottom_margin_list.append(y_bottom_margin)

        ### update random new size and the center point
        size = size_center(s,height,width)[0]
        x = size_center(s,height,width)[1]
        y = size_center(s,height,width)[2]

    return x_left_margin_list, x_right_margin_list, y_top_margin_list, y_bottom_margin_list 

## creating the final cutout function
def Cutout(images,s, num_patches):
    '''
    Arguments:
    s: the random uniform boundaries for cutout sizes and the center points
    num_patches: the number of patches in each image
    images: all input images (features)
    
    return:
    cutout_image: all images covered with cutout masks
    '''
    ### get the full boundary (the height and width from our images)
    height = images.shape[-2]
    width = images.shape[-1]
    ### create the defult mask
    mask = torch.ones(images.shape[0],images.shape[1],height,width,dtype=torch.int64)

    ### creating the cutout masks (zero padding areas) for all images
    for i in range(images.shape[0]):
        #### setting the size and the center point in each image
        size = size_center(s,height,width)[0]
        x = size_center(s,height,width)[1]
        y = size_center(s,height,width)[2]
        #### geting the margin lists from mask boundaries (all possible margins in each image)
        x_left_margin_list = mask_boundary(size, x, y, num_patches, height, width,s)[0]
        x_right_margin_list = mask_boundary(size, x, y, num_patches, height, width,s)[1]
        y_top_margin_list = mask_boundary(size, x, y, num_patches, height, width,s)[2]
        y_bottom_margin_list = mask_boundary(size, x, y, num_patches, height, width,s)[3]
    
        #### zero padding cutout masks for each image  
        for j in range(images.shape[1]):
            for k in range(num_patches):
                mask[i][j][x_left_margin_list[k]:x_right_margin_list[k], y_top_margin_list[k]:y_bottom_margin_list[k]] = 0
        
    ### getting the new images covered with cutout masks
    cutout_image = images * mask
  
    return cutout_image



# create SE-ResNet (attention layer)
# resnet with soft-attention map
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
    
class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


    
def se_resnet50(num_classes=1, pretrained=True):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    from torchvision.models import ResNet

    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url("https://download.pytorch.org/models/resnet50-0676ba61.pth")#"https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl", model_dir="./model_data")
        
        model.load_state_dict(state_dict) #"https://download.pytorch.org/models/resnet50-0676ba61.pth"))
        model.fc = nn.Linear(model.fc.in_features,num_classes)
        
    return model
# create resnet

# MobileNet-SSD
class MobileNetSSD(nn.Module):
    # (64,1): out_channels is 64, stride is 1 (block configure in MobileNet)
    block_config = [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2),
                    (512, 1), (512, 1), (512, 1), (512, 1), (512, 1)]

    # block configures in SSD
    ssd_config = [(1024, 256, 1), (256, 512, 2), (512, 128, 1), (128, 256, 2), (256, 128, 1), (128, 256, 2),
                  (256, 128, 1), (128, 256, 1), (256, 128, 1), (128, 256, 1)]

    def __init__(self, num_classes):
        """
        Create MobileNet-SSD to bounding box task. This network uses MobileNet as the backbone to
        speed up the original SSD network. MobileNet mainly works with Depth-wise convolution and
        SSD is traditional convolution.
        :param num_classes: the number of classes/ out channels we would like to generate
        """
        super(MobileNetSSD, self).__init__()

        # pre-layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32,
                               kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # make dw layers
        self.layers = self.make_dw_layer(in_channels=32)

        # transition layers from Mobilenet to SSD
        self.trans_conv1 = nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.trans_bn1 = nn.BatchNorm2d(1024)
        self.trans_conv2 = nn.Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.trans_bn2 = nn.BatchNorm2d(1024)

        # make SSD layers
        self.ssd_layers = self.make_ssd()

        # Fully Linear layer in mobilenet
        self.linear = nn.Linear(256, num_classes)

    def make_dw_layer(self, in_channels):
        layers = []

        for i in self.block_config:
            out_channels = i[0]
            stride = i[1]
            layers.append(MobileNetSSD.Depthwise(in_channels, out_channels, stride))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def make_ssd(self):
        ssd_layers = []

        for i in self.ssd_config:
            in_channels = i[0]
            out_channels = i[1]
            stride = i[2]
            ssd_layers.append(MobileNetSSD.SSD(in_channels, out_channels, stride))

        return nn.Sequential(*ssd_layers)

    def forward(self, x):
        out = f.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = f.relu(self.trans_bn1(self.trans_conv1(out)))
        out = f.relu(self.trans_bn2(self.trans_conv2(out)))
        out = self.ssd_layers(out).view(-1, 256)
        out = self.linear(out)

        return out

    class Depthwise(nn.Module):
        def __init__(self, in_channels, out_channels, stride):
            super().__init__()

            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                   kernel_size=(3, 3), stride=stride, padding=1, groups=in_channels, bias=False)
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)

        def forward(self, x):
            out = f.relu(self.bn1(self.conv1(x)))
            out = f.relu(self.bn2(self.conv2(out)))
            return out

    class SSD(nn.Module):
        def __init__(self, in_channels, out_channels, stride):
            super().__init__()

            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=(1, 1), stride=(stride, stride), padding=0)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                   kernel_size=(3, 3), stride=(stride, stride), padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)

        def forward(self, x):
            out = f.relu(self.bn1(self.conv1(x)))
            out = f.relu(self.bn2(self.conv2(out)))
            return out


## create Variant-LSTM
# variants in LSTM (2 LSTM layers with dropout in each layer)
class LSTM_dropout(nn.Module):
    """
    Param:
    input_size: feature size
    hidden_size: the number of hidden layers
    output_size: the number of output
    num_layers: the number of layers for LSTM
    dropout: the probability ratio of dropout
    """

    def __init__(self, input_size, output_size, hidden_size=32, num_layers=2, dropout=0.2):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        # size, batch, hidden = out.shape
        # out = out.view(batch, size * hidden)
        # out = self.flatten(out)
        out = self.linear(out[:,-1])
        return out

    
# stack 2 convolution layers + maxpooling + conv, LSTM with softmax
class LSTM_CONV(nn.Module):
    """
    Param:
    input_size: feature size
    hidden_size: the number of hidden layers
    output_size: the number of output
    num_layers: the number of layers for LSTM
    """

    def __init__(self, input_size, output_size, num_layers=3, hidden_size=32):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.conv = nn.Sequential(OrderedDict([("conv1", nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=9,padding=4)),
                                                   ("relu_1", nn.ReLU()),
                                                   ("maxpool1", nn.MaxPool1d(2,2)),
                                                   ("conv2", nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=9,padding=4)),
                                                   ("relu_2", nn.ReLU()),
                                                   ("maxpool2", nn.MaxPool1d(2,2)),
                                                   ("conv3", nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=9,padding=4))
                                                   ]))
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.bn(x)
        out = self.conv(x)
        out = out.permute(0, 2, 1)
        out,_ = self.lstm(out)
        # out = self.flatten(out[:,-1])
        # out = self.softmax(self.linear(out))
        out = self.linear(out[:,-1])
        return out


# dual_lstm also late fusion
class Dual_LSTM(nn.Module):
    """
    Param:
    input_size: feature size
    hidden_size: the number of hidden layers
    output_size: the number of output
    num_layers: the number of layers for LSTM
    """

    def __init__(self, input_size, output_size, hidden_size=32, num_layers=3):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.lstm_1 = nn.LSTM(26, hidden_size, num_layers, batch_first=True)
        self.lstm_2 = nn.LSTM(4, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        out_1, _ = self.lstm_1(x[:,:,:26])
        out_2, _ = self.lstm_2(x[:,:,26:])
        # out = self.linear(out[:,-1])
        # out = self.softmax(out)
        # out = self.flatten(out[:,-1])
        out = torch.cat((out_1[:,-1], out_2[:,-1]), 1)
        out = self.linear(out)
        return out
    

class SE_Block1D(nn.Module):
    def __init__(self, c, k=32, r=16):
        super().__init__()
        self.k = k
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n,_ = x.shape
        w = self.excitation(x)
        w, index = w.sort(dim=-1, descending=True)
        w = w[:,:self.k]
        index = torch.where(index<self.k)
        x = x[index].reshape(n, -1)
  
        return x * w

# dual_lstm also late fusion
class SE_LSTM(nn.Module):
    """
    Param:
    input_size: feature size
    hidden_size: the number of hidden layers
    output_size: the number of output
    num_layers: the number of layers for LSTM
    """

    def __init__(self, input_size, output_size, hidden_size=32, num_layers=3, scalar=1.5):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.lstm_1 = nn.LSTM(26, int(24*scalar), num_layers, batch_first=True)
        self.lstm_2 = nn.LSTM(4,  int(8*scalar), num_layers, batch_first=True)
        self.se = SE_Block1D(int(hidden_size*scalar), k=32)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        out_1, _ = self.lstm_1(x[:,:,:26])
        out_2, _ = self.lstm_2(x[:,:,26:])
        out = torch.cat((out_1[:,-1], out_2[:,-1]), 1)
        out = self.se(out)
        out = self.linear(out)
        return out