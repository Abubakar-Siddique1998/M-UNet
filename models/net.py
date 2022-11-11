import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50 
resnet=resnet50(pretrained=True)
import torch.nn.functional as F

from collections import OrderedDict
import os

import torch
import torch.nn as nn


def conv_batch(in_num, out_num, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU())
class PSD(nn.Module):
    def __init__(self,in_channels,channels):
        super(PSD,self).__init__()

        upsampling=[
                   nn.PixelShuffle(upscale_factor=2),
                   nn.Conv2d(channels,channels,3,1,1),
                   nn.BatchNorm2d(channels),
                   nn.PReLU(),
                   ]
        self.upsampling=nn.Sequential(*upsampling)
    def forward(self,x):

        x=self.upsampling(x)

        return x


class MFE(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(MFE,self).__init__()

        reduced_channels = int(in_channel/4)
        scaled_channel=int(reduced_channels*3)
        self.layer1=nn.Conv2d(in_channel,reduced_channels,kernel_size=1,padding=0)

        self.layer2=nn.Conv2d(reduced_channels,reduced_channels,kernel_size=3,padding=1)
        
        self.layer3=nn.Conv2d(reduced_channels,reduced_channels,kernel_size=5,padding=2)
        
        self.layer4=nn.Conv2d(reduced_channels,reduced_channels,kernel_size=7,padding=3)#96
        self.layer7 = conv_batch(scaled_channel, out_channel,kernel_size=1,padding=0)#64
    def forward(self,x):
        residual =x
        x=self.layer1(x)
  
        out2=self.layer2(x)
        out3=self.layer3(x)
        out4=self.layer4(x)
        out=torch.cat((out2,out3,out4),1)
        out=self.layer7(out)
        out+=residual
        return out

def weight_xavier_init(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                # nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

class MUNET(nn.Module):
    def __init__(self,block,N_classes=6):
        super(MUNET,self).__init__()
        self.num_classes=N_classes
        self.layer0=nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.downsmaple=resnet.maxpool
        self.layer1, self.layer2, self.layer3 = resnet.layer1, resnet.layer2, resnet.layer3

        self.residual_block1 = self.make_layer(block, in_channels=64, out_channel=64, num_blocks=1)
        self.residual_block2 = self.make_layer(block, in_channels=256,out_channel=256, num_blocks=1)
        self.residual_block3 = self.make_layer(block, in_channels=512,out_channel=512, num_blocks=1)
        self.residual_block4 = self.make_layer(block, in_channels=1024,out_channel=1024, num_blocks=1)
        self.center=nn.Sequential(conv_batch(1024,1024,kernel_size=1,padding=0))
        self.decoder4 = PSD(1024,256)
        self.decoder3 = PSD(768,192) 
        self.decoder2 = PSD(448,112)
        self.decoder1 = PSD(176,44)
        
        self.logit = nn.Sequential(conv_batch(44,6,3,1,1))
        weight_xavier_init(self.residual_block1, self.residual_block2, 
                           self.residual_block3,self.residual_block4,self.center)
        
    def forward(self,x):
        x=self.layer0(x) ##3 => 64
        e1=self.residual_block1(x) ##64 => 64
        e2=self.downsmaple(e1)
        e2=self.layer1(e2) #64=>256
        e2=self.residual_block2(e2) ##256 =>256
        e3=self.layer2(e2) #256 =>512
        e3=self.residual_block3(e3) ##512 =>512
        e4=self.layer3(e3) ## 512 => 1024
        e4=self.residual_block4(e4)
        f = self.center(e4)
        

        
        d4 = self.decoder4(f) ##image size ==32 channel==256
        d3 = self.decoder3(torch.cat([d4, e3], 1)) ##image size == 64 channel==192
        d2 = self.decoder2(torch.cat([d3,e2],1))
        d1=  self.decoder1(torch.cat([d2,e1],1))

      
        logit=self.logit(d1)
        return logit
        
    
    def make_layer(self, block, in_channels,out_channel, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels,out_channel))
        return nn.Sequential(*layers)
def munet():
    return MUNET(MFE)
