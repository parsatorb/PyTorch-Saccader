"""
Module for the representation network, aka the bagnet-77-lowD newtwork
found in https://arxiv.org/pdf/1908.07644.pdf
"""

import torch.nn as nn
import torch.nn.functional as F
import math

class Bottleneck(nn.Module):
    """
    Small yellow boxes from the figure in page 16 of https://arxiv.org/pdf/1908.07644.pdf
    
    Note: A Bottleneck outputs 4 times as many channels as the planes parameter
    """

    def __init__(self, inplanes, planes, stride=2, kernel_size=1):
        super().__init__()
        conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False, padding=0)
        bn1 = nn.BatchNorm2d(planes)

        conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride,
                               padding=0, bias=False)
        bn2 = nn.BatchNorm2d(planes)

        conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, stride=1, bias=False, padding=0)
        bn3 = nn.BatchNorm2d(planes * 4)

        relu1 = nn.ReLU(inplace=True)
        relu2 = nn.ReLU(inplace=True)
        relu3 = nn.ReLU(inplace=True)

        self.module_list = nn.ModuleList([
            conv1, bn1, relu1, conv2, bn2, relu2, conv3, bn3, relu3
        ])

    def forward(self, out, **kwargs):
        for module in self.module_list:
            out = module(out)

        return out

class Layer(nn.Module):
    def __init__(self, block, inplanes, planes, stride, repeat):
        super().__init__()
        self.block1 = block(inplanes=inplanes, planes=planes, stride=stride)
        self.block2 = block(inplanes=4*planes, planes=planes, stride=1)

        self.resblock = nn.Conv2d(inplanes, planes*4, kernel_size=1, stride=stride,
                                    padding=0, bias=False)
        self.res_bn = nn.BatchNorm2d(inplanes*4)

        self.relu = nn.ReLU(inplace=True)

        self.module_lists = nn.ModuleList([
            self.block1, self.block2, self.resblock, self.res_bn, self.relu    
        ])

        self.repeat_section = []
        for i in range(repeat):
            _block = block(inplanes=4*planes, planes=planes, stride=1)
            self.module_lists.extend([_block])
            self.repeat_section.append(_block)
            
    def forward(self, x, **kwargs):
        out = self.block1(x)

        residual = self.resblock(x)
        residual = self.res_bn(residual)
        residual = self.relu(residual)

        out += residual
        residual = out

        out = self.block2(out)

        for section in self.repeat_section:
            out += residual
            #residual = out
            out = section(out)
            
        return out

class BagNet77(nn.Module):
    """
    From the diagram on page 16 of https://arxiv.org/pdf/1908.07644.pdf
    Also known as the representation network
    """

    def __init__(self, repeats=[1, 2, 4, 1], strides=[2, 2, 2, 1], n_classes=1000):
        super(BagNet77, self).__init__()
        conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0, bias=False)
        bn1 = nn.BatchNorm2d(64)
        relu = nn.ReLU(inplace=True)

        layer1 = Layer(Bottleneck, inplanes=64, planes=64, stride=strides[0], repeat=repeats[0])
        layer2 = Layer(Bottleneck, inplanes=256, planes=128, stride=strides[1], repeat=repeats[1])
        layer3 = Layer(Bottleneck, inplanes=512, planes=256, stride=strides[1], repeat=repeats[2])
        layer4 = Layer(Bottleneck, inplanes=1024, planes=512, stride=strides[1], repeat=repeats[3])
        
        conv_what = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        bn2 = nn.BatchNorm2d(512)
        relu2 = nn.ReLU(inplace=True)
        conv_classes = nn.Conv2d(512, 1000, kernel_size=1, stride=1, padding=0, bias=False)
        bn3 = nn.BatchNorm2d(1000)
        relu3 = nn.ReLU(inplace=True)
        
        self.modules_list = nn.ModuleList([
            conv1, bn1, relu, layer1, layer2, layer3, layer4, 
            conv_what, bn2, relu2, conv_classes, bn3, relu3
        ])

        #Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, out, **kwargs):
        
        for m in self.modules_list:
            out = m(out)

        return F.adaptive_avg_pool2d(out, (1, 1))
