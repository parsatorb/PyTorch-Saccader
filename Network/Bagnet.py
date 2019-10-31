"""
Module for the representation network, aka the bagnet-77-lowD newtwork
found in https://arxiv.org/pdf/1908.07644.pdf
"""

import torch.nn as nn
import math
import torch
from collections import OrderedDict

class Bottleneck(nn.Module):
    """
    Small yellow boxes from the figure in page 16 of https://arxiv.org/pdf/1908.07644.pdf
    """

    def __init__(self, inplanes, planes, stride=2, kernel_size=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False, padding=0)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, stride=1, bias=False, padding=0)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, **kwargs):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        return out

class Layer(nn.Module):
    def __init__(self, block, inplanes, planes, stride, repeat):
        self.block1 = block(inplanes=inplanes, planes=planes, stride=stride)
        self.block2 = block(inplanes=planes, planes=planes, stride=1)

        self.resblock = nn.Conv2d(inplanes, planes*4, kernal_size=1, stide=stride,
                                    padding=0, bias=False)
        self.res_bn = nn.BatchNorm2d(inplanes*4)
        self.relu = nn.ReLU(inplace=True)

        self.repeat_section = []
        for i in range(repeat):
            self.repeat_section.append(block(inplanes=planes, planes=planes, stride=1))

    def foward(self, x):
        residual1 = x

        out = self.block1(x)

        residual = self.resblock(residual)
        residual = self.res_bn(residual)
        residual = self.relu(residual)

        out += residual
        residual = out

        out = self.block2(out)

        for section in self.repeat_section:
            out += residual
            residual = out
            out = section(out)

        return out

class BagNet(nn.Module):
    """
    From the diagram on page 16 of https://arxiv.org/pdf/1908.07644.pdf
    Also known as the representation network
    """

    def __init__(self, repeats=[1, 2, 4, 1], strides=[2, 2, 2, 1]):
        self.inplanes = 64
        super(BagNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = Layer(Bottleneck, inplanes=64, planes=64, stride=strides[0], repeat=repeats[0])
        self.layer2 = Layer(Bottleneck, inplanes=256, planes=128, stride=strides[1], repeat=repeats[1])
        self.layer3 = Layer(Bottleneck, inplanes=512, planes=256, stride=strides[1], repeat=repeats[2])
        self.layer4 = Layer(Bottleneck, inplanes=1024, planes=512, stride=strides[1], repeat=repeats[3])

        #Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out
