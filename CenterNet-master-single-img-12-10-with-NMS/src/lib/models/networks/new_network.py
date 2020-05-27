# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import numpy as np
from os.path import join
from time import *

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable

# from DCNv2.dcn_v2 import DCN

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class New_Network_Backbone(nn.Module):
    def __init__(self):
        super(New_Network_Backbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7,
                               stride=1, padding=3,
                               bias=False, dilation=1)
        self.bn1 = nn.BatchNorm2d(16, momentum=BN_MOMENTUM)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2_1 = nn.Conv2d(16, 16, kernel_size=3,
                               stride=2, padding=1,
                               bias=False, dilation=1)
        self.conv2_2 = nn.Conv2d(16, 16, kernel_size=3,
                                 stride=2, padding=2,
                                 bias=False, dilation=2)
        self.conv2_3 = nn.Conv2d(16, 16, kernel_size=3,
                                 stride=2, padding=3,
                                 bias=False, dilation=3)
        self.bn2 = nn.BatchNorm2d(48, momentum=BN_MOMENTUM)  # 特征图·拼接之后再进行bn和relu操作
        self.relu2 = nn.ReLU(inplace=True)
        # concatnate

        self.conv3 = nn.Conv2d(48, 32, kernel_size=1,
                               stride=2, padding=0,
                               bias=False, dilation=1)
        self.bn3 = nn.BatchNorm2d(32, momentum=BN_MOMENTUM)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3,
                               stride=1, padding=1,
                               bias=False, dilation=1)
        self.bn4 = nn.BatchNorm2d(32, momentum=BN_MOMENTUM)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=5,
                               stride=1, padding=2,
                               bias=False, dilation=1)
        self.bn5 = nn.BatchNorm2d(32, momentum=BN_MOMENTUM)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(32, 32, kernel_size=7,
                               stride=1, padding=3,
                               bias=False, dilation=1)
        # conv6 不需要做bn和relu，因为conv6直接和前面几个conv层cat起来，没有继续向前传播
        # 下面的bn6和relu6是给cat之后的特征图准备的
        self.bn6 = nn.BatchNorm2d(128, momentum=BN_MOMENTUM)
        self.relu6 = nn.ReLU(inplace=True)
        # concatnate

        self.conv7 = nn.Conv2d(128, 64, kernel_size=1,
                               stride=1, padding=0,
                               bias=False, dilation=1)
        self.bn7 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu7 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x_1 = self.conv2_1(x)
        x_2 = self.conv2_2(x)
        x_3 = self.conv2_3(x)

        # concatnate
        x = torch.cat((x_1, x_2, x_3), dim=1)  # ???????????????????dims=2
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        conv3_x = x   # 如果要cat，那就是直接cat 经过conv之后的特征图，不能经过bn和relu
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        conv4_x = x
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        conv5_x = x
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        conv6_x = x

        # concatnate
        x = torch.cat([conv3_x, conv4_x, conv5_x, conv6_x], dim=1)
        x = self.bn6(x)
        x = self.relu6(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu7(x)

        return x

class New_Network_FCA(nn.Module):
    def __init__(self, inplanes):
        super(New_Network_FCA, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=3,
                               stride=1, padding=1,
                               bias=False, dilation=1)
        self.bn1 = nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(inplanes, inplanes, kernel_size=3,
                               stride=1, padding=2,
                               bias=False, dilation=2)

        # bn2 relu2是给经过cat的conv1和conv2用的
        self.bn2 = nn.BatchNorm2d(2*inplanes, momentum=BN_MOMENTUM)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(2*inplanes, 1, kernel_size=3,
                               stride=1, padding=1,
                               bias=False, dilation=1)
        self.bn3 = nn.BatchNorm2d(1, momentum=BN_MOMENTUM)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print('     input:', x.size())
        resblock = x
        x = self.conv1(x)
        conv1_x = x   # 如果要cat，那就是直接cat 经过conv之后的特征图，不能经过bn和relu
        x = self.bn1(x)
        x = self.relu1(x)
        # print('     conv1_x:', x.size())

        x = self.conv2(x)
        conv2_x = x

        x = torch.cat([conv1_x, conv2_x], dim=1)
        x = self.bn2(x)
        x = self.relu2(x)
        # print('     cat:', x.size())

        x = self.conv3(x)
        x = self.bn3(x)
        # x = self.sigmoid(x)
        scale = F.sigmoid(x)  # broadcasting   # ???????????????????????????????????
        # print('     scale:', x.size())
        # print('     resblock*scale:', (resblock*scale).size())

        return resblock*scale

class New_Network_CFA(nn.Module):
    def __init__(self, inplanes):
        super(New_Network_CFA, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=1,
                               stride=1, padding=0,
                               bias=False, dilation=1)
        self.bn1 = nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(inplanes, inplanes, kernel_size=1,
                               stride=1, padding=0,
                               bias=False, dilation=1)
        # self.bn2 = nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM)
        # self.relu2 = nn.ReLU(inplace=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        resblock = x
        # print('     input:', x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # print('     conv1:', x.size())

        avg_pool_x = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))  # ????????????
        avg_pool_x = self.conv2(avg_pool_x)
        # print('     avg_pool_x:', avg_pool_x.size())

        max_pool_x = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))  # ????????????
        max_pool_x = self.conv2(max_pool_x)
        # print('     max_pool_x:', avg_pool_x.size())

        x = avg_pool_x + max_pool_x
        # x = self.bn2(x)     # x： [1, 256, 1, 1]，每个通道的尺寸是1×1,所以不需要BN
        # x = self.sigmoid(x)
        scale = F.sigmoid( x ).expand_as(resblock)  # ??????????????????????????????????????
        # print('     scale:', scale.size())
        # print('     resblock*scale:', (resblock*scale).size())

        return resblock*scale

class New_Network_ResBlock(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(New_Network_ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes//2, kernel_size=1,
                               stride=1, padding=0,
                               bias=False, dilation=1)
        self.bn1 = nn.BatchNorm2d(outplanes//2, momentum=BN_MOMENTUM)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outplanes//2, outplanes, kernel_size=3,
                               stride=1, padding=1,
                               bias=False, dilation=1)
        self.bn2 = nn.BatchNorm2d(outplanes, momentum=BN_MOMENTUM)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(outplanes, outplanes // 2, kernel_size=1,
                               stride=1, padding=0,
                               bias=False, dilation=1)
        self.bn3 = nn.BatchNorm2d(outplanes // 2, momentum=BN_MOMENTUM)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(outplanes // 2, outplanes, kernel_size=1,
                               stride=1, padding=0,
                               bias=False, dilation=1)
        self.bn4 = nn.BatchNorm2d(outplanes, momentum=BN_MOMENTUM)
        self.relu4 = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        conv2_x = x
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        conv4_x = x
        x = conv2_x + conv4_x
        x = self.bn4(x)
        x = self.relu4(x)

        return x


class New_Network(nn.Module):
    def __init__(self):
        super(New_Network, self).__init__()
        self.backbone = New_Network_Backbone()
        self.resblock1_1 = New_Network_ResBlock(inplanes=64, outplanes=64)
        self.resblock1_2 = New_Network_ResBlock(inplanes=64, outplanes=64)
        self.fca1 = New_Network_FCA(inplanes=64)

        self.resblock2_1 = New_Network_ResBlock(inplanes=64, outplanes=128)
        self.resblock2_2 = New_Network_ResBlock(inplanes=128, outplanes=128)
        self.fca2 = New_Network_FCA(inplanes=128)

        self.resblock3_1 = New_Network_ResBlock(inplanes=128, outplanes=256)
        self.resblock3_2 = New_Network_ResBlock(inplanes=256, outplanes=256)
        self.cfa = New_Network_CFA(inplanes=256)

        self.conv = nn.Conv2d(256, 64, kernel_size=1,
                               stride=1, padding=0,
                               bias=False, dilation=1)
        self.bn = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.backbone(x)
        # print('backbone:', x.size())
        x = self.resblock1_1(x)
        x = self.resblock1_2(x)
        # print('resblock1:', x.size())
        x = self.fca1(x)
        # print('fca1:', x.size())

        x = self.resblock2_1(x)
        x = self.resblock2_2(x)
        # print('resblock2:', x.size())
        x = self.fca2(x)
        # print('fca2:', x.size())

        x = self.resblock3_1(x)
        x = self.resblock3_2(x)
        # print('resblock3:', x.size())
        x = self.cfa(x)
        # print('cfa:', x.size())

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        # print('output:', x.size())

        return x

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class New_Network_to_dla(nn.Module):
    def __init__(self, num_layers, heads, head_conv):
        super(New_Network_to_dla, self).__init__()
        self.new_network = New_Network()
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(64, head_conv,
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes,
                              kernel_size=1, stride=1,
                              padding=0, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(64, classes,
                               kernel_size=1, stride=1,
                               padding=0, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        x = self.new_network(x)
        # print('after_new_ntework', x.size())
        ret = {}
        # print('self.heads', self.heads)
        for head in self.heads:
            # print(head, 'before:', x.size())
            # print(head, 'struture:', self.__getattr__(head))
            ret[head] = self.__getattr__(head)(x)
            # print(head, 'after:', ret[head].size())
        return [ret]



def get_pose_net(num_layers, heads, head_conv=256):
    model = New_Network_to_dla(num_layers, heads, head_conv=head_conv)
    return model



# strat = time()
# model = get_pose_net(num_layers=0, heads={'hm': 1, 'wh': 2, 'reg': 2}, head_conv=256)
# out = model(Variable(torch.randn(1,3,512,512)))
# cost = time() - strat
# print('cost:', cost)

