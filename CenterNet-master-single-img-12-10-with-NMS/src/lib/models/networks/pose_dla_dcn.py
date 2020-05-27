from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import numpy as np
from os.path import join
import time

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable

from src.lib.models.utils import _sigmoid

from .DCNv2.dcn_v2 import DCN

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    # DLA([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512], block=BasicBlock, **kwargs)
    def __init__(self, levels, channels, num_classes=1000, block=BasicBlock, residual_root=False, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        # print('input:', x.size())
        x = self.base_layer(x)
        # print('after baselayer:', x.size())
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            # print(i, x.size())
            y.append(x)
        # print('y:', len(y))
        return y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        # fc = self.fc
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights)
        # self.fc = fc


def dla34(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
    # print(model)
    return model


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):
    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = DeformConv(c, o)
            node = DeformConv(o, o)
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f,
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        # print(222222222222, len(layers), startp, endp)
        feature_128_get_from_IDAUp = None
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            # print(3333333, i, layers[i].size())
            layers[i] = upsample(project(layers[i]))  # project(*)操作将特征图的深度减半， upsample(*)操作将特征图的尺寸扩大一倍
            # upsample(project(*)), 经过这两步操作，将layers[i]的特征图转换成了和layers[i-1]相同深度相同尺寸
            # print(4444444, i, layers[i].size(), layers[i - 1].size())
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])  # node(*)就是将layers[i] + layers[i - 1]后的结果做了简单的DCN操作
            # print(5555555, i, layers[i].size())
            if i==3:
                feature_128_get_from_IDAUp = layers[i].clone()  # clone() :这个函数将被克隆的tensor记录在计算图中。传递到克隆张量的梯度将传播到原始张量。
                # print('feature_128_get_from_IDAUp:', feature_128_get_from_IDAUp.size())
            # if len(layers)==6:
            #     print(len(layers), layers[0].size(), layers[1].size(), layers[2].size(), layers[3].size(), layers[4].size(), layers[5].size())
            # if len(layers)==3:
            #     print(len(layers), layers[0].size(), layers[1].size(), layers[2].size())
        return feature_128_get_from_IDAUp


class FPN_out(nn.Module):
    def __init__(self):
        super(FPN_out, self).__init__()  # ！！！！！！！！！！！！ 这里各个卷积层的通道数对效果的影响有待确定
        self.down_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.node_1 = DeformConv(128, 64)
        self.down_2 = nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1, dilation=1)
        self.node_2 = DeformConv(256, 64)
        self.down_3 = nn.Conv2d(64, 512, kernel_size=3, stride=2, padding=1, dilation=1)
        self.node_3 = DeformConv(512, 64)

    def forward(self, x):
        x_d4 = x[0]
        x_d8 = self.node_1(self.down_1(x[0]) + x[1])
        x_d16 = self.node_2(self.down_2(x_d8) + x[2])
        x_d32 = self.node_3(self.down_3(x_d16) + x[3])
        # print('!!!!!!!!!!!!!FPN_Out:', [x_d4.size(), x_d8.size(), x_d16.size(), x_d32.size()])
        return [x_d4, x_d8, x_d16, x_d32]


class DLAUp(nn.Module):  # DLAUp(  2,    [64, 128, 256, 512],    [1, 2, 4, 8])
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i), IDAUp(channels[j], in_channels[j:], scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]]  # start with 16*16
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            feature_128_get_from_IDAUp = ida(layers, len(layers) - i - 2, len(layers)) # !!!!!!!
            out.insert(0, layers[-1])
        return out, feature_128_get_from_IDAUp


class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
        return x


class DLASeg_ori(nn.Module):
    def __init__(self, base_name, heads, pretrained, down_ratio, final_kernel,
                 last_level, head_conv, out_channel=0):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]  # down_ratio=4
        self.first_level = int(np.log2(down_ratio))  # self.first_level = 2
        self.last_level = last_level  # self.last_level = 5
        self.base = globals()[base_name](pretrained=pretrained)  # globals() 函数会以字典类型返回当前位置的全部全局变量。
        # self.base = dla34(pretrained=pretrained)
        channels = self.base.channels  # channels = [16, 32, 64, 128, 256, 512]
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]  # scales = [1, 2, 4, 8]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)
        #             DLAUp(  2,              [64, 128, 256, 512],         [1, 2, 4, 8])

        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level],
                            [2 ** i for i in range(self.last_level - self.first_level)])

        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(channels[self.first_level], head_conv,
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes,
                              kernel_size=final_kernel, stride=1,
                              padding=final_kernel // 2, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(channels[self.first_level], classes,
                               kernel_size=final_kernel, stride=1,
                               padding=final_kernel // 2, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        print('input:', x.size())
        x = self.base(x)
        print('after_base:', type(x), len(x), x[0].size(), x[1].size(), x[2].size(), x[3].size(), x[4].size(),
              x[5].size())
        x = self.dla_up(x)
        print('after_dla_up:', type(x), len(x), x[0].size(), x[1].size(), x[2].size(), x[3].size())

        y = []
        for i in range(self.last_level - self.first_level):  # 5-2
            y.append(x[i].clone())
        print('before_ida_up:', type(y), len(y), y[0].size(), y[1].size(), y[2].size())
        self.ida_up(y, 0, len(y))
        print('after_ida_up:', type(y), len(y), y[0].size(), y[1].size(), y[2].size())
        # y[0]->y[1]->y[2]聚合度越来越高，却接近图表中的OUT标志

        z = {}
        print('self.heads:', self.heads)
        for head in self.heads:
            print(head, 'before:', y[-1].size())
            print(head, 'struture:', self.__getattr__(head))
            z[head] = self.__getattr__(head)(y[-1])
            print(head, 'after:', z[head].size())
        return [z]


class DLASeg_fpn(nn.Module):  # 这个网络是将原本的3个输出作为FPN结构
    def __init__(self, base_name, heads, pretrained, down_ratio, final_kernel,
                 last_level, head_conv, out_channel=0):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]  # down_ratio=4
        self.first_level = int(np.log2(down_ratio))  # self.first_level = 2
        self.last_level = last_level  # self.last_level = 5
        self.base = globals()[base_name](pretrained=pretrained)  # globals() 函数会以字典类型返回当前位置的全部全局变量。
        # self.base = dla34(pretrained=pretrained)
        channels = self.base.channels  # channels = [16, 32, 64, 128, 256, 512]
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]  # scales = [1, 2, 4, 8]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)
        #             DLAUp(  2,              [64, 128, 256, 512],         [1, 2, 4, 8])

        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level],
                            [2 ** i for i in range(self.last_level - self.first_level)])

        # self.get_fpn_out_layers = FPN_out()

        # self.up_delta_conv_0 = nn.Conv2d(64, 64, kernel_size=3, stride=(1, 2), padding=1, bias=False, dilation=1)
        # # self.up_delta_conv_1 = nn.Conv2d(64, 64, kernel_size=3, stride=(1, 2), padding=1, bias=False, dilation=1)
        # # self.up_delta_conv_2 = nn.Conv2d(64, 64, kernel_size=3, stride=(1, 2), padding=1, bias=False, dilation=1)
        #
        # self.btm_delta_conv_0 = nn.Conv2d(64, 64, kernel_size=3, stride=(2, 1), padding=1, bias=False, dilation=1)
        # # self.btm_delta_conv_1 = nn.Conv2d(64, 64, kernel_size=3, stride=(2, 1), padding=1, bias=False, dilation=1)
        # # self.btm_delta_conv_2 = nn.Conv2d(64, 64, kernel_size=3, stride=(2, 1), padding=1, bias=False, dilation=1)
        #
        # self.transconv_y0_up = nn.ConvTranspose2d(64, 64, (1, 4), stride=(1, 4), padding=0, output_padding=0, groups=64,
        #                                           bias=False)  #  !!注意groups参数！！
        # self.transconv_y0_btm = nn.ConvTranspose2d(64, 64, (4, 1), stride=(4, 1), padding=0, output_padding=0, groups=64,
        #                                           bias=False)
        # self.transconv_y1_up = nn.ConvTranspose2d(64, 64, (2, 8), stride=(2, 8), padding=0, output_padding=0, groups=64,
        #                                           bias=False)
        # self.transconv_y1_btm = nn.ConvTranspose2d(64, 64, (8, 2), stride=(8, 2), padding=0, output_padding=0, groups=64,
        #                                           bias=False)
        # self.transconv_y2_up = nn.ConvTranspose2d(64, 64, (4, 16), stride=(4, 16), padding=0, output_padding=0, groups=64,
        #                                           bias=False)
        # self.transconv_y2_btm = nn.ConvTranspose2d(64, 64, (16, 4), stride=(16, 4), padding=0, output_padding=0,
        #                                            groups=64, bias=False)
        # self.transconv_fpn1 = nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0, output_padding=0, groups=64,
        #                                          bias=False)
        # self.transconv_fpn2 = nn.ConvTranspose2d(64, 64, 4, stride=4, padding=0, output_padding=0, groups=64,
        #                                          bias=False)
        # self.transconv_fpn3 = nn.ConvTranspose2d(64, 64, 8, stride=8, padding=0, output_padding=0, groups=64,
        #                                          bias=False)

        # fill_up_weights(self.transconv_y0_up)
        # fill_up_weights(self.transconv_y0_btm)
        # fill_up_weights(self.transconv_y1_up)
        # fill_up_weights(self.transconv_y1_btm)
        # fill_up_weights(self.transconv_y2_up)
        # fill_up_weights(self.transconv_y2_btm)
        # fill_up_weights(self.transconv_fpn1)
        # fill_up_weights(self.transconv_fpn2)
        # fill_up_weights(self.transconv_fpn3)

        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(channels[self.first_level], head_conv,
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes,
                              kernel_size=final_kernel, stride=1,
                              padding=final_kernel // 2, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(channels[self.first_level], classes,
                               kernel_size=final_kernel, stride=1,
                               padding=final_kernel // 2, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        print('input:', x.size())
        x = self.base(x)
        print('after_base:', type(x), len(x), x[0].size(), x[1].size(), x[2].size(), x[3].size(), x[4].size(), x[5].size())
        x = self.dla_up(x)
        print('after_dla_up:', type(x), len(x), x[0].size(), x[1].size(), x[2].size(), x[3].size())

        # fpn_in = x[:]
        # fpn_out = self.get_fpn_out_layers(fpn_in)

        # transconv_fpn0 = fpn_out[0]
        # transconv_fpn1 = self.transconv_fpn1(fpn_out[1])
        # transconv_fpn2 = self.transconv_fpn2(fpn_out[2])
        # transconv_fpn3 = self.transconv_fpn3(fpn_out[3])
        # transconv_fpn_out = [transconv_fpn0, transconv_fpn1, transconv_fpn2, transconv_fpn3]

        # y0_up_1 = self.up_delta_conv_0(fpn_out[0])
        # # print('y0_up_1:', y0_up_1.size())
        # y0_up_2 = self.up_delta_conv_0(y0_up_1)
        # # print('y0_up_2:', y0_up_2.size())
        # y0_up_out = self.transconv_y0_up(y0_up_2)
        # # print('y0_up_out:', y0_up_out.size())
        # y0_btm_1 = self.btm_delta_conv_0(fpn_out[0])
        # # print('y0_btm_1:', y0_btm_1.size())
        # y0_btm_2 = self.btm_delta_conv_0(y0_btm_1)
        # # print('y0_btm_2:', y0_btm_2.size())
        # y0_btm_out = self.transconv_y0_btm(y0_btm_2)
        # # print('y0_btm_out:', y0_btm_out.size())
        #
        # y1_up_1 = self.up_delta_conv_0(fpn_out[1])
        # # print('y1_up_1:', y1_up_1.size())
        # y1_up_2 = self.up_delta_conv_0(y1_up_1)
        # # print('y1_up_2:', y1_up_2.size())
        # y1_up_out = self.transconv_y1_up(y1_up_2)
        # # print('y1_up_out:', y1_up_out.size())
        # y1_btm_1 = self.btm_delta_conv_0(fpn_out[1])
        # # print('y1_btm_1:', y1_btm_1.size())
        # y1_btm_2 = self.btm_delta_conv_0(y1_btm_1)
        # # print('y1_btm_2:', y1_btm_2.size())
        # y1_btm_out = self.transconv_y1_btm(y1_btm_2)
        # # print('y1_btm_out:', y1_btm_out.size())
        #
        # y2_up_1 = self.up_delta_conv_0(fpn_out[2])
        # # print('y2_up_1:', y2_up_1.size())
        # y2_up_2 = self.up_delta_conv_0(y2_up_1)
        # # print('y2_up_2:', y2_up_2.size())
        # y2_up_out = self.transconv_y2_up(y2_up_2)
        # # print('y2_up_out:', y2_up_out.size())
        # y2_btm_1 = self.btm_delta_conv_0(fpn_out[2])
        # # print('y2_btm_1:', y2_btm_1.size())
        # y2_btm_2 = self.btm_delta_conv_0(y2_btm_1)
        # # print('y2_btm_2:', y2_btm_2.size())
        # y2_btm_out = self.transconv_y2_btm(y2_btm_2)
        # # print('y2_btm_out:', y2_btm_out.size())
        #
        # xnet_out = [y0_up_out, y0_btm_out, y1_up_out, y1_btm_out, y2_up_out, y2_btm_out]


        y = []
        for i in range(self.last_level - self.first_level):  # 5-2
            y.append(x[i].clone())
        print('before_ida_up:', type(y), len(y), y[0].size(), y[1].size(), y[2].size())
        self.ida_up(y, 0, len(y))
        print('after_ida_up:', type(y), len(y), y[0].size(), y[1].size(), y[2].size())
        # y[0]->y[1]->y[2]聚合度越来越高，却接近图表中的OUT标志

        final_out = []

        dla_z = {}
        # print('self.heads:', self.heads)
        for head in self.heads:
            # print(head, 'before:', y[-1].size())
            # print(head, 'struture:', self.__getattr__(head))
            dla_z[head] = self.__getattr__(head)(y[-1])
            # print(head, 'after:', z[head].size())
        dla_z['hm'] = _sigmoid(dla_z['hm'])
        final_out.append(dla_z)    # 1个输出
        # # return [dla_z]

        # 这是直接从原始的dla网络中得到的金字塔网络
        # for out0 in y:   #　 y[0]->y[1]->y[2] ,虽然特征图尺寸一致，但是y[0]中特征图高分辨率信息多，应该用来回归小物体；y[2]用来回归大物体
        #     dla_fpn_z = {}
        #     # print('self.heads:', self.heads)
        #     for head in self.heads:
        #         # print(head, 'before:', y[-1].size())
        #         # print(head, 'struture:', self.__getattr__(head))
        #         dla_fpn_z[head] = self.__getattr__(head)(out0)
        #         # print(head, 'after:', z[head].size())
        #     dla_fpn_z['hm'] = _sigmoid(dla_fpn_z['hm'])
        #     final_out.append(dla_fpn_z)  # 3个输出

        # for out1 in transconv_fpn_out:
        #     fpn_z = {}
        #     # print('self.heads:', self.heads)
        #     for head in self.heads:
        #         # print(head, 'before:', y[-1].size())
        #         # print(head, 'struture:', self.__getattr__(head))
        #         fpn_z[head] = self.__getattr__(head)(out1)
        #         # print(head, 'after:', fpn_z[head].size())
        #     fpn_z['hm'] = _sigmoid(fpn_z['hm'])
        #     final_out.append(fpn_z)  # 4个输出

        # for out2 in xnet_out:
        #     xnet_z = {}
        #     # print('self.heads:', self.heads)
        #     for head in self.heads:
        #         # print(head, 'before:', y[-1].size())
        #         # print(head, 'struture:', self.__getattr__(head))
        #         xnet_z[head] = self.__getattr__(head)(out2)
        #         # print(head, 'after:', xnet_z[head].size())
        #     xnet_z['hm'] = _sigmoid(xnet_z['hm'])
        #     # final_out.append(xnet_z)    # 6个输出

        # print(len(final_out), final_out[0].keys(), final_out[0]['hm'].size(), final_out[0]['wh'].size(), final_out[0]['reg'].size())
        return final_out
        # return [z]

class DLASeg(nn.Module):  #这个网络在FPN改进的基础上，加入了浅层特征与深层特征的特征融合
    def __init__(self, base_name, heads, pretrained, down_ratio, final_kernel,
                 last_level, head_conv, out_channel=0):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]  # down_ratio=4
        self.first_level = int(np.log2(down_ratio))  # self.first_level = 2
        self.last_level = last_level  # self.last_level = 5
        self.base = globals()[base_name](pretrained=pretrained)  # globals() 函数会以字典类型返回当前位置的全部全局变量。
        # self.base = dla34(pretrained=pretrained)
        channels = self.base.channels  # channels = [16, 32, 64, 128, 256, 512]
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]  # scales = [1, 2, 4, 8]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)
        #             DLAUp(  2,              [64, 128, 256, 512],         [1, 2, 4, 8])

        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level],
                            [2 ** i for i in range(self.last_level - self.first_level)])

        self.conv_cat1 = nn.Conv2d(128, 64, kernel_size=1, stride=1)
        self.conv_cat2 = nn.Conv2d(128, 64, kernel_size=1, stride=1)

        # self.get_fpn_out_layers = FPN_out()

        # self.up_delta_conv_0 = nn.Conv2d(64, 64, kernel_size=3, stride=(1, 2), padding=1, bias=False, dilation=1)
        # # self.up_delta_conv_1 = nn.Conv2d(64, 64, kernel_size=3, stride=(1, 2), padding=1, bias=False, dilation=1)
        # # self.up_delta_conv_2 = nn.Conv2d(64, 64, kernel_size=3, stride=(1, 2), padding=1, bias=False, dilation=1)
        #
        # self.btm_delta_conv_0 = nn.Conv2d(64, 64, kernel_size=3, stride=(2, 1), padding=1, bias=False, dilation=1)
        # # self.btm_delta_conv_1 = nn.Conv2d(64, 64, kernel_size=3, stride=(2, 1), padding=1, bias=False, dilation=1)
        # # self.btm_delta_conv_2 = nn.Conv2d(64, 64, kernel_size=3, stride=(2, 1), padding=1, bias=False, dilation=1)
        #
        # self.transconv_y0_up = nn.ConvTranspose2d(64, 64, (1, 4), stride=(1, 4), padding=0, output_padding=0, groups=64,
        #                                           bias=False)  #  !!注意groups参数！！
        # self.transconv_y0_btm = nn.ConvTranspose2d(64, 64, (4, 1), stride=(4, 1), padding=0, output_padding=0, groups=64,
        #                                           bias=False)
        # self.transconv_y1_up = nn.ConvTranspose2d(64, 64, (2, 8), stride=(2, 8), padding=0, output_padding=0, groups=64,
        #                                           bias=False)
        # self.transconv_y1_btm = nn.ConvTranspose2d(64, 64, (8, 2), stride=(8, 2), padding=0, output_padding=0, groups=64,
        #                                           bias=False)
        # self.transconv_y2_up = nn.ConvTranspose2d(64, 64, (4, 16), stride=(4, 16), padding=0, output_padding=0, groups=64,
        #                                           bias=False)
        # self.transconv_y2_btm = nn.ConvTranspose2d(64, 64, (16, 4), stride=(16, 4), padding=0, output_padding=0,
        #                                            groups=64, bias=False)
        # self.transconv_fpn1 = nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0, output_padding=0, groups=64,
        #                                          bias=False)
        # self.transconv_fpn2 = nn.ConvTranspose2d(64, 64, 4, stride=4, padding=0, output_padding=0, groups=64,
        #                                          bias=False)
        # self.transconv_fpn3 = nn.ConvTranspose2d(64, 64, 8, stride=8, padding=0, output_padding=0, groups=64,
        #                                          bias=False)

        # fill_up_weights(self.transconv_y0_up)
        # fill_up_weights(self.transconv_y0_btm)
        # fill_up_weights(self.transconv_y1_up)
        # fill_up_weights(self.transconv_y1_btm)
        # fill_up_weights(self.transconv_y2_up)
        # fill_up_weights(self.transconv_y2_btm)
        # fill_up_weights(self.transconv_fpn1)
        # fill_up_weights(self.transconv_fpn2)
        # fill_up_weights(self.transconv_fpn3)

        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(channels[self.first_level], head_conv,
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes,
                              kernel_size=final_kernel, stride=1,
                              padding=final_kernel // 2, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(channels[self.first_level], classes,
                               kernel_size=final_kernel, stride=1,
                               padding=final_kernel // 2, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        # print('input:', x.size())
        x = self.base(x)
        # print('after_base:', type(x), len(x), x[0].size(), x[1].size(), x[2].size(), x[3].size(), x[4].size(), x[5].size())
        feature_128_get_from_base = x[2]  # torch.Size([1, 64, 128, 128])
        x, feature_128_get_from_IDAUp = self.dla_up(x)
        # print('after_dla_up:', type(x), len(x), x[0].size(), x[1].size(), x[2].size(), x[3].size())
        # print('两个从浅层网络中提取出来的特征图：', feature_128_get_from_base.size(), feature_128_get_from_IDAUp.size())
        '''
        # fpn_in = x[:]
        # fpn_out = self.get_fpn_out_layers(fpn_in)

        # transconv_fpn0 = fpn_out[0]
        # transconv_fpn1 = self.transconv_fpn1(fpn_out[1])
        # transconv_fpn2 = self.transconv_fpn2(fpn_out[2])
        # transconv_fpn3 = self.transconv_fpn3(fpn_out[3])
        # transconv_fpn_out = [transconv_fpn0, transconv_fpn1, transconv_fpn2, transconv_fpn3]

        # y0_up_1 = self.up_delta_conv_0(fpn_out[0])
        # # print('y0_up_1:', y0_up_1.size())
        # y0_up_2 = self.up_delta_conv_0(y0_up_1)
        # # print('y0_up_2:', y0_up_2.size())
        # y0_up_out = self.transconv_y0_up(y0_up_2)
        # # print('y0_up_out:', y0_up_out.size())
        # y0_btm_1 = self.btm_delta_conv_0(fpn_out[0])
        # # print('y0_btm_1:', y0_btm_1.size())
        # y0_btm_2 = self.btm_delta_conv_0(y0_btm_1)
        # # print('y0_btm_2:', y0_btm_2.size())
        # y0_btm_out = self.transconv_y0_btm(y0_btm_2)
        # # print('y0_btm_out:', y0_btm_out.size())
        #
        # y1_up_1 = self.up_delta_conv_0(fpn_out[1])
        # # print('y1_up_1:', y1_up_1.size())
        # y1_up_2 = self.up_delta_conv_0(y1_up_1)
        # # print('y1_up_2:', y1_up_2.size())
        # y1_up_out = self.transconv_y1_up(y1_up_2)
        # # print('y1_up_out:', y1_up_out.size())
        # y1_btm_1 = self.btm_delta_conv_0(fpn_out[1])
        # # print('y1_btm_1:', y1_btm_1.size())
        # y1_btm_2 = self.btm_delta_conv_0(y1_btm_1)
        # # print('y1_btm_2:', y1_btm_2.size())
        # y1_btm_out = self.transconv_y1_btm(y1_btm_2)
        # # print('y1_btm_out:', y1_btm_out.size())
        #
        # y2_up_1 = self.up_delta_conv_0(fpn_out[2])
        # # print('y2_up_1:', y2_up_1.size())
        # y2_up_2 = self.up_delta_conv_0(y2_up_1)
        # # print('y2_up_2:', y2_up_2.size())
        # y2_up_out = self.transconv_y2_up(y2_up_2)
        # # print('y2_up_out:', y2_up_out.size())
        # y2_btm_1 = self.btm_delta_conv_0(fpn_out[2])
        # # print('y2_btm_1:', y2_btm_1.size())
        # y2_btm_2 = self.btm_delta_conv_0(y2_btm_1)
        # # print('y2_btm_2:', y2_btm_2.size())
        # y2_btm_out = self.transconv_y2_btm(y2_btm_2)
        # # print('y2_btm_out:', y2_btm_out.size())
        #
        # xnet_out = [y0_up_out, y0_btm_out, y1_up_out, y1_btm_out, y2_up_out, y2_btm_out]
        '''

        y = []
        for i in range(self.last_level - self.first_level):  # 5-2
            y.append(x[i].clone())
        # print('before_ida_up:', type(y), len(y), y[0].size(), y[1].size(), y[2].size())
        self.ida_up(y, 0, len(y))
        # print('after_ida_up:', type(y), len(y), y[0].size(), y[1].size(), y[2].size())
        # y[0]->y[1]->y[2]聚合度越来越高，却接近图表中的OUT标志
        y[1] = self.conv_cat1(torch.cat((y[1],feature_128_get_from_IDAUp), 1))
        y[2] = self.conv_cat2(torch.cat((y[2], feature_128_get_from_base), 1))
        # print('after_concatnate:', y[0].size(), y[1].size(), y[2].size())

        final_out = []

        # dla_z = {}
        # # print('self.heads:', self.heads)
        # for head in self.heads:
        #     # print(head, 'before:', y[-1].size())
        #     # print(head, 'struture:', self.__getattr__(head))
        #     dla_z[head] = self.__getattr__(head)(y[-1])
        #     # print(head, 'after:', z[head].size())
        # dla_z['hm'] = _sigmoid(dla_z['hm'])
        # final_out.append(dla_z)    # 1个输出
        # # return [dla_z]

        # 这是直接从原始的dla网络中得到的金字塔网络
        for out0 in y:   #　 y[0]->y[1]->y[2] ,虽然特征图尺寸一致，但是y[0]中特征图高分辨率信息多，应该用来回归小物体；y[2]用来回归大物体
            dla_fpn_z = {}
            # print('self.heads:', self.heads)
            for head in self.heads:
                # print(head, 'before:', y[-1].size())
                # print(head, 'struture:', self.__getattr__(head))
                dla_fpn_z[head] = self.__getattr__(head)(out0)
                # print(head, 'after:', z[head].size())
            dla_fpn_z['hm'] = _sigmoid(dla_fpn_z['hm'])
            final_out.append(dla_fpn_z)  # 3个输出

        # for out1 in transconv_fpn_out:
        #     fpn_z = {}
        #     # print('self.heads:', self.heads)
        #     for head in self.heads:
        #         # print(head, 'before:', y[-1].size())
        #         # print(head, 'struture:', self.__getattr__(head))
        #         fpn_z[head] = self.__getattr__(head)(out1)
        #         # print(head, 'after:', fpn_z[head].size())
        #     fpn_z['hm'] = _sigmoid(fpn_z['hm'])
        #     final_out.append(fpn_z)  # 4个输出

        # for out2 in xnet_out:
        #     xnet_z = {}
        #     # print('self.heads:', self.heads)
        #     for head in self.heads:
        #         # print(head, 'before:', y[-1].size())
        #         # print(head, 'struture:', self.__getattr__(head))
        #         xnet_z[head] = self.__getattr__(head)(out2)
        #         # print(head, 'after:', xnet_z[head].size())
        #     xnet_z['hm'] = _sigmoid(xnet_z['hm'])
        #     # final_out.append(xnet_z)    # 6个输出

        # print(len(final_out), final_out[0].keys(), final_out[0]['hm'].size(), final_out[0]['wh'].size(), final_out[0]['reg'].size())
        return final_out
        # return [z]

def get_pose_net(num_layers, heads, head_conv=256, down_ratio=4):
    model = DLASeg('dla{}'.format(num_layers), heads,
                   pretrained=True,  # 本来pretrained = True，为了做实验才设为False
                   down_ratio=down_ratio,
                   final_kernel=1,
                   last_level=5,
                   head_conv=head_conv)
    return model


    # strat = time.time()
    # model = get_pose_net(num_layers=34, heads={'hm': 1, 'wh': 2, 'reg': 2}, head_conv=256)
    # out = model(Variable(torch.randn(1,3,512,512)))
    # cost = time.time() - strat
    # print('cost:', cost)