###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from encoding.nn import SegmentationLosses, BatchNorm2d

import encoding
from .base import BaseNet
from .fcn import FCNHead
from ..nn import PyramidPooling
from .LadderNetv66 import Decoder,BasicBlock
__all__ = ['LadderNet', 'get_laddernet', 'get_laddernet_resnet50_pcontext',
           'get_laddernet_resnet101_pcontext', 'get_laddernet_resnet50_ade']

class LadderBlock(nn.Module):

    def __init__(self,planes,layers,kernel=3,block=BasicBlock):
        super().__init__()
        self.planes = planes
        self.layers = layers
        self.kernel = kernel

        self.padding = int((kernel-1)/2)
        self.inconv = block(planes,planes)

        # create module list for down branch
        self.down_module=block(planes*(2**0),planes*(2**0))

        # use strided conv instead of pooling
        self.down_conv = nn.Conv2d(planes*2**0,planes*2**(0+1),stride=2,kernel_size=kernel,padding=self.padding)

        # create module for bottom block
        self.bottom = block(planes*(2**layers),planes*(2**layers))

        # create module list for up branch
        self.up_conv = nn.ConvTranspose2d(planes*2**(layers-0), planes*2**max(0,layers-0-1), kernel_size=3,
                                                        stride=2,padding=1,output_padding=1,bias=True)

        self.up_dense = block(planes*2**max(0,layers-0-1),planes*2**max(0,layers-0-1))


    def forward(self, x):
        out = self.inconv(x[-1])

        down_out = []
        # down branch
        out = out + x[-1]
        out = self.down_module(out)
        down_out.append(out)

        out = self.down_conv(out)
        out = F.relu(out)

        # bottom branch
        out = self.bottom(out)
        bottom = out

        # up branch
        up_out = []
        up_out.append(bottom)


        out = self.up_conv(out) + down_out[self.layers-1]
        #out = F.relu(out)
        out = self.up_dense(out)
        up_out.append(out)

        return up_out


class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = BatchNorm2d(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

class LadderNet(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=True, lateral=False,
                 norm_layer=BatchNorm2d, **kwargs):
        super(LadderNet, self).__init__(nclass, backbone, aux, se_loss,
                                     norm_layer=norm_layer, **kwargs)
        self.head = LadderHead(in_channels=2048, inter_channels=512,out_channels=nclass,
                               norm_layer=norm_layer,up_kwargs=self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer=norm_layer)

    def forward(self, x):
        imsize = x.size()[2:]
        features = self.base_forward(x)

        x = [self.head(features)]

        x[0] = F.upsample(x[0], imsize, **self._up_kwargs)
        if self.aux:
            auxout = self.auxlayer(features[2])
            auxout = F.upsample(auxout, imsize, **self._up_kwargs)
            x.append(auxout)
        return tuple(x)

from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter

class PyramidPooling_v2(Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels, norm_layer, up_kwargs,out_channels=None):
        super(PyramidPooling_v2, self).__init__()
        self.pool1 = AdaptiveAvgPool2d(1)
        self.pool2 = AdaptiveAvgPool2d(2)
        self.pool3 = AdaptiveAvgPool2d(3)
        self.pool4 = AdaptiveAvgPool2d(6)

        if out_channels is None:
            out_channels = int(in_channels/4)

        self.conv1 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels))
        self.conv2 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels))
        self.conv3 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels))
        self.conv4 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels))
        # bilinear upsample options
        self._up_kwargs = up_kwargs
        self.relu = nn.ReLU(True)
    def forward(self, input):
        x = input
        _, _, h, w = x.size()
        feat1 = F.upsample(self.conv1(self.pool1(x)), (h, w), **self._up_kwargs)
        feat2 = F.upsample(self.conv2(self.pool2(x)), (h, w), **self._up_kwargs)
        feat3 = F.upsample(self.conv3(self.pool3(x)), (h, w), **self._up_kwargs)
        feat4 = F.upsample(self.conv4(self.pool4(x)), (h, w), **self._up_kwargs)
        feat = feat1+feat2+feat3+feat4
        feat = self.relu(feat)
        return torch.cat((x,feat), 1)

class LadderHead(nn.Module):
    def __init__(self, in_channels, inter_channels,out_channels, norm_layer, up_kwargs):
        super(LadderHead, self).__init__()

        # 1015 structure
        #self.conv5 = nn.Sequential(PyramidPooling(in_channels, norm_layer,up_kwargs=up_kwargs,out_channels=in_channels//4),
        #                           nn.Conv2d(in_channels+in_channels, inter_channels, 3, padding=2,dilation=2, bias=False),
        #                           norm_layer(inter_channels),
        #                           nn.ReLU(True))

        # 1014 and 1011 structure
        self.conv5 = nn.Sequential(
            PyramidPooling_v2(in_channels, norm_layer, up_kwargs=up_kwargs, out_channels=in_channels // 8),
            nn.Conv2d(in_channels + in_channels//8, inter_channels, 3, padding=2, dilation=2, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True))
        #self.conv5 = nn.Sequential(
        #    nn.Conv2d(in_channels, inter_channels, 3, padding=2, bias=False,dilation=2),
        #    norm_layer(inter_channels),
        #    nn.ReLU(True))
        self.decoder = Decoder(planes=inter_channels//2,layers=1)
        self.ladder = LadderBlock(planes=inter_channels//2,layers=1)
        self.final = nn.Conv2d(inter_channels//2, out_channels, 1)
    def forward(self, x):
        x1,x2,x3,x4 = x
        conv5 = self.conv5(x4)
        #conv5 = self.conv5([x2,x4])

        out = self.decoder([x1,conv5])
        out = self.ladder(out)

        pred = self.final(out[-1])

        return pred


def get_laddernet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
               root='~/.encoding/models', **kwargs):
    r"""EncNet model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    dataset : str, default pascal_voc
        The dataset that model pretrained on. (pascal_voc, ade20k)
    backbone : str, default resnet50
        The backbone network. (resnet50, 101, 152)
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_encnet(dataset='pascal_voc', backbone='resnet50', pretrained=False)
    >>> print(model)
    """
    acronyms = {
        'pascal_voc': 'voc',
        'ade20k': 'ade',
        'pcontext': 'pcontext',
    }
    kwargs['lateral'] = True if dataset.lower() == 'pcontext' else False
    # infer number of classes
    from ..datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = LadderNet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file(backbone, root=root)))
    return model

def get_laddernet_resnet50_pcontext(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_encnet_resnet50_pcontext(pretrained=True)
    >>> print(model)
    """
    return get_laddernet('pcontext', 'resnet50', pretrained, root=root, aux=True, **kwargs)

def get_laddernet_resnet101_pcontext(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_encnet_resnet101_pcontext(pretrained=True)
    >>> print(model)
    """
    return get_laddernet('pcontext', 'resnet101', pretrained, root=root, aux=True, **kwargs)

def get_laddernet_resnet50_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_encnet_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_laddernet('ade20k', 'resnet50', pretrained, root=root, aux=True, **kwargs)
