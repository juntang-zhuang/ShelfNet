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
from .LadderNetv66_small import Decoder,BasicBlock,LadderBlock
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter
# infer number of classes
from ..datasets import datasets

__all__ = ['LadderNet', 'get_laddernet', 'get_laddernet_resnet50_pcontext',
           'get_laddernet_resnet101_pcontext', 'get_laddernet_resnet50_ade']

class LadderNet(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=True, lateral=False,
                 norm_layer=BatchNorm2d,dilated=False, **kwargs):
        super(LadderNet, self).__init__(nclass, backbone, aux, se_loss,
                                     norm_layer=norm_layer, dilated=dilated,**kwargs)

        self.head = LadderHead(base_inchannels=256,base_outchannels=64,out_channels=nclass,
                               norm_layer=norm_layer,se_loss = se_loss,nclass=nclass,up_kwargs=self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer=norm_layer)

    def forward(self, x):
        imsize = x.size()[2:]
        features = self.base_forward(x)

        x = list(self.head(features))

        x[0] = F.upsample(x[0], imsize, **self._up_kwargs)
        if self.aux:
            auxout = self.auxlayer(features[2])
            auxout = F.upsample(auxout, imsize, **self._up_kwargs)
            x.append(auxout)
        return tuple(x)


class LadderHead(nn.Module):
    def __init__(self, base_inchannels, base_outchannels,out_channels, norm_layer, se_loss,nclass,up_kwargs):
        super(LadderHead, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=base_inchannels,out_channels=base_outchannels,kernel_size=1,bias=False)
        self.conv2 = nn.Conv2d(in_channels=base_inchannels*2,out_channels=base_outchannels*2,kernel_size=1,bias=False)
        self.conv3 = nn.Conv2d(in_channels=base_inchannels*2**2,out_channels=base_outchannels*2**2,kernel_size=1,bias=False)
        self.conv4 = nn.Conv2d(in_channels=base_inchannels*2**3,out_channels=base_outchannels*2**3,kernel_size=1,bias=False)

        self.bn1 = norm_layer(base_outchannels)
        self.bn2 = norm_layer(base_outchannels*2)
        self.bn3 = norm_layer(base_outchannels*2**2)
        self.bn4 = norm_layer(base_outchannels*2**3)

        self.decoder = Decoder(planes=base_outchannels,layers=4)
        self.ladder = LadderBlock(planes=base_outchannels,layers=4)
        self.final = nn.Conv2d(base_outchannels, out_channels, 1)

        self.se_loss = se_loss

        if self.se_loss:
            self.selayer = nn.Linear(base_outchannels*2**3, nclass)

    def forward(self, x):
        x1,x2,x3,x4 = x

        out1 = self.conv1(x1)
        out1 = self.bn1(out1)
        out1 = F.relu(out1)

        out2 = self.conv2(x2)
        out2 = self.bn2(out2)
        out2 = F.relu(out2)

        out3 = self.conv3(x3)
        out3 = self.bn3(out3)
        out3 = F.relu(out3)

        out4 = self.conv4(x4)
        out4 = self.bn4(out4)
        out4 = F.relu(out4)


        out = self.decoder([out1, out2, out3, out4])
        out = self.ladder(out)

        pred = [self.final(out[-1])]

        if self.se_loss:
            enc = F.max_pool2d(out[0],kernel_size=out[0].size()[2:])
            enc = torch.squeeze(enc,-1)
            enc = torch.squeeze(enc,-1)
            se = self.selayer(enc)
            pred.append(se)

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
