import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import BatchNorm2d
from modules.bn import InPlaceABNSync
drop = 0.1

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = InPlaceABNSync(out_chan)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size= 1, bias=False)
        self.bn_atten = InPlaceABNSync(out_chan, activation='none')
        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, rate=1,downsample=None):
        super(BasicBlock, self).__init__()
        if inplanes!= planes:
            self.conv0 = conv3x3(inplanes,planes,rate)

        self.inplanes = inplanes
        self.planes = planes

        self.conv1 = conv3x3(planes, planes, stride)
        self.bn1 = InPlaceABNSync(planes)
        #self.relu = nn.ReLU(inplace=True)
        #self.conv2 = conv3x3(planes, planes)

        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.drop = nn.Dropout2d(p=drop)

    def forward(self, x):
        if self.inplanes != self.planes:
            x = self.conv0(x)
            x = F.relu(x)

        out = self.conv1(x)
        out = self.bn1(out)
        #out = self.relu(out)

        out = self.drop(out)

        out1 = self.conv1(out)
        out1 = self.bn2(out1)
        #out1 = self.relu(out1)

        out2 = out1 + x

        return F.relu(out2)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes//self.expansion, kernel_size=1, bias=False)
        self.bn1 = InPlaceABNSync(planes//self.expansion)
        self.conv2 = nn.Conv2d(planes//self.expansion, planes//self.expansion, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = InPlaceABNSync(planes//self.expansion)
        self.conv3 = nn.Conv2d(planes//self.expansion, planes, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.drop = nn.Dropout2d(p=drop)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        #out = self.relu(out)

        out = self.drop(out)

        out = self.conv2(out)
        out = self.bn2(out)
        #out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Decoder(nn.Module):
    def __init__(self,planes,layers,kernel=3,block=BasicBlock):
        super().__init__()
        self.planes = planes
        self.layers = layers
        self.kernel = kernel
        self.padding = int((kernel - 1) / 2)
        self.inconv = block(planes, planes)
        # create module for bottom block
        self.bottom = block(planes * (2 ** (layers-1)), planes * (2 ** (layers-1)))

        # create module list for up branch
        self.up_conv_list = nn.ModuleList()
        self.up_dense_list = nn.ModuleList()
        for i in range(0, layers-1):
            self.up_conv_list.append(
                AttentionRefinementModule(planes * 2 ** (layers-1 - i), planes * 2 ** max(0, layers - i - 2))
                )
            self.up_dense_list.append(ConvBNReLU(in_chan=planes * 2 ** max(0, layers - i - 2), out_chan=planes * 2 ** max(0, layers - i - 2),
                                                 ks=3,stride=1))

    def forward(self, x):
        # bottom branch
        out = self.bottom(x[-1])
        bottom = out

        # up branch
        up_out = []
        up_out.append(bottom)

        for j in range(0, self.layers-1):
            out = self.up_conv_list[j](out)
            out = F.interpolate(out, (out.size(2)*2, out.size(3)*2), mode='nearest') + x[self.layers - j - 2]
            # out = F.relu(out)
            out = self.up_dense_list[j](out)
            up_out.append(out)

        return up_out

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, InPlaceABNSync) or isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

class LadderBlock(nn.Module):

    def __init__(self,planes,layers,kernel=3,block=BasicBlock):
        super().__init__()
        self.planes = planes
        self.layers = layers
        self.kernel = kernel

        self.padding = int((kernel-1)/2)
        self.inconv = block(planes,planes)

        # create module list for down branch
        self.down_module_list = nn.ModuleList()
        for i in range(0,layers-1):
            self.down_module_list.append(block(planes*(2**i),planes*(2**i)))

        # use strided conv instead of pooling
        self.down_conv_list = nn.ModuleList()
        for i in range(0,layers-1):
            self.down_conv_list.append(nn.Conv2d(planes*2**i,planes*2**(i+1),stride=2,kernel_size=kernel,padding=self.padding))

        # create module for bottom block
        self.bottom = block(planes*(2**(layers-1)),planes*(2**(layers-1)))

        # create module list for up branch
        self.up_conv_list = nn.ModuleList()
        self.up_dense_list = nn.ModuleList()
        for i in range(0, layers-1):
            self.up_conv_list.append(
                AttentionRefinementModule(planes * 2 ** (layers - 1 - i), planes * 2 ** max(0, layers - i - 2))
            )
            self.up_dense_list.append(
                ConvBNReLU(in_chan=planes * 2 ** max(0, layers - i - 2), out_chan=planes * 2 ** max(0, layers - i - 2),
                           ks=3, stride=1))


    def forward(self, x):
        out = self.inconv(x[-1])

        down_out = []
        # down branch
        for i in range(0,self.layers-1):
            out = out + x[-i-1]
            out = self.down_module_list[i](out)
            down_out.append(out)

            out = self.down_conv_list[i](out)
            out = F.relu(out)

        # bottom branch
        out = self.bottom(out)
        bottom = out

        # up branch
        up_out = []
        up_out.append(bottom)

        for j in range(0, self.layers - 1):
            out = self.up_conv_list[j](out)
            out = F.interpolate(out, (out.size(2) * 2, out.size(3) * 2), mode='nearest') + down_out[self.layers - j - 2]
            # out = F.relu(out)
            out = self.up_dense_list[j](out)
            up_out.append(out)

        return up_out

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, InPlaceABNSync) or isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

class Encoder(nn.Module):

    def __init__(self,planes,layers,kernel=3,block=BasicBlock):
        super().__init__()
        self.planes = planes
        self.layers = layers
        self.kernel = kernel

        self.padding = int((kernel-1)/2)
        self.inconv = block(planes,planes)

        # create module list for down branch
        self.down_module_list = nn.ModuleList()
        for i in range(0,layers-1):
            self.down_module_list.append(block(planes*(2**i),planes*(2**i)))

        # use strided conv instead of pooling
        self.down_conv_list = nn.ModuleList()
        for i in range(0,layers-2):
            self.down_conv_list.append(nn.Conv2d(planes*2**i,planes*2**(i+1),stride=2,kernel_size=kernel,padding=self.padding))


    def forward(self, x):
        out = self.inconv(x[-1])

        down_out = []
        # down branch
        for i in range(0,self.layers-1):
            out = out + x[-i-1]
            out = self.down_module_list[i](out)
            down_out.append(out)

            if i<self.layers-2: #  not do down-sample for last layer
                out = self.down_conv_list[i](out)
                out = F.relu(out)

        return down_out

class Final_LadderBlock(nn.Module):

    def __init__(self,planes,layers,kernel=3,block=BasicBlock,inplanes = 3):
        super().__init__()
        self.block = LadderBlock(planes,layers,kernel=kernel,block=block)

    def forward(self, x):
        out = self.block(x)
        return out[-1]
