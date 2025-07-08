import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch import Tensor
from torch.nn.common_types import _size_any_t, _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple, Union
from typing import Callable
from torch.nn.modules.batchnorm import _BatchNorm
from spikingjelly.activation_based import surrogate, base, neuron, functional, layer, encoding
from utils.general import make_divisible
from utils.tal.anchor_generator import make_anchors, dist2bbox
#from mamba_ssm import Mamba
from .common import Conv
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from .common import SP
from torchvision import transforms
from scipy.stats import norm, gaussian_kde


time_step = 4

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class SEncoder(nn.Module):
    def __init__(
            self,
            c1: int,
            c2: int,
            k: int,
            s: int = 1,
            p=None,
            g: int = 1,
            d: int = 1,
            act=True,
            lif: callable = None,
            step_mode: str = 's'
    ):
        super().__init__()
        self.default_act = neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True,step_mode='s')
        self.conv = layer.Conv2d(c1, c2, k, 1, autopad(k, p, d), groups=g, dilation=d, bias=False,step_mode='s')
        self.bn = seBatchNorm(c2,time_step)
        self.act = self.default_act if act is True else nn.Identity()#else act if isinstance(act, neuron.BaseNode) or isinstance(act,nn.Module) else nn.Identity()
        self.conv2 = layer.Conv2d(c2, c2, k, 2, autopad(k, p, d), groups=g, dilation=d, bias=False,step_mode='s')
        self.bn2 = seBatchNorm(c2,time_step)
        self.act2 = neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True,step_mode='s')

    def forward(self, x):
        x = [x for _ in range(time_step)]
        x = [self.conv(x[i]) for i in range(time_step)]
        x = self.bn(x)
        out = [self.act(x[i]) for i in range(time_step)]
        #out = Denoise(out)
        out = [self.conv2(out[i]) for i in range(time_step)]
        out = self.bn2(out)
        out = [self.act2(out[i]) for i in range(time_step)]

        return out


class SDConv(nn.Module):
    def __init__(
            self,
            c1: int,
            c2: int,
            k: int,
            s: int = 1,
            p=None,
            g: int = 1,
            d: int = 1,
            act=True,
            lif: callable = None,
            step_mode: str = 's'
    ):
        super().__init__()
        self.default_act = neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True,step_mode='s',v_threshold=float('inf'))
        self.conv = layer.Conv2d(c1, c2, k, step_mode='s')
        self.bn = seBatchNorm(c2,time_step)
        self.act = self.default_act

    def forward(self, x):
        x = [self.conv(x[i]) for i in range(time_step)]
        x = torch.stack(x, 0)
        out = x.mean(0)
        return out

class SConv(nn.Module):
    def __init__(
            self,
            c1: int,
            c2: int,
            k: int,
            s: int = 1,
            p=None,
            g: int = 1,
            d: int = 1,
            act=True,
            lif: callable = None,
            step_mode: str = 's',
            n=1.0
    ):
        super().__init__()
        self.default_act = neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True,step_mode='s')
        self.conv = layer.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False,step_mode='s')
        self.bn = seBatchNorm(c2,time_step,n)
        self.act = self.default_act if act is True else nn.Identity()#else act if isinstance(act, neuron.BaseNode) or isinstance(act,nn.Module) else nn.Identity()

    def forward(self, x):
        x = [self.conv(x[i]) for i in range(time_step)]
        out = self.bn(x)
        out = [self.act(out[i]) for i in range(time_step)]
        return out

class SSP(nn.Module):
    def __init__(self, k=3, s=1):
        super(SSP, self).__init__()
        self.m = layer.MaxPool2d(kernel_size=k, stride=s, padding=k // 2,step_mode='s')

    def forward(self, x):
        out = [self.m(x[i]) for i in range(time_step)]
        return out

class SDFL(nn.Module):
    # DFL module
    def __init__(self, c1=17):
        super().__init__()
        self.conv = layer.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        self.conv.weight.data[:] = nn.Parameter(torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1))  # / 120.0
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)

class SConcat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        out = [torch.cat([n[i] for n in x], self.d) for i in range(time_step)]
        return out

class SUpsample(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, scale_factor, mode='nearest'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode=mode)

    def forward(self, x):
        out = [self.up(x[i]) for i in range(time_step)]
        return out
    
class TransitionBlock(nn.Module):
    # spp-elan
    def __init__(self, c1, c2, c3):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = c3
        self.conv = layer.Conv2d(c1, c3, 1, 1)
        self.bn = seBatchNorm(c3,time_step)
        self.act = neuron.IFNode(v_threshold=1.0, surrogate_function=surrogate.ATan(), detach_reset=True,step_mode='s')
        self.cv2 = SSP(5)
        self.cv3 = SSP(5)
        self.cv5 = SConv(3 * c3, c2, 1, 1)

    def forward(self, x):
        x = [self.conv(x[i]) for i in range(time_step)]
        x = self.bn(x)
        y = [x]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        out = [torch.cat([n[i] for n in y], 1) for i in range(time_step)]
        out = [self.act(out[i]) for i in range(time_step)]
        out = self.cv5(out)
        return out

class BasicBlock1(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, c5=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.cvres = SConv(c1, c2//2, 1, 2,act=False,n=2)
        self.cv0 = SConv(c1, c2, 3, 2,act=False)
        self.cv2 = SConv(c4, c4, 3, 1,act=False,n=2)
        self.act1 = neuron.IFNode(v_threshold=1.0, surrogate_function=surrogate.ATan(), detach_reset=True,step_mode='s')
        self.act2 = neuron.IFNode(v_threshold=1.0, surrogate_function=surrogate.ATan(), detach_reset=True,step_mode='s')
        self.act3 = neuron.IFNode(v_threshold=1.0, surrogate_function=surrogate.ATan(), detach_reset=True,step_mode='s')
        self.c = c4

    def forward(self, x):
        x1 = []
        x2 = []

        xres = self.cvres(x)
        x = self.cv0(x)

        for i in range(time_step):
            y1, y2 = x[i].chunk(2, 1)
            x1.append(y1)
            x2.append(y2)

        x3 = [self.act2(x2[i]) for i in range(time_step)]
        x4 = self.cv2(x3)
        for i in range(time_step):
            x4[i] = (x4[i]+ xres[i])
        y = [x1, x4]

        out = [torch.cat([n[i] for n in y], 1) for i in range(time_step)]
        out = [self.act3(out[i]) for i in range(time_step)]       
        return out
    
class BasicBlock2(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, c5=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.cv0 = SConv(c1, c2, 1, 1,act=False)
        self.cv2 = SConv(c4, c4, 3, 1,act=False)
        self.act1 = neuron.IFNode(v_threshold=1.0, surrogate_function=surrogate.ATan(), detach_reset=True,step_mode='s')
        self.act2 = neuron.IFNode(v_threshold=1.0, surrogate_function=surrogate.ATan(), detach_reset=True,step_mode='s')
        self.act3 = neuron.IFNode(v_threshold=1.0, surrogate_function=surrogate.ATan(), detach_reset=True,step_mode='s')

    def forward(self, x):

        x1 = []
        x2 = []

        x = self.cv0(x)

        for i in range(time_step):
            y1, y2 = x[i].chunk(2, 1)
            x1.append(y1)
            x2.append(y2)

        x3 = [self.act2(x2[i]) for i in range(time_step)]
        x4 = self.cv2(x3)
        y = [x1, x4]

        out = [torch.cat([n[i] for n in y], 1) for i in range(time_step)]
        out = [self.act3(out[i]) for i in range(time_step)]
        return out

class SDDetect(nn.Module):
    # YOLO Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = make_divisible(max((ch[0] // 4, self.reg_max * 4, 16)), 4), max(
            (ch[0], min((self.nc * 2, 128))))  # channels
        self.cv2 = nn.ModuleList(nn.Sequential(SConv(x, c2, 3), SConv(c2, c2, 3, g=4), SDConv(c2, 4 * self.reg_max, 1, g=4)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(SConv(x, c3, 3), SConv(c3, c3, 3), SDConv(c3, self.nc, 1)) for x in ch)
        self.dfl = SDFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0][0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape


        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        m = self  # self.model[-1]  # Detect() module
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].conv.bias.data[:] = 1.0  # box
            b[-1].conv.bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)
    
def Denoise(x):
    x = [1-x[i] for i in range(time_step)]
    x = [FindPoints(x[i],3) for i in range(time_step)]
    x = [FindPoints(x[i],2) for i in range(time_step)]
    x = [1-x[i] for i in range(time_step)]
    x = [FindPoints(x[i],1) for i in range(time_step)]
    return x

def Denoise2(x):
    x = [1-x[i] for i in range(time_step)]
    x = [FindPoints(x[i],3) for i in range(time_step)]
    x = [FindPoints(x[i],2) for i in range(time_step)]
    x = [1-x[i] for i in range(time_step)]
    #x = [FindPoints(x[i],1) for i in range(time_step)]
    return x

def Denoise3(x):
    x = [FindPoints(x[i],0) for i in range(time_step)]
    x = [FindPoints(x[i],1) for i in range(time_step)]
    return x

def FindPoints(tensor,type):
    if type == 1 or type == 0:
        if tensor.dtype == torch.float32:
            kernel = torch.tensor([[-1, -1, -1],
                                   [-1,  8, -1],
                                   [-1, -1, -1]], dtype=torch.float32)
        if tensor.dtype == torch.float16:
            kernel = torch.tensor([[-1, -1, -1],
                                   [-1,  8, -1],
                                   [-1, -1, -1]], dtype=torch.float16)
    if type == 2:
        if tensor.dtype == torch.float32:
            kernel = torch.tensor([[2, 0, 2],
                                   [ 0, -8,  0],
                                   [2, 0, 2]], dtype=torch.float32)
        if tensor.dtype == torch.float16:
            kernel = torch.tensor([[2, 0, 2],
                                   [ 0, -8,  0],
                                   [2, 0, 2]], dtype=torch.float16)
    if type == 3:
        if tensor.dtype == torch.float32:
            kernel = torch.tensor([[ 0, 2,  0],
                                   [2,  -8, 2],
                                   [ 0, 2,  0]], dtype=torch.float32)
        if tensor.dtype == torch.float16:
            kernel = torch.tensor([[0, 2,  0],
                                   [2, -8, 2],
                                   [0, 2,  0]], dtype=torch.float16)
    if type == 4:
        if tensor.dtype == torch.float32:
            kernel = torch.tensor([[-1,-1,-1,-1,-1],
                                   [-1, 1, 1, 1,-1],
                                   [-1, 1, 1, 1,-1],
                                   [-1, 1, 1, 1,-1],
                                   [-1,-1,-1,-1,-1]], dtype=torch.float32)
        if tensor.dtype == torch.float16:
            kernel = torch.tensor([[-1,-1,-1,-1,-1],
                                   [-1, 1, 1, 1,-1],
                                   [-1, 1, 1, 1,-1],
                                   [-1, 1, 1, 1,-1],
                                   [-1,-1,-1,-1,-1]], dtype=torch.float16)
    
    # kernel 需要扩展到与输入 tensor 的 channels 维度匹配
    # kernel 是 1x1x3x3， 我们需要变为 channels x 1 x 3 x 3 以应用到每个通道
    channels = tensor.size(1)  # 获取输入的 channels 数
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # 变成 1x1x3x3
    if type==4:
        kernel = kernel.expand(channels, 1, 5, 5)  # 变成 channels x 1 x 3 x 3
    else:
        kernel = kernel.expand(channels, 1, 3, 3)  # 变成 channels x 1 x 3 x 3
    kernel = kernel.to(tensor.device)
    # 使用卷积操作
    if type==4:
        conv_result = F.conv2d(tensor, kernel, padding=2, groups=channels).detach()
    else:
        conv_result = F.conv2d(tensor, kernel, padding=1, groups=channels).detach()
    
    # 找到卷积结果为 4 的位置
    if type==1:
        positive_points = (conv_result >=8).int().detach()
        negative_points = 0
    if type==0:
        positive_points = 0
        negative_points = (conv_result <=-6).int().detach()
    if type==4:
        positive_points = 0
        negative_points = (conv_result <=-12).int().detach()
    if type==2:
        positive_points = (conv_result <=-8).int().detach()
        negative_points = 0
    if type==3:
        positive_points = (conv_result <=-8).int().detach()
        negative_points = 0
    # 将孤立点的位置设置为 0
    result = tensor + negative_points - positive_points
    return result

class seBatchNorm(nn.Module):
    def __init__(self,c,t,n=1.0):
        super(seBatchNorm, self).__init__()
        self.bn = SeBatchNorm2d(t*c, n)
    def forward(self, x):
        if isinstance(x,list):
            x = torch.stack(x,0)
            T,B,C,H,W = x.shape
            x = x.permute(1, 0, 2, 3, 4).reshape(B,T*C,H,W)
            x = self.bn(x)
            x = x.reshape(B,T,C,H,W).permute(1, 0, 2, 3, 4)
            out = torch.unbind(x,0)
        else:
            T,B,C,H,W = x.shape
            x = x.permute(1, 0, 2, 3, 4).reshape(B,T*C,H,W)
            x = self.bn(x)
            out = x.reshape(B,T,C,H,W).permute(1, 0, 2, 3, 4)
        return out
    
class SeBatchNorm2d(torch.nn.modules.batchnorm._BatchNorm):
    def __init__(self, num_features, n=1.0, eps=1e-5, momentum=0.1, 
                 affine=True, track_running_stats=True):
        super(SeBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.var_scale = 1.0/n

    def forward(self, input):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.training:
            # 计算当前batch的均值和方差
            mean = input.mean(dim=(0, 2, 3))
            var = input.var(dim=(0, 2, 3), unbiased=False)
            # 调整方差
            var_adjusted = var / self.var_scale

            with torch.no_grad():
                # 更新running_mean和running_var
                self.running_mean = (1 - exponential_average_factor) * self.running_mean + exponential_average_factor * mean
                self.running_var = (1 - exponential_average_factor) * self.running_var + exponential_average_factor * var
        else:
            # 使用存储的running_mean和调整后的running_var
            mean = self.running_mean
            var_adjusted = self.running_var / self.var_scale

        # 应用归一化
        input = (input - mean[None, :, None, None]) / torch.sqrt(var_adjusted[None, :, None, None] + self.eps)

        # 应用仿射变换
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input
    
    