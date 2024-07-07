# Copyright (c) Jiewen Zhu. and UESTC.
"""
SANet
"""
import warnings
from functools import reduce

import torch
from torch import nn



warnings.filterwarnings('ignore', category=UserWarning)
import torch
import torch.nn as nn
import torch
import torch.nn as nn


# from BaseConv import BaseConv
import time
import torch
import torch.nn as nn


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
    
def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad         = (ksize - 1) // 2
        self.conv   = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn     = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act    = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class mlp(nn.Module):

    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc1 = BaseConv(in_channels, hidden_channels, 1, 1)
        self.fc2 = BaseConv(hidden_channels, out_channels, 1, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SAG_atten(nn.Module):
    def __init__(self, dim, bias=False, proj_drop=0.):
        super().__init__()

        self.fc1 = BaseConv(dim, dim, 1, 1, bias=bias)  
        
        self.fc2 = BaseConv(dim, dim, 3, 1, bias=bias)  
        
        self.mix = BaseConv(2*dim, dim, 3, 1, bias=bias) 
        self.reweight = mlp(dim, dim, dim)
        

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        x_1 = x.clone()
        
        # 负数为零
        x_1[x_1<0] = 0
        for i in range(B):
            for j in range(C):
                mean = x_1[i,j,:,:].mean() 
                x_1[i,j,:,:] = x[i,j,:,:]/(mean + 1e-4)   

        x_1 = self.fc1(x_1)
        
        x_2 = self.fc2(x)
        
        x_1 = self.mix(torch.cat([x_1, x_2], dim=1))
        x_1 = self.reweight(x_1)
        
        x = residual * x_1
        return x
    
class SAG_atten_up(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.fc1 = BaseConv(dim, dim, 1, 1, bias=bias)  
        self.fc2 = BaseConv(dim, dim, 1, 1, bias=bias)  
        self.fc3 = BaseConv(dim, dim, 1, 1, bias=bias)
        self.fc = BaseConv(dim, dim, 3, 1, bias=bias)
        
        self.mix = BaseConv(4*dim, dim, 3, 1, bias=bias) 
        self.reweight = mlp(dim, dim, dim)
        
    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        x_1 = torch.max(x,torch.tensor([0.]))
        x = torch.cat([x] + [self.SagMean(H,W,x_1,size) for size in [1,2,4]], dim=1)
        x = self.mix(x)
        x = self.reweight(x)
        return x*residual
    
    def SagMean(self, H, W, x, size=1):
        h, w = H//size, W//size
        x_ = x.clone()
        for i in range(size):
            for j in range(size):
                mean = x_[:, :, i*h:(i+1)*h, j*w:(j+1)*w].mean()
                x_[:, :, i*h:(i+1)*h, j*w:(j+1)*w] = x_[:, :, i*h:(i+1)*h, j*w:(j+1)*w]/(mean+1e-4) 
        if size == 1:
            x = self.fc1(x_)
        elif size == 2:
            x = self.fc2(x_)
        else:
            x = self.fc3(x_)                
        return x

if __name__ == "__main__":
    
    x = torch.rand([2,32,640,640])
    t = time.time()
    x = SAG_atten_up(32)(x)
    t = time.time() - t
    print(x.shape, t)

class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
    
def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad         = (ksize - 1) // 2
        self.conv   = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn     = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act    = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        patch_top_left  = x[...,  ::2,  ::2]
        patch_bot_left  = x[..., 1::2,  ::2]
        patch_top_right = x[...,  ::2, 1::2]
        patch_bot_right = x[..., 1::2, 1::2]
        
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1,)
        return self.conv(x)
    
    
class SPPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):

        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1      = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m          = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
       
        conv2_channels  = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2      = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, act="silu",):

        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = BaseConv

        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y
    
class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, act="silu",):

        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  

        self.conv1  = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)

        self.conv2  = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)

        self.conv3  = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)

        module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, act=act) for _ in range(n)]
        self.m      = nn.Sequential(*module_list)

    def forward(self, x):

        x_1 = self.conv1(x)
        x_2 = self.conv2(x)

        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)

        return self.conv3(x)
    
    
class SKConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,M=2, r=16, L=32):
        super(SKConv, self).__init__()
        d = max(in_channels//r, L) 
        self.M = M
        self.out_channels = out_channels
        self.conv = nn.ModuleList() 
        for i in range(M):
            self.conv.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, padding=1+i, dilation=1+i, groups=32, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1) 
        self.fc1 = nn.Sequential(
            nn.Conv2d(out_channels, d, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
        )   
        self.fc2 = nn.Conv2d(d, out_channels*M, 1, 1, bias=False)  
        self.softmax = nn.Softmax(dim=1) 

    def forward(self, input):
        batch_size = input.size(0)
        output = []

        for i,conv in enumerate(self.conv):
            output.append(conv(input))   

        U = reduce(lambda x,y:x+y, output) 
        s = self.global_pool(U)  
        z = self.fc1(s) 
        a_b = self.fc2(z) 
        a_b = a_b.reshape(batch_size, self.M, self.out_channels, -1) 
        a_b = self.softmax(a_b)
        a_b = list(a_b.chunk(self.M, dim=1))  
        a_b = list(map(lambda x:x.reshape(batch_size,self.out_channels,1,1), a_b))

        V = list(map(lambda x,y:x*y, output, a_b))
        V = reduce(lambda x,y:x+y, V)
        return V
    
class SANet(nn.Module):
    def __init__(self, dep_mul, wid_mul, act="silu"):

        super().__init__()
        Conv = BaseConv
        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)
        # [3, 640, 640] -> [64, 320, 320] -> [128, 160, 160]
        self.stem = nn.Sequential(
            Focus(3, base_channels, ksize=3, act=act),
            Conv(base_channels, 2*base_channels, 3, 2, act=act)
        )
        
        # [128, 160, 160] -> [128, 160, 160]
        # [128, 160, 160] -> [256, 80, 80]
        self.csp1 = nn.Sequential(
            Conv(2*base_channels, 2*base_channels, 3, 1, act=act),
            CSPLayer(2 * base_channels, 2 * base_channels, n=base_depth, act=act)
        )  
        self.csp2 = nn.Sequential(
            Conv(2*base_channels, 4*base_channels, 3, 2, act=act),
            CSPLayer(4 * base_channels, 4 * base_channels, n=3*base_depth, act=act)
        )
        
        # [128, 160, 160] -> [256, 80, 80]
        # [256, 80, 80] -> [256, 80, 80]
        self.SAG_1 = nn.Sequential(
            SAG_atten(2*base_channels),
            Conv(2*base_channels, 4*base_channels, 3, 2, act=act),
            SAG_atten(4*base_channels),
        )
        self.SAG_2 = nn.Sequential(
            Conv(4*base_channels, 4*base_channels, 3, 1, act=act),
            SAG_atten(4*base_channels),
        )
        
        # [512, 80, 80] -> [512, 40, 40]
        # [512, 40, 40] -> [1024, 20, 20]
        self.sk1 = nn.Sequential(
            Conv(8*base_channels, 8*base_channels, 3, 2, act=act),
            SKConv(8*base_channels, 8*base_channels),
            CSPLayer(8 * base_channels, 8 * base_channels, n=3*base_depth, act=act),
            SKConv(8*base_channels, 8*base_channels)
        )
        self.sk2 = nn.Sequential(
            Conv(8*base_channels, 16*base_channels, 3, 2, act=act),
            SKConv(16*base_channels, 16*base_channels),
            CSPLayer(16 * base_channels, 16 * base_channels, n=3*base_depth, act=act),
            SKConv(16*base_channels, 16*base_channels)
        )
        
        # SPP [1024, 20, 20] -> [1024, 20, 20]
        self.spp = SPPBottleneck(16*base_channels, 16*base_channels, activation=act)
        
        # upsample [1024, 20, 20] -> [512, 20, 20] -> [512, 40, 40]
        self.upsample1 = nn.Sequential(
            Conv(16*base_channels, 8*base_channels, 3, 1, act=act),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        
        # [1024, 40, 40] -> [512, 40, 40]
        self.csp3 = nn.Sequential(
            Conv(16*base_channels, 8*base_channels, 3, 1,act=act),
            CSPLayer(8*base_channels, 8*base_channels, act=act)
        )
        
        # upsample [512, 40, 40] -> [256, 40, 40] -> [256, 80, 80]
        self.upsample2 = nn.Sequential(
            Conv(8*base_channels, 4*base_channels, 3, 1, act=act),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        # [768, 80, 80] -> [512, 80, 80] -> [256, 80, 80]
        self.sk3 = nn.Sequential(
            Conv(12*base_channels, 8*base_channels, 3, 1, act=act),
            SKConv(8*base_channels, 8*base_channels),
            CSPLayer(8 * base_channels, 4 * base_channels, n=3*base_depth, act=act),
            SKConv(4*base_channels, 4*base_channels),
            
        )
        
    def forward(self, x):
        # [b, 6, 640, 640] -> [b, 128, 160, 160]
        x = self.stem(x)
         
        # [b, 128, 160, 160] -> [b, 128, 160, 160]
        left = self.csp1(x)
        # [b, 128, 160, 160] -> [b, 256, 80, 80]
        left = self.csp2(left)
        
        # [b, 128, 160, 160] -> [b, 256, 80, 80]
        right = self.SAG_1(x)
        
        # [b, 256, 80, 80] -> [b, 256, 80, 80]
        right = self.SAG_2(right)
        
        
        # [b, 256, 80, 80] + [b, 256, 80, 80] -> [b, 512, 80, 80]
        x = torch.cat([left, right], dim=1)
         
        # [b, 512, 80, 80] -> [b, 512, 40, 40]
        x = self.sk1(x)
        residual = x.clone()  
        # [b, 512, 40, 40] -> [b, 1024, 20, 20]
        x = self.sk2(x)
         
        # SPP [b, 1024, 20, 20]
        x = self.spp(x)
        
        # [b, 1024, 20, 20] -> [b, 512, 40, 40]
        x = self.upsample1(x)
        #[b, 512, 40, 40] + [b, 512, 40, 40] -> [b, 1024, 40, 40]
        x = torch.cat([x, residual], dim=1)
        # [b, 1024, 40, 40] -> [b, 512, 40, 40]
        x = self.csp3(x)
        
        # [b, 512, 40, 40] -> [b, 256, 80, 80]
        x = self.upsample2(x)
        # 3*[256, 80, 80] -> [768, 80, 80]
        x = torch.cat([x, left, right], dim=1)
        # [768, 80, 80] -> [256, 80, 80]
        x = self.sk3(x)
        
        return x
    
if __name__ == "__main__":
    # pass
    a = torch.randn((2,3,640,640))
    a = SANet(0.33, 0.5)(a)
    print(a.shape)