import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb
import torch
import torch.nn as nn

import torch
import torch.nn as nn

class SiLU(nn.Module):
    """SiLU激活函数"""
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
    
def get_activation(name="silu", inplace=True):
    # inplace为True，将会改变输入的数据 (降低显存)，否则不会改变原输入，只会产生新的输出
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
    """带归一化和激活函数的标准卷积并且保证宽高不变"""
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        """
        :param in_channels: 输入通道
        :param out_channels: 输出通道
        :param ksize: 卷积核大小
        :param stride: 步长
        :param groups: 是否分组卷积
        :param bias: 偏置
        :param act: 所选激活函数
        """
        super().__init__()
        pad         = (ksize - 1) // 2
        self.conv   = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn     = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act    = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
class BiLocalChaFuseReduce(nn.Module):
    def __init__(self, in_high_channels, in_low_channels, out_channels=64, r=4):
        super(BiLocalChaFuseReduce, self).__init__()

        assert in_low_channels == out_channels
        self.high_channels = in_high_channels
        self.low_channels = in_low_channels
        self.out_channels = out_channels
        self.bottleneck_channels = int(out_channels // r)

        self.feature_high = nn.Sequential(
            nn.Conv2d(self.high_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True),
        )

        self.topdown = nn.Sequential(
            nn.Conv2d(self.out_channels, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid()
        )

        self.bottomup = nn.Sequential(
            nn.Conv2d(self.low_channels, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid(),
        )

        self.post = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True),
        )

    def forward(self, xh, xl):
        xh = self.feature_high(xh)
        topdown_wei = self.topdown(xh)
        bottomup_wei = self.bottomup(xl)

        out = 2 * xl * topdown_wei + 2* xh * bottomup_wei
        out = self.post(out)
        return out


class AsymBiChaFuseReduce(nn.Module):
    def __init__(self, in_high_channels, in_low_channels, out_channels=64, r=4):
        super(AsymBiChaFuseReduce, self).__init__()

        assert in_low_channels == out_channels
        self.high_channels = in_high_channels
        self.low_channels = in_low_channels
        self.out_channels = out_channels
        self.bottleneck_channels = int(out_channels // r)

        self.feature_high = nn.Sequential(
            nn.Conv2d(self.high_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

        self.topdown = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.out_channels, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid(),
        )

        self.bottomup = nn.Sequential(
            nn.Conv2d(self.low_channels, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid(),
        )

        self.post = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True),
        )

    def forward(self, xh, xl):
        xh = self.feature_high(xh)

        topdown_wei = self.topdown(xh)
        bottomup_wei = self.bottomup(xl)
        xs = 2 * xl * topdown_wei + 2 * xh * bottomup_wei
        out = self.post(xs)
        return out


class BiGlobalChaFuseReduce(nn.Module):
    def __init__(self, in_high_channels, in_low_channels, out_channels=64, r=4):
        super(BiGlobalChaFuseReduce, self).__init__()

        assert in_low_channels == out_channels
        self.high_channels = in_high_channels
        self.low_channels = in_low_channels
        self.out_channels = out_channels
        self.bottleneck_channels = int(out_channels // r)

        self.feature_high = nn.Sequential(
            nn.Conv2d(self.high_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

        self.topdown = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.out_channels, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid(),
        )

        self.bottomup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.low_channels, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid(),
        )

        self.post = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True),
        )

    def forward(self, xh, xl):
        xh = self.feature_high(xh)

        topdown_wei = self.topdown(xh)
        bottomup_wei = self.bottomup(xl)
        xs = 2 * xl * topdown_wei + 2 * xh * bottomup_wei
        out = self.post(xs)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample):
        super(ResidualBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        residual = x
        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        out = F.relu(x+residual, True)
        return out


class _FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.block(x)


class ASKCResNetFPN(nn.Module):
    def __init__(self, layer_blocks=[4, 4, 4], channels=[8, 16, 32, 64], fuse_mode='AsymBi'):
        super(ASKCResNetFPN, self).__init__()

        stem_width = channels[0]
        self.stem = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, stem_width, 3, 2, 1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(True),

            nn.Conv2d(stem_width, stem_width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(True),

            nn.Conv2d(stem_width, stem_width*2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(stem_width*2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, 1)
        )

        self.layer1 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[0],
                                       in_channels=channels[1], out_channels=channels[1], stride=1)
        self.layer2 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[1],
                                       in_channels=channels[1], out_channels=channels[2], stride=2)
        self.layer3 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[2],
                                       in_channels=channels[2], out_channels=channels[3], stride=2)

        self.fuse23 = self._fuse_layer(channels[3], channels[2], channels[2], fuse_mode)
        self.fuse12 = self._fuse_layer(channels[2], channels[1], channels[1], fuse_mode)

        self.head = _FCNHead(channels[1], 1)

    def forward(self, x):
        _, _, hei, wid = x.shape

        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        out = self.layer3(c2)

        out = F.interpolate(out, size=[hei//8, wid//8], mode='bilinear')
        out = self.fuse23(out, c2)

        out = F.interpolate(out, size=[hei//4, wid//4], mode='bilinear')
        out = self.fuse12(out, c1)

        pred = self.head(out)
        out = F.interpolate(pred, size=[hei, wid], mode='bilinear')

        return out

    def _make_layer(self, block, block_num, in_channels, out_channels, stride):
        downsample = (in_channels != out_channels) or (stride != 1)
        layer = []
        layer.append(block(in_channels, out_channels, stride, downsample))
        for _ in range(block_num-1):
            layer.append(block(out_channels, out_channels, 1, False))
        return nn.Sequential(*layer)

    def _fuse_layer(self, in_high_channels, in_low_channels, out_channels, fuse_mode='AsymBi'):
        assert fuse_mode in ['BiLocal', 'AsymBi', 'BiGlobal']
        if fuse_mode == 'BiLocal':
            fuse_layer = BiLocalChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        elif fuse_mode == 'AsymBi':
            fuse_layer = AsymBiChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        elif fuse_mode == 'BiGlobal':
            fuse_layer = BiGlobalChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        else:
            NameError
        return fuse_layer


class ASKCResUNet(nn.Module):
    def __init__(self, layer_blocks=[4,4,4,4], channels=[8, 16, 32, 64], fuse_mode='AsymBi'):
        super(ASKCResUNet, self).__init__()

        stem_width = int(channels[0])
        self.stem = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, stem_width, 3, 2, 1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(True),

            nn.Conv2d(stem_width, stem_width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(True),

            nn.Conv2d(stem_width, 2*stem_width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(2*stem_width),
            nn.ReLU(True),

            nn.MaxPool2d(3, 2, 1),
        )

        self.layer1 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[0],
                                       in_channels=channels[1], out_channels=channels[1], stride=1)
        self.layer2 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[1],
                                       in_channels=channels[1], out_channels=channels[2], stride=2)
        self.layer3 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[2],
                                       in_channels=channels[2], out_channels=channels[3], stride=2)

        self.deconv2 = nn.ConvTranspose2d(channels[3], channels[2], 4, 2, 1)
        self.fuse2 = self._fuse_layer(channels[2], channels[2], channels[2], fuse_mode)
        self.uplayer2 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[1],
                                         in_channels=channels[2], out_channels=channels[2], stride=1)

        self.deconv1 = nn.ConvTranspose2d(channels[2], channels[1], 4, 2, 1)
        self.fuse1 = self._fuse_layer(channels[1], channels[1], channels[1], fuse_mode)
        self.uplayer1 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[0],
                                         in_channels=channels[1], out_channels=channels[1], stride=1)

        self.head = _FCNHead(channels[1], 1)

    def forward(self, x):
        _, _, hei, wid = x.shape
        #pdb.set_trace()
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        
        deconc2 = self.deconv2(c3)
        #pdb.set_trace()
        fusec2 = self.fuse2(deconc2, c2)
        upc2 = self.uplayer2(fusec2)

        deconc1 = self.deconv1(upc2)
        fusec1 = self.fuse1(deconc1, c1)
        upc1 = self.uplayer1(fusec1)

        pred = self.head(upc1)
        out = F.interpolate(pred, size=[hei, wid], mode='bilinear')
        return out

    def _make_layer(self, block, block_num, in_channels, out_channels, stride):
        layer = []
        downsample = (in_channels != out_channels) or (stride != 1)
        layer.append(block(in_channels, out_channels, stride, downsample))
        for _ in range(block_num-1):
            layer.append(block(out_channels, out_channels, 1, False))
        return nn.Sequential(*layer)

    def _fuse_layer(self, in_high_channels, in_low_channels, out_channels, fuse_mode='AsymBi'):
        assert fuse_mode in ['BiLocal', 'AsymBi', 'BiGlobal']
        if fuse_mode == 'BiLocal':
            fuse_layer = BiLocalChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        elif fuse_mode == 'AsymBi':
            fuse_layer = AsymBiChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        elif fuse_mode == 'BiGlobal':
            fuse_layer = BiGlobalChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        else:
            NameError
        return fuse_layer









class ACMHead(nn.Module):
    def __init__(self, num_classes, in_channels = 256, act='silu'):
        super().__init__()
        # 分类的两层卷积
        self.cls_convs = nn.Sequential(
            BaseConv(int(in_channels), int(in_channels), 3, 1, act=act),
            BaseConv(int(in_channels), int(in_channels), 3, 1, act=act)
        )
        # 分类的预测
        self.cls_preds = nn.Conv2d(in_channels=int(in_channels), out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        
        # 回归的两层卷积
        self.reg_convs = nn.Sequential(
            BaseConv(int(in_channels), int(in_channels), 3, 1, act=act),
            BaseConv(int(in_channels), int(in_channels), 3, 1, act=act)
        )
        # 回归的预测
        self.reg_preds = nn.Conv2d(in_channels=int(in_channels), out_channels=4, kernel_size=1, stride=1, padding=0)
        
        # 是否有检测对象预测层
        self.obj_preds = nn.Conv2d(in_channels=int(in_channels), out_channels=1, kernel_size=1, stride=1, padding=0)
        
    def forward(self, inputs):
        # inputs [b, 256, 32, 32]
        outputs = []
        # 分类提取器 [b, 256, 32, 32] -> [b, 256, 32, 32]
        cls_feat = self.cls_convs(inputs)
        # 分类的输出 [b, 256, 32, 32] -> [b, num_classes, 32, 32]
        cls_output = self.cls_preds(cls_feat)
        
        # 回归特征提取 [b, 256, 32, 32] -> [b, 256, 32, 32]
        reg_feat = self.reg_convs(inputs)
        # 特征点的回归系数 [b, 256, 32, 32] -> [b, 4, 32, 32]
        reg_output = self.reg_preds(reg_feat)
        
        # 判断特征点是否有对应的物体(利用回归特征提取) [b, 256, 32, 32] -> [b, 1, 32, 32]
        obj_output = self.obj_preds(reg_feat)
        
        # 将结果整合到一起，0到3为回归结果，4为是否有物体的结果，其余为种类置信得分
        # [b, 4, 32, 32] + [b, 1, 32, 32] + [b, num_classes, 32, 32] -> [b, 5+num_classes, 32, 32]
        output = torch.cat([reg_output, obj_output, cls_output], dim=1)
        outputs.append(output)
        
        return outputs

class ACMBody(nn.Module):
    def __init__(self, num_classes, phi):
        super().__init__()
        # self.backbone = ASKCResNetFPN()
        self.backbone = ASKCResUNet()
        self.head = ACMHead(num_classes)
        self.conv = nn.Sequential(  
            BaseConv(1, 4, 3, 2), # [b, 4, 320, 320]
            BaseConv(4, 16, 3, 2), # [b, 16, 160, 160]
            BaseConv(16, 64, 3, 2), # [b, 64, 80, 80]
            BaseConv(64, 256, 3, 1), # [b, 256, 80, 80]
        )

    def forward(self, x):
        #单帧
        #pdb.set_trace()
        # if len(x.shape)>4:
        #     x=x.squeeze(2)
        b, _, _, _ = x.shape
        x = self.backbone(x)  # [b, 1, 640, 640]
        # pdb.set_trace()
        # x = x.view(b, 64, 80, 80)
        x = self.conv(x)
        outputs = self.head(x)
        return outputs



if __name__ == "__main__":
    a = torch.rand([2, 3, 512, 512])#[4, 3, 640, 640]
    a = ACMBody(1,'s')(a)
    print(a[0].shape)