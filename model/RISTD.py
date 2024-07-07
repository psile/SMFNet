import torch
import torch.nn as nn
import torch
import torch.nn as nn
from torch.nn import functional as F

# from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

global global_step
global_step = 0

# conv_writer = SummaryWriter(comment='--conv')

import numpy as np

kernels_all = {}  # key:e->kind_kernels;values:[]->kernels
num_cycle = [1, 2, 3, 4, 5]  #


def GenerateKernels():
    """
    生成固定权值卷积核
    :return: None
    """
    for i in num_cycle:  # 第i种卷积核
        kernels = {}
        for j in range(i):  # 第i种卷积核的第j个卷积核
            k_size = (2 * i) + 1  # 卷积核的尺寸
            kernel = np.zeros(shape=(k_size, k_size)).astype(np.float32)  # 生成卷积核
            lt_y = lt_x = k_size // 2 - ((j + 1) * 2 - 1) // 2  # 红色区域左上角的x y轴索引
            red_size = (j + 1) * 2 - 1  # 红色区域尺寸
            # 给中间红色区域值
            red_val = 1 / kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size].size  # 红色区域填充值
            kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size] = red_val  # 赋值
            # 给左、右、上、下蓝色区域赋值
            blue_val = -1 / (k_size ** 2 - kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size].size)  # 蓝色区域填充值
            kernel[0:lt_x, :] = kernel[lt_x + red_size:, :] = kernel[:, :lt_y] = kernel[
                                                                                 :,
                                                                                 lt_y + red_size:] = blue_val  # 赋值
            # 添加到第i中卷积核中
            kernels[j + 1] = kernel
        # 添加到所有卷积核中
        kernels_all[i] = kernels
        pass


# 生成卷积核
GenerateKernels()


def get_kernels(kind):
    """
    获取某种卷积核的所有卷积核
    :param kind: 卷积核种类 1~5
    :return: [kernels of kind]
    """
    try:
        return list(kernels_all[kind].values())
    except KeyError:
        print('下标不对！')
import torch


def GenLikeMap(feature, batch_size, W, H):
    # 生成似然图
    likelihood = torch.full((batch_size, W, H), -float('inf')).cuda()
    for i in range(feature.shape[2]):
        for j in range(feature.shape[3]):
            bl = feature[:, :, i, j]
            likelihood[:, i * 8:i * 8 + 8, j * 8:j * 8 + 8] = bl.reshape(-1, 8, 8)
    return likelihood


class FENetwFW(nn.Module):
    """
    基于固定权值卷积核的特征提取模块
    A feature extraction network based on convolution kernel with fixed weight.(FENetwFW)
    """

    def __init__(self):
        super(FENetwFW, self).__init__()
        kernels = [get_kernels(i) for i in range(1, 6)]  # 获取各种尺寸的卷积核
        self.weights = [
            nn.Parameter(data=torch.FloatTensor(k).unsqueeze(0).unsqueeze(0), requires_grad=False).cuda()
            for ks in kernels for k in ks
        ]  # 将卷积核转换为成pytorch中的格式

    def forward(self, img):
        feature_maps = [img]  # 融合的特征
        for ws in self.weights:  # 用各个卷积核对图像进行卷积，提取不同的特征图
            feature_maps.append(F.conv2d(img, ws, stride=1, padding = 'same'))

        feature_maps = torch.cat(feature_maps, dim=1)  # 对各个卷积核卷积的结果进行融合

        # conv_writer.add_image('固定权值卷积',
        #                       make_grid(torch.unsqueeze(feature_maps[0], dim=0).transpose(0, 1), normalize=True,
        #                                 nrow=3),
        #                       global_step=global_step)
        return feature_maps


class FENetwVW(nn.Module):
    """
    基于变化权值卷积核的特征提取模块
    A Feature extraction network based on convolution kernel with variable weight
    """

    def __init__(self):
        super(FENetwVW, self).__init__()
        self.c1 = nn.Conv2d(16, 32, kernel_size=(11, 11), padding='same', stride=(1, 1), bias=None)  # Convolution_1
        # torch.nn.init.xavier_normal_(self.c1.weight, gain=1.0)
        self.c2 = P1C2(32, 64)  # Pooling_1 & Convolution_2
        self.c3 = P2C3(64, 128)  # Pooling_2 & Convolution_3
        self.FCsubnet = FCsubnet(128, 256)  # Feature concatenation subnetwork
        self.c5 = nn.Conv2d(768, 128, kernel_size=(1, 1), padding='same', stride=(1, 1), bias=None)  # Convolution_5

    def forward(self, fw_out):
        global global_step

        c1 = self.c1(fw_out)
        # conv_writer.add_image('c1',
        #                       make_grid(torch.unsqueeze(c1[0], dim=0).transpose(0, 1), normalize=True, nrow=4),
        #                       global_step=global_step)
        c2 = self.c2(c1)
        # conv_writer.add_image('c2',
        #                       make_grid(torch.unsqueeze(c2[0], dim=0).transpose(0, 1), normalize=True, nrow=4),
        #                       global_step=global_step)
        c3 = self.c3(c2)
        # conv_writer.add_image('c3',
        #                       make_grid(torch.unsqueeze(c3[0], dim=0).transpose(0, 1), normalize=True, nrow=4),
        #                       global_step=global_step)
        FC_subnet = self.FCsubnet(c3)
        # conv_writer.add_image('特征级联子网络',
        #                       make_grid(torch.unsqueeze(FC_subnet[0], dim=0).transpose(0, 1), normalize=True, nrow=4),
        #                       global_step=global_step)
        c5 = self.c5(FC_subnet)
        # conv_writer.add_image('c5',
        #                       make_grid(torch.unsqueeze(c5[0], dim=0).transpose(0, 1), normalize=True, nrow=4),
        #                       global_step=global_step)

        global_step += 1
        return c5


class FMNet(nn.Module):
    """
    特征图映射
    Feature mapping process
    """

    def __init__(self):
        super(FMNet, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, FV, batch_size, W, H):
        # first64 = FV[:, :64, :, :]  # 背景似然
        last64 = FV[:, 64:, :, :]  # 目标似然

        # bg_likemap = GenLikeMap(first64, batch_size, W, H)
        tg_likemap = GenLikeMap(last64, batch_size, W, H)
        # bg_likelihood = self.sigmoid(
        #     bg_likemap)

        tg_likelihood = self.sigmoid(
            tg_likemap)
        return torch.unsqueeze(tg_likelihood, dim=1)


class FCsubnet(nn.Module):
    """
    特征级联子网络
    A Feature concatenation subnetwork
    """

    def __init__(self, in_c, out_c):
        super(FCsubnet, self).__init__()
        self.reorg = ReOrg()  # 特征重组
        self.p3c4 = P3C4(in_c, out_c)  # Pooling_3 & Convolution_4

    def forward(self, c3):
        return torch.cat([self.reorg(c3), self.p3c4(c3)], dim=1)  # 特征拼接


class ReOrg(nn.Module):
    """
    特征重组
    """

    def __init__(self):
        super(ReOrg, self).__init__()

    def forward(self, p2c3):
        w = p2c3.shape[2]
        h = p2c3.shape[3]
        pink = p2c3[:, :, :w // 2, :h // 2]
        green = p2c3[:, :, w // 2:, :h // 2]
        purple = p2c3[:, :, :w // 2, h // 2:]
        red = p2c3[:, :, w // 2:, h // 2:]
        return torch.cat([pink, green, purple, red], dim=1)


class P1C2(nn.Module):
    def __init__(self, in_c, out_c):
        super(P1C2, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_c, out_c, kernel_size=(7, 7), padding='same', stride=(1, 1), bias=None)

    def forward(self, c1):
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        return c2


class P2C3(nn.Module):
    def __init__(self, in_c, out_c):
        super(P2C3, self).__init__()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=(5, 5), padding='same', stride=(1, 1), bias=None)


    def forward(self, c2):
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        return c3


class P3C4(nn.Module):
    def __init__(self, in_c, out_c):
        super(P3C4, self).__init__()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(in_c, out_c, kernel_size=(3, 3), padding='same', stride=(1, 1), bias=None)


    def forward(self, c3):
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        return c4

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


class RISTDHead(nn.Module):
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

class RISTDnet(nn.Module):
    def __init__(self):
        super(RISTDnet, self).__init__()
        self.FW = FENetwFW()
        self.FV = FENetwVW()
        self.FM = FMNet()

    def forward(self, img):
        FW_out = self.FW.forward(img)
        FV_out = self.FV(FW_out)
        FM_out = self.FM(FV_out, img.shape[0], img.shape[2], img.shape[3])
        return FM_out

class RISTDBody(nn.Module):
    def __init__(self, num_classes, phi):
        super().__init__()
        self.down = BaseConv(3, 1, 3, 1)
        self.backbone = RISTDnet()
        self.head = RISTDHead(num_classes)
        self.conv = nn.Sequential(
            BaseConv(1, 4, 3, 2), #[b, 4, 320, 320]
            BaseConv(4, 16, 3, 2), #[b, 16, 160, 160]
            BaseConv(16, 64, 3, 2), #[b, 64, 80, 80]
            BaseConv(64, 256, 3, 1) #[b, 256, 80, 80]
        )

    def forward(self, x):
        b,_,_,_ = x.shape
        x = self.backbone(self.down(x))  # [b, 1, 640, 640]
        # x = x.view(b, 64, 80, 80)
        x = self.conv(x)
        # x = x.view(b, 256, 32, 32)
        outputs = self.head(x)

        return outputs


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = torch.rand([1,3,640,640])
    net = RISTDBody(1, 's').to(device)
    a = net(a.to(device))
    print(a[0].shape)
