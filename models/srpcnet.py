#!/usr/bin/python3
#coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


############################  Tensor Operations  ############################

def cus_sample(feat, **kwargs):
    """
    :param feat: 输入特征
    :param kwargs: size或者scale_factor
    """
    assert len(kwargs.keys()) == 1 and list(kwargs.keys())[0] in ["size", "scale_factor"]
    return F.interpolate(feat, **kwargs, mode="bilinear", align_corners=False)


def upsample_add(*xs):
    y = xs[-1]
    for x in xs[:-1]:
        y = y + F.interpolate(x, size=y.size()[2:], mode="bilinear", align_corners=False)
    return y


def upsample_cat(*xs):
    y = xs[-1]
    out = []
    for x in xs[:-1]:
        out.append(F.interpolate(x, size=y.size()[2:], mode="bilinear", align_corners=False))
    return torch.cat([*out, y], dim=1)


def upsample_reduce(b, a):
    """
    上采样所有特征到最后一个特征的尺度以及前一个特征的通道数
    """
    _, C, _, _ = b.size()
    N, _, H, W = a.size()

    b = F.interpolate(b, size=(H, W), mode="bilinear", align_corners=False)
    a = a.reshape(N, -1, C, H, W).mean(1)

    return b + a


def shuffle_channels(x, groups):
    """
    Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,W] -> [N,C,H,W]
    一共C个channel要分成g组混合的channel，先把C reshape成(g, C/g)的形状，
    然后转置成(C/g, g)最后平坦成C组channel
    """
    N, C, H, W = x.size()
    x = x.reshape(N, groups, C // groups, H, W).permute(0, 2, 1, 3, 4)
    return x.reshape(N, C, H, W)


############################  ConvBR  ############################

class ConvBR(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, dilation=1, stride=1, padding=0):
        super(ConvBR, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, dilation=dilation, 
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


############################  SPA Module  ############################

class SqueezeExcitation(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            reduction: int = 4,
            activation=nn.ReLU,
            scale_activation=nn.Sigmoid,
            pool='avgpool'
    ):
        super(SqueezeExcitation, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels:
            self.transition = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

        if out_channels // reduction == 0:
            reduction = 1

        if pool == 'avgpool':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool == 'maxpool':
            self.pool = nn.AdaptiveMaxPool2d(1)
        else:
            print('Parameter pool is not avgpool or maxpool')
            return
        self.fc1 = nn.Conv2d(out_channels, out_channels // reduction, 1)
        self.fc2 = nn.Conv2d(out_channels // reduction, out_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, x):
        scale = self.pool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            x = self.transition(x)
        scale = self._scale(x)
        return scale * x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, bias=True, se_ratio=2):
        super(DepthwiseSeparableConv, self).__init__()

        if dilation != 1:
            padding = dilation

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                                   groups=in_channels, bias=False, stride=stride)
        self.bn = nn.BatchNorm2d(in_channels)
        self.se = SqueezeExcitation(in_channels=in_channels, out_channels=in_channels, reduction=se_ratio)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = torch.relu(self.bn(out))
        out = self.se(out)
        out = self.pointwise(out)
        return out


class SPA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SPA, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=True)
        self.convbr = ConvBR(out_channels * 3, out_channels, kernel_size=1)

        self.conv1_1 = DepthwiseSeparableConv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=True)
        self.conv1_2 = DepthwiseSeparableConv(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 3), padding=(0, 1), bias=True)
        self.conv1_3 = DepthwiseSeparableConv(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 1), padding=(1, 0), bias=True)

        self.atconv1 = ConvBR(in_planes=out_channels, out_planes=out_channels, kernel_size=3, dilation=1, stride=1, padding=1)
        self.atconv2 = ConvBR(in_planes=out_channels, out_planes=out_channels, kernel_size=3, dilation=2, stride=1, padding=2)
        self.atconv3 = ConvBR(in_planes=out_channels, out_planes=out_channels, kernel_size=3, dilation=3, stride=1, padding=3)

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        xx = self.conv(x)
        x1 = self.conv1_1(xx) + self.conv1_2(xx) + self.conv1_3(xx)
        x1 = self.bn(x1)
        x2 = torch.cat([self.atconv1(xx), self.atconv2(xx), self.atconv3(xx)], dim=1)
        x2 = self.convbr(x2)
        out = x1 + x2 + xx
        return out


############################  HAIM Module  ############################

class GlobalContextBlock(nn.Module):
    def __init__(self, inplanes, ratio, pooling_type='att', fusion_types=('channel_add',)):
        super(GlobalContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            input_x = input_x.view(batch, channel, height * width)
            input_x = input_x.unsqueeze(1)
            context_mask = self.conv_mask(x)
            context_mask = context_mask.view(batch, 1, height * width)
            context_mask = self.softmax(context_mask)
            context_mask = context_mask.unsqueeze(-1)
            context = torch.matmul(input_x, context_mask)
            context = context.view(batch, channel, 1, 1)
        else:
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


class HAIM(nn.Module):
    def __init__(self, num_channels=64):
        super(HAIM, self).__init__()

        self.conv_cross1 = nn.Conv2d(3 * num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_cross2 = nn.Conv2d(2 * num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_cross = nn.BatchNorm2d(num_channels)
        self.Global = GlobalContextBlock(64, 0.25)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.convg = nn.Conv2d(64, 64, 1)
        self.sftmax = nn.Softmax(dim=1)

    def forward(self, in1, in2, in3=None):

        if in3 != None:
            in2 = F.interpolate(in2, size=in1.size()[2:], mode='bilinear')
            in3 = F.interpolate(in3, size=in1.size()[2:], mode='bilinear')
            in3_c = self.convg(self.gap(in3))
            in22 = torch.mul(self.sftmax(in3_c) * in3_c.shape[1], in2)
            in33 = in3*F.sigmoid(in2)
            x = torch.cat((in1, in22, in33), 1)
            x = F.relu(self.bn_cross(self.conv_cross1(x)))
        else:
            if in1.size()[2] > in2.size()[2]:
                in2 = F.interpolate(in2, size=in1.size()[2:], mode='bilinear')
                in1_c = self.convg(self.gap(in1))
                in11 = torch.mul(self.sftmax(in1_c) * in1_c.shape[1], in2)
                in22 = in1 * F.sigmoid(in2)
                x = torch.cat((in11, in22), 1)
            else:
                in2 = F.interpolate(in2, size=in1.size()[2:], mode='bilinear')
                in2_c = self.convg(self.gap(in2))
                in22 = torch.mul(self.sftmax(in2_c) * in2_c.shape[1], in1)
                in11 = in2 * F.sigmoid(in1)
                x = torch.cat((in11, in22), 1)
            x = F.relu(self.bn_cross(self.conv_cross2(x)))

        context = self.Global(x)
        out = x * context

        return out


############################  BSFM Module  ############################

class BSFM(nn.Module):
    def __init__(self, h_C, l_C):
        super(BSFM, self).__init__()
        self.h2l_pool = nn.AvgPool2d((2, 2), stride=2)
        self.l2h_up = cus_sample
        self.relu = nn.ReLU(True)

        self.h2l_0 = nn.Conv2d(h_C, l_C, 3, 1, 1)
        self.h2h_0 = nn.Conv2d(h_C, h_C, 3, 1, 1)
        self.bnl_0 = nn.BatchNorm2d(l_C)
        self.bnh_0 = nn.BatchNorm2d(h_C)

        self.h2h_1 = nn.Conv2d(h_C, h_C, 3, 1, 1)
        self.h2l_1 = nn.Conv2d(h_C, l_C, 3, 1, 1)
        self.l2h_1 = nn.Conv2d(l_C, h_C, 3, 1, 1)
        self.l2l_1 = nn.Conv2d(l_C, l_C, 3, 1, 1)
        self.bnl_1 = nn.BatchNorm2d(l_C)
        self.bnh_1 = nn.BatchNorm2d(h_C)

        self.h2h_2 = nn.Conv2d(h_C, h_C, 3, 1, 1)
        self.l2h_2 = nn.Conv2d(l_C, h_C, 3, 1, 1)

        self.layer1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.layer2_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.layer2_2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.layer_fu = ConvBR(32, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h, w = x.shape[2:]

        # first conv
        x_h = self.relu(self.bnh_0(self.h2h_0(x)))
        x_l = self.relu(self.bnl_0(self.h2l_0(self.h2l_pool(x))))

        # mid conv
        x_h2h = self.h2h_1(x_h)
        x_h2l = self.h2l_1(self.h2l_pool(x_h))
        x_l2l = self.l2l_1(x_l)
        x_l2h = self.l2h_1(self.l2h_up(x_l, size=(h, w)))
        x_h = self.relu(self.bnh_1(x_h2h + x_l2h))
        x_l = self.relu(self.bnl_1(x_l2l + x_h2l))

        # last conv
        x_h2h = self.h2h_2(x_h)
        x_l2h = self.l2h_2(self.l2h_up(x_l, size=(h, w)))

        # Feature Fusion
        weight = nn.Sigmoid()(self.layer1(x_h2h + x_l2h))
        xw_resid_1 = x_h2h + x_h2h.mul(weight)
        xw_resid_2 = x_l2h + x_l2h.mul(weight)
        x1_2 = self.layer2_1(xw_resid_1)
        x2_2 = self.layer2_2(xw_resid_2)
        x_h = self.layer_fu(x1_2 + x2_2)

        return x_h + x


############################  VGG Encoder  ############################

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        feats = list(models.vgg16_bn(pretrained=True).features.children())
        self.conv1 = nn.Sequential(*feats[:6])
        self.conv2 = nn.Sequential(*feats[6:13])
        self.conv3 = nn.Sequential(*feats[13:23])
        self.conv4 = nn.Sequential(*feats[23:33])
        self.conv5 = nn.Sequential(*feats[33:43])

        self.initialize()

        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

    def forward(self, x):
        E1 = self.conv1(x)
        E2 = self.conv2(E1)
        E3 = self.conv3(E2)
        E4 = self.conv4(E3)
        E5 = self.conv5(E4)

        return E1, E2, E3, E4, E5

    def initialize(self):
        pass


############################  Weight Initialization  ############################

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool1d, nn.Identity)):
            pass


############################  SRPCNet Main Model  ############################

class SRPCNet(nn.Module):
    def __init__(self, cfg=None, model_name='SRPC'):
        super(SRPCNet, self).__init__()
        self.cfg = cfg
        self.model_name = model_name
        
        if self.model_name == "SRPC":
            #### VGG Encoder ####
            self.encoder = VGG()
        else:
            print("UNDEFINED BACKBONE NAME.")

        # SPA
        self.spa1 = SPA(64, 64)
        self.spa2 = SPA(128, 64)
        self.spa3 = SPA(256, 64)
        self.spa4 = SPA(512, 64)
        self.spa5 = SPA(512, 64)

        # HAIM
        self.haim1 = HAIM()
        self.haim2 = HAIM()
        self.haim3 = HAIM()
        self.haim4 = HAIM()
        self.haim5 = HAIM()

        # BSFM
        self.upsample_add = upsample_add
        self.upsample = cus_sample
        self.bsfm1 = BSFM(64, 32)
        self.bsfm2 = BSFM(64, 32)
        self.bsfm3 = BSFM(64, 32)
        self.bsfm4 = BSFM(64, 32)
        self.bsfm5 = BSFM(64, 32)
        self.upconv1 = ConvBR(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = ConvBR(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv3 = ConvBR(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = ConvBR(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv5 = ConvBR(64, 64, kernel_size=3, stride=1, padding=1)

        # Predtrans
        self.predtrans1 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.predtrans2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.predtrans3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.predtrans4 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.predtrans5 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        self.initialize()

    def forward(self, x, shape=None):
        features = self.encoder(x)

        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        # SPA
        x1 = self.spa1(x1)
        x2 = self.spa2(x2)
        x3 = self.spa3(x3)
        x4 = self.spa4(x4)
        x5 = self.spa5(x5)

        # HAIM
        x5 = self.haim5(in1=x5, in2=x4)      # hiam5 → haim5
        x4 = self.haim4(in1=x4, in2=x5, in3=x3)  # hiam4 → haim4
        x3 = self.haim3(in1=x3, in2=x4, in3=x2)  # hiam3 → haim3
        x2 = self.haim2(in1=x2, in2=x3, in3=x1)  # hiam2 → haim2
        x1 = self.haim1(in1=x1, in2=x2)          # 已经正确
        # BSFM
        x5 = self.bsfm5(x5)
        x5_up = self.upconv5(x5)

        x4 = self.upsample_add(x5_up, x4)
        x4 = self.bsfm4(x4)
        x4_up = self.upconv4(x4)

        x3 = self.upsample_add(x4_up, x3)
        x3 = self.bsfm3(x3)
        x3_up = self.upconv3(x3)

        x2 = self.upsample_add(x3_up, x2)
        x2 = self.bsfm2(x2)
        x2_up = self.upconv2(x2)

        x1 = self.upsample_add(x2_up, x1)
        x1 = self.bsfm1(x1)
        x1_up = self.upconv1(x1)

        if shape is None:
            shape = x.size()[2:]

        pred1 = F.interpolate(self.predtrans1(x1_up), size=shape, mode='bilinear')
        pred2 = F.interpolate(self.predtrans2(x2_up), size=shape, mode='bilinear')
        pred3 = F.interpolate(self.predtrans3(x3_up), size=shape, mode='bilinear')
        pred4 = F.interpolate(self.predtrans4(x4_up), size=shape, mode='bilinear')
        pred5 = F.interpolate(self.predtrans5(x5_up), size=shape, mode='bilinear')

        return pred1, pred2, pred3, pred4, pred5

    def initialize(self):
        if self.cfg and hasattr(self.cfg, 'snapshot') and self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self)


############################  Main  ############################

if __name__ == '__main__':
    # Test the model
    model = SRPCNet()
    input_tensor = torch.randn(1, 3, 352, 352)
    outputs = model(input_tensor)
    
    print("Model created successfully!")
    print(f"Number of outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"Output {i+1} shape: {out.shape}")
    
    # Optionally compute model complexity
    # Requires: pip install ptflops
    try:
        from ptflops import get_model_complexity_info
        with torch.cuda.device(0):
            macs, params = get_model_complexity_info(model, (3, 352, 352), 
                                                      as_strings=True, 
                                                      print_per_layer_stat=False, 
                                                      verbose=False)
            print(f'\nComputational complexity: {macs}')
            print(f'Number of parameters: {params}')
    except ImportError:
        print("\nInstall ptflops to see model complexity: pip install ptflops")
