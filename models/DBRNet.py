import torch
from torch import nn, Tensor
from torch.nn import functional as F
import time

# ==================== 基础模块 ====================

class FastDownSample(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, groups=c1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True),
            nn.Conv2d(c2, c2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(c2)
        )


class FastDownSample2d(nn.Sequential):
    def __init__(self, c1, ch, c2, k, s=1, p=0):
        super().__init__(
            nn.Conv2d(c1, ch, k, s, p, groups=c1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(True),
            nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(ch),
            nn.Conv2d(ch, c2, k, s, p, bias=False, groups=ch),
            nn.BatchNorm2d(c2),
            nn.ReLU(True),
            nn.Conv2d(c2, c2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(c2),
        )


class ConvBN(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, bias=False),
            nn.BatchNorm2d(c2)
        )


class Conv2BN(nn.Sequential):
    def __init__(self, c1, ch, c2, k, s=1, p=0):
        super().__init__(
            nn.Conv2d(c1, ch, k, s, p, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(True),
            nn.Conv2d(ch, c2, k, s, p, bias=False),
            nn.BatchNorm2d(c2)
        )


class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0):
        super().__init__(
            nn.BatchNorm2d(c1),
            nn.ReLU(True),
            nn.Conv2d(c1, c2, k, s, p, bias=False)
        )


class DWConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0):
        super().__init__(
            nn.BatchNorm2d(c1),
            nn.ReLU(True),
            nn.Conv2d(c1, c2, k, s, p, groups=c1, bias=False)
        )


class Scale(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0):
        super().__init__(
            nn.AvgPool2d(k, s, p),
            nn.BatchNorm2d(c1),
            nn.ReLU(True),
            nn.Conv2d(c1, c2, 1, bias=False)
        )


class ScaleLast(nn.Sequential):
    def __init__(self, c1, c2, k):
        super().__init__(
            nn.AdaptiveAvgPool2d(k),
            nn.BatchNorm2d(c1),
            nn.ReLU(True),
            nn.Conv2d(c1, c2, 1, bias=False)
        )


class Stem(nn.Sequential):
    def __init__(self, c1, c2):
        super().__init__(
            nn.Conv2d(c1, c2, 3, 2, 1),
            nn.BatchNorm2d(c2),
            nn.ReLU(True),
            nn.Conv2d(c2, c2, 3, 2, 1),
            nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)
    return x


# ==================== Backbone模块 ====================

class BasicBlockOriginal(nn.Module):
    expansion = 1

    def __init__(self, c1, c2, s=1, downsample=None, no_relu=False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, 3, s, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(c2, c2, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out if self.no_relu else F.relu(out)


class LFEM(nn.Module):
    expansion = 1
    middleExpansion = 2

    def __init__(self, c1, c2, s=1, downsample=None, no_relu=False) -> None:
        super().__init__()
        self.conv2_last = None
        if s == 2:
            self.conv1 = nn.Conv2d(c1, c2 * self.middleExpansion, 1, 1, 0, bias=False)
            self.bn1 = nn.BatchNorm2d(c2 * self.middleExpansion)
            self.conv2 = nn.Conv2d(c2 * self.middleExpansion, c2 * self.middleExpansion, 3, s, 1,
                                   groups=c2 * self.middleExpansion,
                                   bias=False)
            self.bn2 = nn.BatchNorm2d(c2 * self.middleExpansion)

            self.conv3 = nn.Conv2d(c2 * self.middleExpansion, c2, 1, 1, 0)
            self.bn3 = nn.BatchNorm2d(c2)
            self.downsample = downsample
            self.no_relu = no_relu
        elif s == 1:
            self.conv1 = nn.Conv2d(c1, c2 * self.middleExpansion, 1, 1, 0, bias=False)
            self.bn1 = nn.BatchNorm2d(c2 * self.middleExpansion)

            # 3*3的dw卷积
            self.conv2 = nn.Conv2d(c2, c2, 3, s, 1,
                                   groups=c2,
                                   bias=False)
            self.bn2 = nn.BatchNorm2d(c2)

            # 中间加入的5*5dw卷积
            self.conv2_last = nn.Conv2d(c2, c2, 5, s, 2,
                                         groups=c2,
                                         bias=False)
            self.bn2_last_bn = nn.BatchNorm2d(c2)

            self.conv3 = nn.Conv2d(c2 * self.middleExpansion, c2, 1, 1, 0)
            self.bn3 = nn.BatchNorm2d(c2)
            self.downsample = downsample
            self.no_relu = no_relu

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        if self.conv2_last is not None:
            x1, x2 = out.chunk(2, dim=1)
            out1 = self.bn2(self.conv2(x1))
            out2 = self.bn2_last_bn(self.conv2_last(x2))
            out = torch.concat((out1, out2), 1)
            out = F.relu(out)
        else:
            out1 = self.bn2(self.conv2(out))
            out = F.relu(out1)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        if self.conv2_last is not None:
            out = channel_shuffle(out, 2)
        return out if self.no_relu else F.relu(out)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, c1, c2, s=1, downsample=None, no_relu=False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(c2, c2, 3, s, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.conv3 = nn.Conv2d(c2, c2 * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(c2 * self.expansion)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out if self.no_relu else F.relu(out)


# ==================== AlignedModule ====================

class AlignedModule(nn.Module):
    def __init__(self, c1, c2, k=3):
        super().__init__()
        self.down_h = nn.Conv2d(c1, c2, 1, bias=False)
        self.down_l = nn.Conv2d(c1, c2, 1, bias=False)
        self.flow_make = nn.Conv2d(c2 * 2, 2, k, 1, 1, bias=False)

    def forward(self, low_feature: Tensor, high_feature: Tensor) -> Tensor:
        high_feature_origin = high_feature
        H, W = low_feature.shape[-2:]
        low_feature = self.down_l(low_feature)
        high_feature = self.down_h(high_feature)
        high_feature = F.interpolate(high_feature, size=(H, W), mode='bilinear', align_corners=True)
        flow = self.flow_make(torch.cat([high_feature, low_feature], dim=1))
        high_feature = self.flow_warp(high_feature_origin, flow, (H, W))
        return high_feature

    def flow_warp(self, x: Tensor, flow: Tensor, size: tuple) -> Tensor:
        norm = torch.tensor([[[[*size]]]]).type_as(x).to(x.device)
        H = torch.linspace(-1.0, 1.0, size[0]).view(-1, 1).repeat(1, size[1])
        W = torch.linspace(-1.0, 1.0, size[1]).repeat(size[0], 1)
        grid = torch.cat((W.unsqueeze(2), H.unsqueeze(2)), dim=2)
        grid = grid.repeat(x.shape[0], 1, 1, 1).type_as(x).to(x.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm
        output = F.grid_sample(x, grid, align_corners=False)
        return output


# ==================== DCPPM模块 ====================

class DCPPM(nn.Module):
    def __init__(self, c1, ch, c2):
        super().__init__()
        self.scale1 = Scale(c1, ch, 5, 2, 2)
        self.scale2 = Scale(c1, ch, 9, 4, 4)
        self.scale3 = Scale(c1, ch, 17, 8, 8)
        self.scale4 = ScaleLast(c1, ch, 1)
        self.scale0 = ConvModule(c1, ch, 1)
        # 将ConvModule更改为DWConvModule
        self.process1 = DWConvModule(ch, ch, 3, 1, 1)
        self.process2 = DWConvModule(ch, ch, 3, 1, 1)
        self.process3 = DWConvModule(ch, ch, 3, 1, 1)
        self.process4 = DWConvModule(ch, ch, 3, 1, 1)
        self.compression = ConvModule(ch * 5, c2, 1)
        self.shortcut = ConvModule(c1, c2, 1)

    def forward(self, x: Tensor) -> Tensor:
        outs = [self.scale0(x)]

        outUnsamplesSale1 = F.interpolate(self.scale1(x), size=x.shape[-2:], mode='bilinear', align_corners=False)
        outUnsamplesSale2 = F.interpolate(self.scale2(x), size=x.shape[-2:], mode='bilinear', align_corners=False)
        outUnsamplesSale3 = F.interpolate(self.scale3(x), size=x.shape[-2:], mode='bilinear', align_corners=False)
        outUnsamplesSale4 = F.interpolate(self.scale4(x), size=x.shape[-2:], mode='bilinear', align_corners=False)

        outProcess1 = self.process1(outUnsamplesSale1 + outs[-1])
        outs.append(outProcess1)
        addBranch2_1 = outUnsamplesSale2 + outs[-1]
        outProcess2 = self.process2(addBranch2_1)
        outs.append(outProcess2)
        addBranch3_1 = outUnsamplesSale3 + addBranch2_1
        addBranch3_2 = addBranch3_1 + outs[-1]
        outProcess3 = self.process3(addBranch3_2)
        outs.append(outProcess3)
        addBranch4_1 = outUnsamplesSale4 + addBranch3_1
        addBranch4_2 = addBranch4_1 + addBranch3_2
        addBranch4_3 = addBranch4_2 + outs[-1]
        outProcess4 = self.process4(addBranch4_3)
        outs.append(outProcess4)
        out = self.compression(torch.cat(outs, dim=1)) + self.shortcut(x)
        return out


# ==================== SegHead ====================

class SegHead(nn.Module):
    def __init__(self, c1, ch, c2, scale_factor=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(c1)
        self.conv1 = nn.Conv2d(c1, ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, c2, 1)
        self.scale_factor = scale_factor

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(x)))

        if self.scale_factor is not None:
            H, W = x.shape[-2] * self.scale_factor, x.shape[-1] * self.scale_factor
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out


# ==================== DBRNet主模型 ====================

class DBRNet(nn.Module):
    def __init__(self, backbone: str = None, num_classes: int = 19) -> None:
        super().__init__()
        planes, spp_planes, head_planes = [32, 64, 128, 256, 512], 128, 64
        self.conv1 = Stem(3, planes[0])

        self.layer1 = self._make_layer(BasicBlockOriginal, planes[0], planes[0], 2)
        self.layer2 = self._make_layer(LFEM, planes[0], planes[1], 2, 2)
        self.layer3 = self._make_layer(BasicBlockOriginal, planes[1], planes[2], 2, 2)
        self.layer4 = self._make_layer(LFEM, planes[2], planes[3], 2, 2)
        self.layer5 = self._make_layer(Bottleneck, planes[3], planes[3], 1)

        self.layer3_ = self._make_layer(BasicBlockOriginal, planes[1], planes[1], 1)
        self.layer4_ = self._make_layer(BasicBlockOriginal, planes[1], planes[1], 1)
        self.layer5_ = self._make_layer(Bottleneck, planes[1], planes[1], 1)

        self.alignlayer3_ = AlignedModule(planes[1], planes[1] // 2)

        self.compression3 = ConvBN(planes[2], planes[1], 1)
        self.compression4 = ConvBN(planes[3], planes[1], 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.wavg1 = nn.Conv2d(planes[2], planes[1], kernel_size=1, padding=0)

        self.down3 = ConvBN(planes[1], planes[2], 3, 2, 1)
        self.down4 = Conv2BN(planes[1], planes[2], planes[3], 3, 2, 1)

        self.spp = DCPPM(planes[-1], spp_planes, planes[2])
        self.seghead_extra = SegHead(planes[1], head_planes, num_classes, 8)
        self.final_layer = SegHead(planes[2], head_planes, num_classes, 8)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            self.load_state_dict(torch.load(pretrained, map_location='cpu')['model'], strict=False)

    def _make_layer(self, block, inplanes, planes, depths, s=1) -> nn.Sequential:
        downsample = None
        if inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, 1, s, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = [block(inplanes, planes, s, downsample)]
        inplanes = planes * block.expansion

        for i in range(1, depths):
            if i == depths - 1:
                layers.append(block(inplanes, planes, no_relu=True))
            else:
                layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        H, W = x.shape[-2] // 8, x.shape[-1] // 8
        layers = []

        x = self.conv1(x)
        x = self.layer1(x)
        layers.append(x)

        x = self.layer2(F.relu(x))
        layers.append(x)

        x = self.layer3(F.relu(x))
        layers.append(x)
        x_ = self.layer3_(F.relu(layers[1]))
        x = x + self.down3(F.relu(x_))
        # AFFM融合模块
        templayer3 = F.relu(layers[2])
        # 高级语义信息生成的权重
        weight1 = self.wavg1(self.avg_pool(templayer3))
        compressionlayer3 = self.compression3(templayer3)
        x_ = weight1 * x_ + self.alignlayer3_(x_, compressionlayer3)
        if self.training:
            x_aux = self.seghead_extra(x_)
        x = self.layer4(F.relu(x))
        layers.append(x)
        x_ = self.layer4_(F.relu(x_))
        x = x + self.down4(F.relu(x_))
        x_ = self.layer5_(F.relu(x_))
        x = F.interpolate(self.spp(self.layer5(F.relu(x))), size=(H, W), mode='bilinear', align_corners=False)
        x_ = self.final_layer(x + x_)

        return (x_, x_aux) if self.training else x_


# ==================== 测试代码 ====================

if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    model = DBRNet(num_classes=5).cuda()
    print(model)
    model.train(False)
    model.eval()
    input = torch.randn(1, 3, 1600, 256).cuda()
    warm_iter = 300
    iteration = 1000
    print('=========Speed Testing=========')
    fps_time = []
    for _ in range(iteration):
        if _ < warm_iter:
            model(input)
        else:
            torch.cuda.synchronize()
            start = time.time()
            output = model(input)
            torch.cuda.synchronize()
            end = time.time()
            fps_time.append(end - start)
            print(end - start)
    time_sum = 0
    for i in fps_time:
        time_sum += i
    print("FPS: %f" % (1.0 / (time_sum / len(fps_time))))