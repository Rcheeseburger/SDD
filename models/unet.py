"""
Complete Classical U-Net Implementation
Based on the original paper: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== U-Net Components ====================

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ==================== Complete U-Net Model ====================

class UNet(nn.Module):
    """
    Classical U-Net Architecture

    Args:
        n_channels (int): Number of input channels
        n_classes (int): Number of output classes
        bilinear (bool): Whether to use bilinear upsampling (default: False)
                        False: Use transposed convolution (more parameters, classical)
                        True: Use bilinear interpolation (fewer parameters, faster)
    """

    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder (Contracting Path)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Decoder (Expansive Path)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # Output layer
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder path with skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output
        logits = self.outc(x)
        return logits


# ==================== Helper Functions ====================

def count_parameters(model):
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==================== Usage Examples ====================

if __name__ == "__main__":
    # Test the model
    print("=== Classical U-Net Test ===")

    # Create models
    model_conv = UNet(n_channels=3, n_classes=21, bilinear=False)  # Classical with ConvTranspose
    model_bilinear = UNet(n_channels=3, n_classes=21, bilinear=True)  # Lightweight with bilinear

    # Test forward pass
    model_conv.eval()
    model_bilinear.eval()

    test_image = torch.randn(1, 3, 512, 512)

    with torch.no_grad():
        output_conv = model_conv(test_image)
        output_bilinear = model_bilinear(test_image)

    print(f"Input shape: {test_image.shape}")
    print(f"Output shape (ConvTranspose): {output_conv.shape}")
    print(f"Output shape (Bilinear): {output_bilinear.shape}")

    # Parameter comparison
    params_conv = count_parameters(model_conv)
    params_bilinear = count_parameters(model_bilinear)

    print(f"\n=== Parameter Comparison ===")
    print(f"ConvTranspose U-Net: {params_conv:,} parameters")
    print(f"Bilinear U-Net: {params_bilinear:,} parameters")
    print(f"Parameter reduction: {(params_conv - params_bilinear) / params_conv * 100:.1f}%")

    # Architecture summary
    print(f"\n=== Architecture Summary ===")
    print(f"Input channels: {model_conv.n_channels}")
    print(f"Output classes: {model_conv.n_classes}")
    print(f"Bilinear upsampling: {model_conv.bilinear}")
    print(f"Expected output size: [batch_size, {model_conv.n_classes}, height, width]")

    # Memory usage estimation (rough)
    input_size_mb = test_image.numel() * 4 / 1024 / 1024  # assuming float32
    output_size_mb = output_conv.numel() * 4 / 1024 / 1024
    print(f"\n=== Memory Usage (approx.) ===")
    print(f"Input tensor: {input_size_mb:.2f} MB")
    print(f"Output tensor: {output_size_mb:.2f} MB")