"""
Arquitetura da rede: U-Net 2D para refinamento dos mapas Bz.

Estrutura clássica encoder-decoder com skip connections e três níveis de
downsampling, usando InstanceNorm (mais estável que BatchNorm para batches
pequenos típicos deste problema).
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Dois blocos Conv2d + InstanceNorm + ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet2D(nn.Module):
    """U-Net 2D com 3 níveis (32 -> 64 -> 128 -> bottleneck 256)."""

    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        self.down1 = DoubleConv(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv1 = DoubleConv(64, 32)

        self.out_conv = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        c1 = self.down1(x)
        c2 = self.down2(self.pool1(c1))
        c3 = self.down3(self.pool2(c2))
        bn = self.bottleneck(self.pool3(c3))

        u3 = self.conv3(torch.cat([self.up3(bn), c3], dim=1))
        u2 = self.conv2(torch.cat([self.up2(u3), c2], dim=1))
        u1 = self.conv1(torch.cat([self.up1(u2), c1], dim=1))
        return self.out_conv(u1)


def crop_to_mult8(img):
    """Recorta a imagem para dimensões múltiplas de 8 (exigência dos 3 pools)."""
    ny, nx = img.shape
    return img[:(ny // 8) * 8, :(nx // 8) * 8]
