import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(卷积 => BN => ReLU) * 2"""

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
    """下采样：MaxPool => DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样 + 拼接 + DoubleConv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # 修正：调整转置卷积的通道数计算方式
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # 转置卷积输出通道数应为in_channels//2，与跳跃连接的特征图通道数匹配
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 计算拼接所需的padding（处理尺寸不匹配问题）
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # 拼接特征图（跳跃连接）
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """输出卷积层，将特征映射到目标通道数"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """U-Net模型主类（修正通道数匹配问题）"""
    def __init__(self, n_channels=3, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels  # 输入通道数：3
        self.n_classes = n_classes    # 输出通道数：1
        self.bilinear = bilinear

        # 编码器部分
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)  # 修正：根据上采样方式调整通道数

        # 解码器部分（修正通道数计算）
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # 编码器前向传播
        x1 = self.inc(x)       # [batch, 64, 256, 256]
        x2 = self.down1(x1)    # [batch, 128, 128, 128]
        x3 = self.down2(x2)    # [batch, 256, 64, 64]
        x4 = self.down3(x3)    # [batch, 512, 32, 32]
        x5 = self.down4(x4)    # [batch, 1024, 16, 16] (当bilinear=False时)

        # 解码器前向传播（修正后通道数匹配）
        x = self.up1(x5, x4)   # [batch, 512, 32, 32]
        x = self.up2(x, x3)    # [batch, 256, 64, 64]
        x = self.up3(x, x2)    # [batch, 128, 128, 128]
        x = self.up4(x, x1)    # [batch, 64, 256, 256]
        logits = self.outc(x)  # [batch, 1, 256, 256]
        return logits