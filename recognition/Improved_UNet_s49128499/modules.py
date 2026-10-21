"""Basic 2D U-Net Implementation"""
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
 
    def forward(self, x):
        return self.conv(self.pool(x))
 
 
class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
 
    def forward(self, x, skip):
        x = self.up(x)
        # pad if odd sizes cause a mismatch
        dh, dw = skip.size(2) - x.size(2), skip.size(3) - x.size(3)
        if dh or dw:
            x = nn.functional.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([skip, x], dim=1))

class UNet2D(nn.Module):
    def __init__(self, in_channels=1, num_classes=6, base=32):
        super().__init__()
        b = base
        self.inc = DoubleConv(in_channels, b)
        self.d1 = Down(b, b * 2)
        self.d2 = Down(b * 2, b * 4)
        self.d3 = Down(b * 4, b * 8)
        self.d4 = Down(b * 8, b * 16)
        self.u1 = Up(b * 16, b * 8)
        self.u2 = Up(b * 8, b * 4)
        self.u3 = Up(b * 4, b * 2)
        self.u4 = Up(b * 2, b)
        self.out = nn.Conv2d(b, num_classes, 1)
 
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        return self.out(x)  # logits (B, C, H, W)

def dice_per_class(logits, target, num_classes=6, eps=1e-6):
    """
    Mean Dice per class over the batch; returns tensor (num_classes,).
    Expects class index targets and not one-hot
    """
    pred = logits.argmax(1)
    scores = []
    for c in range(num_classes):
        p, t = (pred == c).float(), (target == c).float()
        inter = (p * t).sum()
        scores.append((2 * inter + eps) / (p.sum() + t.sum() + eps))
    return torch.stack(scores)