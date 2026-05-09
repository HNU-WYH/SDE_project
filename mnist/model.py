import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianFourier(nn.Module):
    def __init__(self, dim, scale=16.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim // 2) * scale, requires_grad=False)

    def forward(self, v):
        proj = v[:, None] * self.W[None, :] * 2 * math.pi
        return torch.cat([proj.sin(), proj.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.proj = nn.Linear(emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.proj(F.silu(emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class ScoreNet(nn.Module):
    def __init__(self, ch=32, emb_dim=128):
        super().__init__()
        self.emb = nn.Sequential(
            GaussianFourier(emb_dim),
            nn.Linear(emb_dim, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.in_conv = nn.Conv2d(1, ch, 3, padding=1)
        self.b1 = ResBlock(ch, ch, emb_dim)
        self.down1 = nn.Conv2d(ch, 2 * ch, 3, stride=2, padding=1)
        self.b2 = ResBlock(2 * ch, 2 * ch, emb_dim)
        self.down2 = nn.Conv2d(2 * ch, 4 * ch, 3, stride=2, padding=1)
        self.b3 = ResBlock(4 * ch, 4 * ch, emb_dim)
        self.b4 = ResBlock(4 * ch, 4 * ch, emb_dim)
        self.up2 = nn.ConvTranspose2d(4 * ch, 2 * ch, 4, stride=2, padding=1)
        self.b5 = ResBlock(4 * ch, 2 * ch, emb_dim)
        self.up1 = nn.ConvTranspose2d(2 * ch, ch, 4, stride=2, padding=1)
        self.b6 = ResBlock(2 * ch, ch, emb_dim)
        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv2d(ch, 1, 3, padding=1)

    def forward(self, x, sigma):
        emb = self.emb(sigma.log())
        h1 = self.b1(self.in_conv(x), emb)
        h2 = self.b2(self.down1(h1), emb)
        h3 = self.b4(self.b3(self.down2(h2), emb), emb)
        u2 = self.b5(torch.cat([self.up2(h3), h2], dim=1), emb)
        u1 = self.b6(torch.cat([self.up1(u2), h1], dim=1), emb)
        out = self.out_conv(F.silu(self.out_norm(u1)))
        return -out / sigma.view(-1, 1, 1, 1)


class Classifier(nn.Module):
    def __init__(self, num_classes=10, ch=32, emb_dim=128):
        super().__init__()
        self.emb = nn.Sequential(
            GaussianFourier(emb_dim),
            nn.Linear(emb_dim, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.in_conv = nn.Conv2d(1, ch, 3, padding=1)
        self.b1 = ResBlock(ch, ch, emb_dim)
        self.down1 = nn.Conv2d(ch, 2 * ch, 3, stride=2, padding=1)
        self.b2 = ResBlock(2 * ch, 2 * ch, emb_dim)
        self.down2 = nn.Conv2d(2 * ch, 4 * ch, 3, stride=2, padding=1)
        self.b3 = ResBlock(4 * ch, 4 * ch, emb_dim)
        self.norm = nn.GroupNorm(8, 4 * ch)
        self.fc = nn.Linear(4 * ch, num_classes)

    def forward(self, x, sigma):
        emb = self.emb(sigma.log())
        h = self.b1(self.in_conv(x), emb)
        h = self.b2(self.down1(h), emb)
        h = self.b3(self.down2(h), emb)
        return self.fc(F.silu(self.norm(h)).mean(dim=[2, 3]))
