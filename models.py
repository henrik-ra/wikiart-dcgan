"""DCGAN generator and discriminator, as in the PyTorch DCGAN tutorial:
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

With the default arguments both networks are exactly the tutorial's. The optional
arguments are only used by the experiments in train.py:
  n_classes  conditional DCGAN: G gets a label embedding next to z, D an extra
             input channel with a learned 64x64 map of the label
  norm       "none" drops BatchNorm in D (spectral norm, WGAN-GP)
  sn         spectral normalization of every conv in D (Miyato et al., 2018)
  sigmoid    False returns the raw score (hinge loss, WGAN-GP)
  image_size 128 adds one layer to each network (G starts at 4x4x(ngf*16), D ends
             at 4x4x(ndf*16)), as in the art-DCGAN fork; 64 is the tutorial
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        # with spectral norm the trainable weight is weight_orig
        w = m.weight_orig if hasattr(m, "weight_orig") else m.weight
        nn.init.normal_(w.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3, n_classes=0, emb_dim=50, image_size=64):
        super().__init__()
        assert image_size in (64, 128)
        self.emb = nn.Embedding(n_classes, emb_dim) if n_classes else None
        nin = nz + (emb_dim if n_classes else 0)
        if image_size == 64:
            first = [
                # input is Z, going into a convolution
                nn.ConvTranspose2d(nin, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
            ]
        else:  # 128x128: one more layer, 4x4x(ngf*16) -> 8x8x(ngf*8), all later sizes doubled
            first = [
                nn.ConvTranspose2d(nin, ngf * 16, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 16),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
            ]
        self.main = nn.Sequential(
            *first,
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            # state size. (nc) x 64 x 64
        )

    def forward(self, input, labels=None):
        if self.emb is not None:
            input = torch.cat([input, self.emb(labels)[:, :, None, None]], 1)
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3, n_classes=0, norm="bn", sn=False, sigmoid=True, image_size=64):
        super().__init__()
        assert image_size in (64, 128)
        self.image_size = image_size
        self.emb = nn.Embedding(n_classes, image_size * image_size) if n_classes else None
        nin = nc + (1 if n_classes else 0)

        def conv(cin, cout, *a):
            c = nn.Conv2d(cin, cout, *a, bias=False)
            return nn.utils.spectral_norm(c) if sn else c

        def bn(c):
            return [nn.BatchNorm2d(c)] if norm == "bn" else []

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            conv(nin, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            conv(ndf, ndf * 2, 4, 2, 1),
            *bn(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            conv(ndf * 2, ndf * 4, 4, 2, 1),
            *bn(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            conv(ndf * 4, ndf * 8, 4, 2, 1),
            *bn(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            *([] if image_size == 64 else [  # 128x128: one more layer down to 4x4x(ndf*16)
                conv(ndf * 8, ndf * 16, 4, 2, 1), *bn(ndf * 16), nn.LeakyReLU(0.2, inplace=True)]),
            conv(ndf * (8 if image_size == 64 else 16), 1, 4, 1, 0),
            *([nn.Sigmoid()] if sigmoid else []),
        )

    def forward(self, input, labels=None):
        if self.emb is not None:
            input = torch.cat([input, self.emb(labels).view(-1, 1, self.image_size, self.image_size)], 1)
        return self.main(input)


def diff_augment(x):
    """DiffAugment (Zhao et al., 2020) with the policy color,translation,cutout.

    Differentiable, so it is applied to real and fake images before every D call,
    including the generator step. x: images in [-1, 1], N x C x H x W.
    """
    b, dev = x.size(0), x.device
    # color: brightness, saturation, contrast
    x = x + (torch.rand(b, 1, 1, 1, device=dev) - 0.5)
    m = x.mean(1, keepdim=True)
    x = (x - m) * (torch.rand(b, 1, 1, 1, device=dev) * 2) + m
    m = x.mean([1, 2, 3], keepdim=True)
    x = (x - m) * (torch.rand(b, 1, 1, 1, device=dev) + 0.5) + m
    # translation by up to 1/8 of the image size, zero padding
    shift = torch.empty(b, 2, device=dev).uniform_(-0.25, 0.25)  # 1/8 of the size = 0.25 in [-1, 1] coords
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0] = theta[:, 1, 1] = 1
    theta[:, :, 2] = shift
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    # cutout: one square of half the image size set to 0
    h, w = x.shape[2:]
    ch, cw = h // 2, w // 2
    cy = torch.randint(0, h, (b, 1, 1), device=dev)  # center of the square, may be cut at the border
    cx = torch.randint(0, w, (b, 1, 1), device=dev)
    yy = torch.arange(h, device=dev).view(1, h, 1)
    xx = torch.arange(w, device=dev).view(1, 1, w)
    inside = ((yy - cy).abs() < ch // 2) & ((xx - cx).abs() < cw // 2)
    return x * (~inside).unsqueeze(1).float()


# ---------------------------------------------------------------------------
# ResNet GAN with self-attention, as in SAGAN (Zhang et al., 2019): residual blocks
# from SNGAN (Miyato et al., 2018), spectral norm in G and D, one self-attention
# layer at 32x32 in both networks, hinge loss. Channel widths are chosen so that
# the networks are about as large as the tutorial's DCGAN (~3-4M parameters).

def _sn(m):
    return nn.utils.spectral_norm(m)


class SelfAttention(nn.Module):
    """SAGAN self-attention; keys and values are max-pooled 2x2 as in BigGAN/SAGAN code."""

    def __init__(self, c):
        super().__init__()
        self.q = _sn(nn.Conv2d(c, c // 8, 1, bias=False))
        self.k = _sn(nn.Conv2d(c, c // 8, 1, bias=False))
        self.v = _sn(nn.Conv2d(c, c // 2, 1, bias=False))
        self.o = _sn(nn.Conv2d(c // 2, c, 1, bias=False))
        self.gamma = nn.Parameter(torch.zeros(1))  # starts as identity

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.q(x).flatten(2)                     # B x C/8 x N
        k = F.max_pool2d(self.k(x), 2).flatten(2)    # B x C/8 x N/4
        v = F.max_pool2d(self.v(x), 2).flatten(2)    # B x C/2 x N/4
        attn = torch.softmax(q.transpose(1, 2) @ k, -1)  # B x N x N/4
        out = (v @ attn.transpose(1, 2)).view(b, c // 2, h, w)
        return x + self.gamma * self.o(out)


class GBlock(nn.Module):
    # BN -> ReLU -> 2x upsample -> conv3x3 -> BN -> ReLU -> conv3x3, plus upsampled 1x1 shortcut
    def __init__(self, cin, cout):
        super().__init__()
        self.bn1, self.bn2 = nn.BatchNorm2d(cin), nn.BatchNorm2d(cout)
        self.c1 = _sn(nn.Conv2d(cin, cout, 3, 1, 1))
        self.c2 = _sn(nn.Conv2d(cout, cout, 3, 1, 1))
        self.sc = _sn(nn.Conv2d(cin, cout, 1))

    def forward(self, x):
        h = F.interpolate(F.relu(self.bn1(x)), scale_factor=2)
        h = self.c2(F.relu(self.bn2(self.c1(h))))
        return h + self.sc(F.interpolate(x, scale_factor=2))


class DBlock(nn.Module):
    # (ReLU) -> conv3x3 -> ReLU -> conv3x3 -> 2x avg pool, plus 1x1 shortcut
    def __init__(self, cin, cout, first=False):
        super().__init__()
        self.first = first
        self.c1 = _sn(nn.Conv2d(cin, cout, 3, 1, 1))
        self.c2 = _sn(nn.Conv2d(cout, cout, 3, 1, 1))
        self.sc = _sn(nn.Conv2d(cin, cout, 1))

    def forward(self, x):
        h = x if self.first else F.relu(x)
        h = F.avg_pool2d(self.c2(F.relu(self.c1(h))), 2)
        s = self.sc(F.avg_pool2d(x, 2)) if self.first else F.avg_pool2d(self.sc(x), 2)
        return h + s


class ResGenerator(nn.Module):
    # z -> linear -> 4x4x(8ch) -> 8x8x(4ch) -> 16x16x(2ch) -> 32x32x(ch) -> attention -> 64x64x(ch) -> RGB
    def __init__(self, nz=100, ch=64, nc=3):
        super().__init__()
        self.ch = ch
        self.lin = _sn(nn.Linear(nz, 4 * 4 * ch * 8))
        self.blocks = nn.Sequential(GBlock(ch * 8, ch * 4), GBlock(ch * 4, ch * 2), GBlock(ch * 2, ch),
                                    SelfAttention(ch), GBlock(ch, ch))
        self.out = nn.Sequential(nn.BatchNorm2d(ch), nn.ReLU(True), _sn(nn.Conv2d(ch, nc, 3, 1, 1)), nn.Tanh())
        _xavier(self)

    def forward(self, input, labels=None):
        h = self.lin(input.flatten(1)).view(-1, self.ch * 8, 4, 4)
        return self.out(self.blocks(h))


class ResDiscriminator(nn.Module):
    # 64x64 -> 32x32x(ch) -> attention -> 16x16x(2ch) -> 8x8x(4ch) -> 4x4x(8ch) -> ReLU, sum pool -> linear
    def __init__(self, ch=64, nc=3):
        super().__init__()
        self.blocks = nn.Sequential(DBlock(nc, ch, first=True), SelfAttention(ch), DBlock(ch, ch * 2),
                                    DBlock(ch * 2, ch * 4), DBlock(ch * 4, ch * 8))
        self.lin = _sn(nn.Linear(ch * 8, 1))
        _xavier(self)

    def forward(self, input, labels=None):
        return self.lin(F.relu(self.blocks(input)).sum([2, 3]))  # raw score, for the hinge loss


def _xavier(net):
    # SNGAN/SAGAN initialize convs and linears with Xavier (uniform), not N(0, 0.02)
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w = m.weight_orig if hasattr(m, "weight_orig") else m.weight
            nn.init.xavier_uniform_(w.data)
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)
