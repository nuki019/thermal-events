"""Compact CenterNet-style detector with optional ConvLSTM (RVT-lite).

Shared across benchmark channels (frame / event-voxel / fusion) so the input
representation is the only varied factor. Recurrent variant adds a ConvLSTM
at the stride-8 feature level for event-native temporal processing.

Heads: center heatmap (focal loss), box size + local offset (L1 at positives).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(cin, cout, k=3, s=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, k // 2, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ConvLSTMCell(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(2 * c, 4 * c, 3, 1, 1)
        self.c = c

    def forward(self, x, state):
        h, c = state if state is not None else (
            torch.zeros_like(x), torch.zeros_like(x))
        g = self.conv(torch.cat([x, h], 1))
        i, f, o, gg = g.chunk(4, 1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        c = f * c + i * torch.tanh(gg)
        h = o * torch.tanh(c)
        return h, (h, c)


class Backbone(nn.Module):
    def __init__(self, cin, width=32):
        super().__init__()
        w = width
        self.s1 = nn.Sequential(conv_bn(cin, w, s=2), conv_bn(w, w))          # /2
        self.s2 = nn.Sequential(conv_bn(w, 2 * w, s=2), conv_bn(2 * w, 2 * w))  # /4
        self.s3 = nn.Sequential(conv_bn(2 * w, 4 * w, s=2), conv_bn(4 * w, 4 * w))  # /8
        self.s4 = nn.Sequential(conv_bn(4 * w, 4 * w, s=2), conv_bn(4 * w, 4 * w))  # /16

    def forward(self, x):
        a = self.s1(x)
        b = self.s2(a)
        c = self.s3(b)
        d = self.s4(c)
        return a, b, c, d


class CenterNet(nn.Module):
    """out_stride=4 CenterNet. n_classes=1 (objectness-style single class)."""

    def __init__(self, cin=1, n_classes=1, width=32, recurrent=False):
        super().__init__()
        self.backbone = Backbone(cin, width)
        self.recurrent = recurrent
        w = width
        if recurrent:
            self.lstm = ConvLSTMCell(4 * w)
        self.up3 = nn.ConvTranspose2d(4 * w, 2 * w, 2, 2)     # /16 -> /8
        self.fuse3 = conv_bn(6 * w, 2 * w)
        self.up2 = nn.ConvTranspose2d(2 * w, w, 2, 2)         # /8 -> /4
        self.fuse2 = conv_bn(3 * w, w)
        self.head_hm = nn.Sequential(conv_bn(w, w), nn.Conv2d(w, n_classes, 1))
        self.head_wh = nn.Sequential(conv_bn(w, w), nn.Conv2d(w, 2, 1))
        self.head_off = nn.Sequential(conv_bn(w, w), nn.Conv2d(w, 2, 1))
        self.head_hm[-1].bias.data.fill_(-2.19)

    def forward(self, x, state=None):
        _, f4, f8, f16 = self.backbone(x)
        if self.recurrent:
            f8, state = self.lstm(f8, state)
        u3 = self.fuse3(torch.cat([self.up3(f16), f8], 1))
        u2 = self.fuse2(torch.cat([self.up2(u3), f4], 1))
        hm = self.head_hm(u2)
        wh = self.head_wh(u2)
        off = self.head_off(u2)
        return dict(hm=hm, wh=wh, off=off), state


def gaussian_radius(det_size, min_overlap=0.7):
    h, w = det_size
    a1, b1 = 1, (h + w)
    c1 = h * w * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + (b1 ** 2 - 4 * a1 * c1) ** 0.5) / (2 * a1)
    return max(float(r1), 0.5)


def draw_gaussian(heatmap, center, radius):
    diameter = 2 * int(radius) + 1
    sigma = diameter / 6
    ax = torch.arange(diameter, device=heatmap.device) - radius
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g2 = (g[:, None] * g[None, :]).to(heatmap.dtype)
    x, y = int(center[0]), int(center[1])
    H, W = heatmap.shape
    x0, x1 = max(x - int(radius), 0), min(x + int(radius) + 1, W)
    y0, y1 = max(y - int(radius), 0), min(y + int(radius) + 1, H)
    if x0 >= x1 or y0 >= y1:
        return
    gx0, gy0 = x0 - (x - int(radius)), y0 - (y - int(radius))
    gx1, gy1 = gx0 + (x1 - x0), gy0 + (y1 - y0)
    heatmap[y0:y1, x0:x1] = torch.maximum(heatmap[y0:y1, x0:x1], g2[gy0:gy1, gx0:gx1])


def focal_loss(pred, target):
    """CornerNet focal (Objects as Points). pred logits [B,C,H,W], target heatmap."""
    p = torch.sigmoid(pred).clamp(1e-4, 1 - 1e-4)
    pos_mask = target.eq(1.0)
    neg_mask = ~pos_mask
    neg_weights = torch.pow(1 - target, 4)
    pos_loss = torch.log(p) * torch.pow(1 - p, 2) * pos_mask
    neg_loss = torch.log(1 - p) * torch.pow(p, 2) * neg_weights * neg_mask
    num_pos = pos_mask.sum()
    if num_pos == 0:
        return -neg_loss.sum()
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos


def build_targets(boxes, H, W, stride=4, max_objs=64, device='cpu'):
    """boxes: list of [cx,cy,w,h] in input pixels. Returns heatmap, wh, off, mask."""
    ho, wo = H // stride, W // stride
    hm = torch.zeros(1, ho, wo, device=device)
    wh = torch.zeros(max_objs, 2, device=device)
    off = torch.zeros(max_objs, 2, device=device)
    ind = torch.zeros(max_objs, dtype=torch.long, device=device)
    mask = torch.zeros(max_objs, device=device)
    for j, b in enumerate(boxes[:max_objs]):
        cx, cy, bw, bh = b
        cx, cy, bw, bh = cx / stride, cy / stride, bw / stride, bh / stride
        r = max(gaussian_radius((bh, bw)), 1.0)
        draw_gaussian(hm[0], (cx, cy), r)
        xi, yi = int(cx), int(cy)
        if 0 <= xi < wo and 0 <= yi < ho:
            ind[j] = yi * wo + xi
            wh[j] = torch.tensor([bw, bh], device=device)
            off[j] = torch.tensor([cx - xi, cy - yi], device=device)
            mask[j] = 1
            # ensure the exact center is a strict positive (value exactly 1)
            hm[0, yi, xi] = 1.0
    return hm, wh, off, ind, mask


def reg_l1(pred, ind, mask, target):
    """pred [B,2,H,W] gathered at ind."""
    B = pred.shape[0]
    p = pred.permute(0, 2, 3, 1).reshape(B, -1, 2)
    loss = 0.0
    for b in range(B):
        m = mask[b] > 0
        if m.any():
            loss = loss + F.l1_loss(p[b][ind[b][m]], target[b][m], reduction='sum')
    return loss / max(float(mask.sum()), 1.0)


@torch.no_grad()
def decode(out, stride=4, k=100, thresh=0.3):
    """Returns list of per-image arrays [N,5]: x0,y0,x1,y1,score."""
    hm = torch.sigmoid(out['hm'])
    pooled = F.max_pool2d(hm, 3, 1, 1)
    keep = (pooled == hm).float()
    hm = hm * keep
    B, C, H, W = hm.shape
    scores, inds = torch.topk(hm.reshape(B, -1), k, dim=1)
    wh = out['wh'].permute(0, 2, 3, 1).reshape(B, -1, 2)
    off = out['off'].permute(0, 2, 3, 1).reshape(B, -1, 2)
    results = []
    for b in range(B):
        dets = []
        for s, i in zip(scores[b].tolist(), inds[b].tolist()):
            if s < thresh:
                break
            y, x = i // W, i % W
            ow, oh = wh[b][i].tolist()
            ox, oy = off[b][i].tolist()
            cx = (x + ox) * stride
            cy = (y + oy) * stride
            w, h = ow * stride, oh * stride
            dets.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, s])
        results.append(dets)
    return results
