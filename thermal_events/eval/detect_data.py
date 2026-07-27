"""Datasets for the STED benchmark: frame / event-voxel / fusion channels."""
from __future__ import annotations
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

from ..sim.eventsim import voxel_grid


class SeqStore:
    """Lazy per-sequence container."""

    def __init__(self, seq_dir):
        self.dir = seq_dir
        self.meta = json.load(open(os.path.join(seq_dir, 'meta.json')))
        self._frames = None
        self._events = None
        self._boxes = None

    @property
    def frames(self):
        if self._frames is None:
            self._frames = np.load(os.path.join(self.dir, 'frames.npy'), mmap_mode='r')
        return self._frames

    @property
    def events(self):
        if self._events is None:
            z = np.load(os.path.join(self.dir, 'events.npz'))
            self._events = dict(t=z['t'], x=z['x'], y=z['y'], p=z['p'])
        return self._events

    @property
    def boxes(self):
        if self._boxes is None:
            self._boxes = json.load(open(os.path.join(self.dir, 'boxes.json')))
        return self._boxes


class StedDataset(Dataset):
    """Per-frame samples. channel: 'frame' | 'event' | 'fusion'.

    Event voxel window: [t_i - 1/fps, t_i) aligned to label frame i.
    """

    def __init__(self, root, split, channel='event', bins=5, fps=30.0,
                 size=None, train=False, max_seqs=None):
        self.channel = channel
        self.bins = bins
        self.fps = fps
        # infer resolution from first sequence meta if not given
        if size is None:
            idx0 = json.load(open(os.path.join(root, 'index.json')))
            sc = idx0[0]['scene']
            size = (int(sc.get('height', 480)), int(sc.get('width', 640)))
        self.H, self.W = size
        self.train = train
        idx = json.load(open(os.path.join(root, 'index.json')))
        seqs = [e for e in idx if e['split'] == split]
        if max_seqs:
            seqs = seqs[:max_seqs]
        self.stores = [SeqStore(os.path.join(root, e['seq'])) for e in seqs]
        self.metas = [e for e in seqs]
        self.index = []          # (seq_i, frame_i)
        for si, st in enumerate(self.stores):
            T = st.meta['scene']['duration_s'] * st.meta['scene']['fps']
            for fi in range(1, int(T)):   # skip frame 0 (no event window)
                self.index.append((si, fi))

    def __len__(self):
        return len(self.index)

    def _voxel(self, st, fi):
        t1 = fi / self.fps
        t0 = t1 - 1.0 / self.fps
        v = voxel_grid(st.events, self.H, self.W, self.bins, t0, t1)
        return np.clip(v, -3, 3) / 3.0

    def __getitem__(self, i):
        si, fi = self.index[i]
        st = self.stores[si]
        if self.channel == 'frame':
            x = st.frames[fi].astype(np.float32)[None] / 255.0
        elif self.channel == 'event':
            x = self._voxel(st, fi)
        else:  # fusion
            fr = st.frames[fi].astype(np.float32)[None] / 255.0
            x = np.concatenate([fr, self._voxel(st, fi)], 0)
        boxes = np.array([[b['cx'], b['cy'], b['w'], b['h']] for b in st.boxes[fi]],
                         dtype=np.float32).reshape(-1, 4)
        attrs = [dict(speed=b['speed'], dT=b['dT'], area=b['w'] * b['h']) for b in st.boxes[fi]]
        if self.train and np.random.random() < 0.5:
            x = x[:, :, ::-1].copy()
            boxes[:, 0] = self.W - boxes[:, 0]
        return torch.from_numpy(x), torch.from_numpy(boxes), (si, fi), attrs


def collate(batch):
    xs = torch.stack([b[0] for b in batch])
    boxes = [b[1] for b in batch]
    keys = [b[2] for b in batch]
    attrs = [b[3] for b in batch]
    return xs, boxes, keys, attrs


class StedChunkDataset(Dataset):
    """Sequence chunks (T_chunk consecutive frames) for recurrent training."""

    def __init__(self, root, split, channel='event', bins=5, fps=30.0,
                 size=None, chunk=8, train=False, max_seqs=None):
        self.channel = channel
        self.bins = bins
        self.fps = fps
        if size is None:
            idx0 = json.load(open(os.path.join(root, 'index.json')))
            sc = idx0[0]['scene']
            size = (int(sc.get('height', 480)), int(sc.get('width', 640)))
        self.H, self.W = size
        self.chunk = chunk
        self.train = train
        idx = json.load(open(os.path.join(root, 'index.json')))
        seqs = [e for e in idx if e['split'] == split]
        if max_seqs:
            seqs = seqs[:max_seqs]
        self.stores = [SeqStore(os.path.join(root, e['seq'])) for e in seqs]
        self.index = []
        for si, st in enumerate(self.stores):
            T = int(st.meta['scene']['duration_s'] * st.meta['scene']['fps'])
            for f0 in range(1, T - chunk + 1, chunk if train else 1):
                self.index.append((si, f0))

    def __len__(self):
        return len(self.index)

    def _voxel(self, st, fi):
        t1 = fi / self.fps
        t0 = t1 - 1.0 / self.fps
        v = voxel_grid(st.events, self.H, self.W, self.bins, t0, t1)
        return np.clip(v, -3, 3) / 3.0

    def _frame_x(self, st, fi):
        if self.channel == 'frame':
            return st.frames[fi].astype(np.float32)[None] / 255.0
        if self.channel == 'event':
            return self._voxel(st, fi)
        fr = st.frames[fi].astype(np.float32)[None] / 255.0
        return np.concatenate([fr, self._voxel(st, fi)], 0)

    def __getitem__(self, i):
        si, f0 = self.index[i]
        st = self.stores[si]
        flip = self.train and np.random.random() < 0.5
        xs, boxes_all = [], []
        for j in range(self.chunk):
            fi = f0 + j
            x = self._frame_x(st, fi)
            boxes = np.array([[b['cx'], b['cy'], b['w'], b['h']] for b in st.boxes[fi]],
                             dtype=np.float32).reshape(-1, 4)
            if flip:
                x = x[:, :, ::-1].copy()
                boxes[:, 0] = self.W - boxes[:, 0]
            xs.append(torch.from_numpy(x))
            boxes_all.append(torch.from_numpy(boxes))
        return xs, boxes_all, (si, f0)


def collate_chunk(batch):
    B = len(batch)
    T = len(batch[0][0])
    xs = [torch.stack([batch[b][0][t] for b in range(B)]) for t in range(T)]
    boxes = [[batch[b][1][t] for b in range(B)] for t in range(T)]
    keys = [batch[b][2] for b in range(B)]
    return xs, boxes, keys
