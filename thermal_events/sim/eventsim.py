"""GPU event-camera simulators for thermal video conversion.

Modes:
  'v2e'    : threshold-crossing DVS model w/ per-pixel mismatch, log shot
             noise, refractory period, leak events (Hu et al. CVPRW 2021).
  'volt'   : DVS-Voltmeter-style Brownian thresholds (Lin et al. ECCV 2022).
  'poisson': uncooled Poisson-bolometer variant - inhomogeneous Poisson
             emission, rate exponential in log-intensity overdrive
             (Mousa et al. arXiv:2601.18583).

Thermal hooks (RQ3): lp_cutoff_hz = first-order low-pass on log intensity
(microbolometer tau, ADV2E-style); netd_mk used by pipeline for threshold
mapping th = log(1 + NETD / T_scene).
"""
from __future__ import annotations
import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class SimConfig:
    th_on: float = 0.2
    th_off: float = 0.2
    sigma_th: float = 0.03
    shot_noise: float = 0.01
    leak_rate_hz: float = 0.1
    refractory_us: float = 0.0
    lp_cutoff_hz: float = 0.0
    mode: str = 'v2e'
    volt_drift: float = 0.01
    poisson_r0: float = 50.0
    poisson_beta: float = 8.0
    seed: int = 0
    device: str = 'cuda'


class EventSim:
    def __init__(self, cfg: SimConfig, h: int, w: int, max_events_per_step: int = 2_000_000):
        self.cfg = cfg
        self.h, self.w = h, w
        dev = cfg.device if torch.cuda.is_available() else 'cpu'
        self.dev = torch.device(dev)
        g = torch.Generator(device='cpu').manual_seed(cfg.seed)
        self.th_on_map = (cfg.th_on * (1.0 + cfg.sigma_th * torch.randn(h, w, generator=g))).clamp(0.01).to(self.dev)
        self.th_off_map = (cfg.th_off * (1.0 + cfg.sigma_th * torch.randn(h, w, generator=g))).clamp(0.01).to(self.dev)
        self.Lref = torch.zeros(h, w, device=self.dev)
        self.Lprev = None
        self.Lsmooth = None
        self.last_t = torch.full((h, w), -1e9, device=self.dev)
        self.initialized = False
        self.max_events_per_step = max_events_per_step

    def _log(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.log(frames.clamp_min(1e-3) + 1e-3)

    def reset(self):
        self.initialized = False
        self.Lprev = None
        self.Lsmooth = None
        self.last_t.fill_(-1e9)

    @torch.no_grad()
    def run(self, frames_u8: np.ndarray, fps_in: float) -> dict:
        """frames_u8: [T,H,W] uint8 at fps_in. Returns events dict of numpy arrays."""
        cfg = self.cfg
        T, H, W = frames_u8.shape
        dt = 1.0 / fps_in
        ts, xs, ys, ps = [], [], [], []
        step = 60
        for s0 in range(0, T, step):
            self._run_block(frames_u8[s0:s0 + step], s0 * dt, dt, ts, xs, ys, ps)
        if cfg.leak_rate_hz > 0:
            n_leak = int(cfg.leak_rate_hz * self.h * self.w * T * dt)
            if n_leak > 0:
                ts.append(torch.rand(n_leak, device=self.dev) * (T * dt))
                xs.append(torch.randint(0, self.w, (n_leak,), device=self.dev))
                ys.append(torch.randint(0, self.h, (n_leak,), device=self.dev))
                ps.append((torch.randint(0, 2, (n_leak,), device=self.dev) * 2 - 1).to(torch.int8))
        if ts:
            t = torch.cat(ts).cpu().numpy().astype(np.float32)
            x = torch.cat(xs).cpu().numpy().astype(np.uint16)
            y = torch.cat(ys).cpu().numpy().astype(np.uint16)
            p = torch.cat(ps).cpu().numpy().astype(np.int8)
            order = np.argsort(t, kind='stable')
            return dict(t=t[order], x=x[order], y=y[order], p=p[order])
        return dict(t=np.empty(0, np.float32), x=np.empty(0, np.uint16),
                    y=np.empty(0, np.uint16), p=np.empty(0, np.int8))

    @torch.no_grad()
    def _run_block(self, frames_u8: np.ndarray, t_block0: float, dt: float,
                   ts, xs, ys, ps):
        cfg = self.cfg
        T = frames_u8.shape[0]
        frames = torch.from_numpy(frames_u8).to(self.dev).float() / 255.0
        L = self._log(frames)
        if cfg.lp_cutoff_hz > 0:
            a = 1.0 - float(np.exp(-2.0 * np.pi * cfg.lp_cutoff_hz * dt))
            Ls = torch.empty_like(L)
            start = 1
            if self.Lsmooth is not None:
                start = 0
            for i in range(start, T):
                prev = Ls[i - 1] if i > 0 else self.Lsmooth
                Ls[i] = prev + a * (L[i] - prev)
            if start == 1:
                Ls[0] = L[0] if self.Lsmooth is None else self.Lsmooth + a * (L[0] - self.Lsmooth)
            self.Lsmooth = Ls[-1].clone()
            L = Ls
        if cfg.shot_noise > 0:
            std = cfg.shot_noise * (1.0 + (0.5 - frames).abs())
            L = L + torch.randn_like(L) * std
        del frames
        if not self.initialized:
            self.Lref = L[0].clone()
            self.initialized = True
        if self.Lprev is None:
            self.Lprev = L[0]
            start_i = 1
        else:
            start_i = 0
        for i in range(start_i, T):
            Lnew = L[i]
            t_prev = t_block0 + (i - 1) * dt if i > 0 else t_block0 - dt
            d = Lnew - self.Lref
            if cfg.mode == 'volt':
                self.th_on_map = (self.th_on_map + cfg.volt_drift * torch.randn_like(self.th_on_map)).clamp(0.01)
                self.th_off_map = (self.th_off_map + cfg.volt_drift * torch.randn_like(self.th_off_map)).clamp(0.01)
            pos = d > 0
            n_on = torch.where(pos, torch.floor(d / self.th_on_map), torch.zeros_like(d))
            n_off = torch.where(~pos, torch.floor((-d) / self.th_off_map), torch.zeros_like(d))
            if cfg.mode == 'poisson':
                over_on = (d / self.th_on_map).clamp(0.0, 1.5)
                lam_on = (cfg.poisson_r0 * dt * torch.expm1(cfg.poisson_beta * over_on)).clamp(max=4.0)
                over_off = ((-d) / self.th_off_map).clamp(0.0, 1.5)
                lam_off = (cfg.poisson_r0 * dt * torch.expm1(cfg.poisson_beta * over_off)).clamp(max=4.0)
                n_on = torch.poisson(lam_on)
                n_off = torch.poisson(lam_off)
            if cfg.refractory_us > 0:
                ok = (t_prev - self.last_t) > cfg.refractory_us * 1e-6
                n_on = torch.where(ok, n_on, torch.zeros_like(n_on))
                n_off = torch.where(ok, n_off, torch.zeros_like(n_off))
            for nmap, pol in ((n_on, 1), (n_off, -1)):
                tot = int(nmap.sum().item())
                if tot == 0:
                    continue
                if tot > self.max_events_per_step:
                    scale = self.max_events_per_step / tot
                    nmap = torch.floor(nmap * scale)
                    tot = int(nmap.sum().item())
                    if tot == 0:
                        continue
                yy, xx = torch.nonzero(nmap > 0, as_tuple=True)
                counts = nmap[yy, xx].long()
                rep_y = torch.repeat_interleave(yy, counts)
                rep_x = torch.repeat_interleave(xx, counts)
                tot_n = rep_y.numel()
                frac = torch.rand(tot_n, device=self.dev)
                tt = t_prev + frac * dt
                ts.append(tt); xs.append(rep_x); ys.append(rep_y)
                ps.append(torch.full((tot_n,), pol, device=self.dev, dtype=torch.int8))
            fired = (n_on + n_off) > 0
            upd = n_on * self.th_on_map - n_off * self.th_off_map
            self.Lref = torch.where(fired, self.Lref + upd, self.Lref)
            if cfg.mode == 'poisson':
                self.Lref = torch.where(fired, Lnew, self.Lref)
            self.last_t = torch.where(fired, torch.full_like(self.last_t, t_prev), self.last_t)
            self.Lprev = Lnew
        del L


def voxel_grid(events: dict, h: int, w: int, bins: int, t_start: float, t_end: float) -> np.ndarray:
    """Bilinear voxel grid [bins,H,W] from events in [t_start,t_end]."""
    t, x, y, p = events['t'], events['x'], events['y'], events['p']
    m = (t >= t_start) & (t < t_end)
    t, x, y, p = t[m], x[m], y[m], p[m]
    vox = np.zeros((bins, h, w), np.float32)
    if len(t) == 0:
        return vox
    tn = (t - t_start) / max(t_end - t_start, 1e-9) * bins
    t0 = np.floor(tn).astype(int)
    t1 = t0 + 1
    w1 = (tn - t0).astype(np.float32)
    w0 = 1.0 - w1
    for ti, wi in ((t0, w0), (t1, w1)):
        ok = (ti >= 0) & (ti < bins)
        np.add.at(vox, (ti[ok], y[ok], x[ok]), (p[ok] * wi[ok]).astype(np.float32))
    return vox


def event_frame(events: dict, h: int, w: int, t_start: float, t_end: float) -> np.ndarray:
    """2-channel polarity-count frame [2,H,W] float32."""
    t, x, y, p = events['t'], events['x'], events['y'], events['p']
    m = (t >= t_start) & (t < t_end)
    fr = np.zeros((2, h, w), np.float32)
    if m.sum():
        mp = m & (p > 0); mn = m & (p < 0)
        np.add.at(fr[0], (y[mp], x[mp]), 1.0)
        np.add.at(fr[1], (y[mn], x[mn]), 1.0)
    return fr
