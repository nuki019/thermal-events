"""Controllable synthetic thermal-infrared scene generator.

Radiometric-truth sequences with ground truth boxes; microbolometer physics
(temporal low-pass tau, NETD noise, PRNU, 14-bit quantization) and AGC
gain/offset drift + jumps ("Thermal is Always Wild", CVPR 2026).
Units: arbitrary temperature units (atu); 1 atu ~ 1 K so NETD in mK maps directly.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
import cv2


@dataclass
class SceneConfig:
    width: int = 640
    height: int = 480
    fps: float = 30.0
    duration_s: float = 10.0
    seed: int = 0
    bg_mean: float = 20.0
    bg_tex_scale: float = 2.0
    bg_tex_amp: float = 3.0
    bg_drift_rate: float = 0.02
    n_objects: Tuple[int, int] = (2, 8)
    obj_temp_contrast: Tuple[float, float] = (2.0, 15.0)
    obj_frac_hot: float = 0.7
    obj_size_px: Tuple[int, int] = (12, 80)
    obj_speed_px_s: Tuple[float, float] = (5.0, 120.0)
    tau_ms: float = 10.0
    netd_mk: float = 50.0
    prnu: float = 0.01
    agc_drift_std: float = 0.02
    agc_jump_prob_per_s: float = 0.1
    agc_jump_mag: Tuple[float, float] = (0.05, 0.25)


@dataclass
class ObjState:
    x: float; y: float; vx: float; vy: float
    w: float; h: float; dT: float; shape: str
    alive: bool = True


def _smooth_noise_field(rng, h, w, scale_px):
    ch, cw = max(2, int(h / scale_px)), max(2, int(w / scale_px))
    coarse = rng.standard_normal((ch, cw)).astype(np.float32)
    fine = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    fine /= (fine.std() + 1e-6)
    return fine


class ThermalScene:
    def __init__(self, cfg: SceneConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self._bg_phase = 0.0
        self._bg1 = _smooth_noise_field(self.rng, cfg.height, cfg.width, 8.0 * cfg.bg_tex_scale)
        self._bg2 = _smooth_noise_field(self.rng, cfg.height, cfg.width, 24.0 * cfg.bg_tex_scale)
        self._prnu_map = 1.0 + cfg.prnu * self.rng.standard_normal((cfg.height, cfg.width)).astype(np.float32)
        self._bolo_state = None
        self._agc_gain = 1.0
        self._agc_offset = 0.0
        self.objects: List[ObjState] = []
        self._spawn_objects()

    def _spawn_objects(self):
        cfg, rng = self.cfg, self.rng
        n = rng.integers(cfg.n_objects[0], cfg.n_objects[1] + 1)
        for _ in range(n):
            s = rng.uniform(cfg.obj_size_px[0], cfg.obj_size_px[1])
            if rng.random() < 0.5:
                w = s; h = s * rng.uniform(1.0, 2.5)
            else:
                w = s * rng.uniform(0.4, 1.0); h = s
            speed = rng.uniform(*cfg.obj_speed_px_s)
            ang = rng.uniform(0, 2 * np.pi)
            dT = rng.uniform(*cfg.obj_temp_contrast) * (1 if rng.random() < cfg.obj_frac_hot else -1)
            self.objects.append(ObjState(
                x=rng.uniform(0, cfg.width), y=rng.uniform(0, cfg.height),
                vx=speed * np.cos(ang), vy=speed * np.sin(ang),
                w=w, h=h, dT=dT, shape='ellipse' if rng.random() < 0.6 else 'rect'))

    def _radiometric_frame(self, t):
        cfg = self.cfg
        self._bg_phase += cfg.bg_drift_rate / cfg.fps
        ph = self._bg_phase
        frame = (cfg.bg_mean + cfg.bg_tex_amp * (np.cos(ph) * self._bg1 + np.sin(ph) * self._bg2)).astype(np.float32)
        for o in self.objects:
            if not o.alive: continue
            cx, cy = int(round(o.x)), int(round(o.y))
            mask = np.zeros_like(frame)
            if o.shape == 'ellipse':
                cv2.ellipse(mask, (cx, cy), (max(1, int(o.w / 2)), max(1, int(o.h / 2))), 0, 0, 360, 1.0, -1)
            else:
                cv2.rectangle(mask, (int(cx - o.w / 2), int(cy - o.h / 2)), (int(cx + o.w / 2), int(cy + o.h / 2)), 1.0, -1)
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1.0, o.w / 8))
            frame += o.dT * mask
        return frame

    def _step_objects(self, dt):
        cfg = self.cfg
        for o in self.objects:
            o.x += o.vx * dt; o.y += o.vy * dt
            if o.x < 0 or o.x > cfg.width:  o.vx *= -1; o.x = float(np.clip(o.x, 0, cfg.width))
            if o.y < 0 or o.y > cfg.height: o.vy *= -1; o.y = float(np.clip(o.y, 0, cfg.height))

    def _apply_sensor(self, radio):
        cfg = self.cfg
        dt_ms = 1000.0 / cfg.fps
        alpha = 1.0 - np.exp(-dt_ms / max(cfg.tau_ms, 1e-3))
        if self._bolo_state is None:
            self._bolo_state = radio.copy()
        self._bolo_state += alpha * (radio - self._bolo_state)
        out = self._bolo_state * self._prnu_map
        out += self.rng.standard_normal(out.shape).astype(np.float32) * (cfg.netd_mk / 1000.0)
        lo, hi = cfg.bg_mean - 4 * cfg.bg_tex_amp - 20, cfg.bg_mean + 4 * cfg.bg_tex_amp + 20
        q = np.clip((out - lo) / (hi - lo), 0, 1)
        return (np.round(q * 16383) / 16383.0).astype(np.float32)

    def _apply_agc(self, q14, dt):
        cfg = self.cfg
        self._agc_gain *= float(np.exp(cfg.agc_drift_std * np.sqrt(dt) * self.rng.standard_normal()))
        if self.rng.random() < cfg.agc_jump_prob_per_s * dt:
            self._agc_gain *= 1.0 + self.rng.uniform(*cfg.agc_jump_mag) * (1 if self.rng.random() < 0.5 else -1)
        self._agc_gain = float(np.clip(self._agc_gain, 0.5, 2.0))
        disp = (q14 - 0.5) * self._agc_gain + 0.5 + self._agc_offset
        return np.clip(disp * 255.0, 0, 255).astype(np.uint8)

    def boxes(self):
        return [dict(cx=o.x, cy=o.y, w=o.w, h=o.h, cls=0 if o.shape == 'rect' else 1,
                     dT=o.dT, speed=float(np.hypot(o.vx, o.vy)))
                for o in self.objects if o.alive]

    def run(self, keep_radio=False):
        cfg = self.cfg
        T = int(round(cfg.duration_s * cfg.fps))
        radio14 = np.zeros((T, cfg.height, cfg.width), np.float32) if keep_radio else None
        disp8 = np.zeros((T, cfg.height, cfg.width), np.uint8)
        all_boxes = []
        dt = 1.0 / cfg.fps
        for i in range(T):
            q14 = self._apply_sensor(self._radiometric_frame(i * dt))
            if keep_radio:
                radio14[i] = q14
            disp8[i] = self._apply_agc(q14, dt)
            all_boxes.append(self.boxes())
            self._step_objects(dt)
        return dict(radio14=radio14, disp8=disp8, boxes=all_boxes,
                    meta=dict(fps=cfg.fps, tau_ms=cfg.tau_ms, netd_mk=cfg.netd_mk))


if __name__ == '__main__':
    cfg = SceneConfig(duration_s=2.0, fps=30.0, seed=1)
    out = ThermalScene(cfg).run()
    print('radio14', out['radio14'].shape, float(out['radio14'].min()), float(out['radio14'].max()))
    print('disp8', out['disp8'].shape, int(out['disp8'].min()), int(out['disp8'].max()))
    print('boxes frame0:', out['boxes'][0])
    cv2.imwrite('scene_sample.png', out['disp8'][30])
    print('saved scene_sample.png')
