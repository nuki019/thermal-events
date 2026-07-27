"""Resume STED generation, skipping completed sequences."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from dataclasses import asdict

from thermal_events.sim.thermal_scene import ThermalScene, SceneConfig
from thermal_events.sim.eventsim import SimConfig
from thermal_events.pipeline.convert import convert_frames, save_events
from thermal_events.eval.dataset import sample_scene_cfg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data', 'sted')
SIM = SimConfig(mode='v2e')
INTERP_K = 4

rng = np.random.default_rng(12345)
index = []
if os.path.exists(os.path.join(OUT, 'index.json')):
    index = json.load(open(os.path.join(OUT, 'index.json')))

# rebuild identical rng consumption order: train 0..99 then val 0..23
cfgs = []
for split, n in (('train', 100), ('val', 24)):
    for i in range(n):
        seed = 1000 + (0 if split == 'train' else 50000) + i
        profile = 'agc' if (i % 4 == 3) else 'mixed'
        cfg = sample_scene_cfg(rng, seed, profile)
        sid = f'{split}_{i:04d}'
        cfgs.append((split, i, seed, profile, cfg, sid))

done = {e['seq'] for e in index}
for split, i, seed, profile, cfg, sid in cfgs:
    if sid in done:
        continue
    print(f'generating {sid} ...', flush=True)
    out = ThermalScene(cfg).run()
    events, meta = convert_frames(out['disp8'], cfg.fps, SIM, interp_k=INTERP_K,
                                  interp_method='linear')
    sdir = os.path.join(OUT, sid)
    os.makedirs(sdir, exist_ok=True)
    np.save(os.path.join(sdir, 'frames.npy'), out['disp8'])
    save_events(os.path.join(sdir, 'events.npz'), events, meta)
    json.dump(out['boxes'], open(os.path.join(sdir, 'boxes.json'), 'w'))
    smeta = dict(seq=sid, split=split, profile=profile,
                 scene={k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
                 night=bool(cfg.bg_mean < 18), n_events=int(len(events['t'])))
    json.dump(smeta, open(os.path.join(sdir, 'meta.json'), 'w'), indent=1)
    index.append(smeta)
    json.dump(index, open(os.path.join(OUT, 'index.json'), 'w'), indent=1)
    print(f'  {sid}: {len(events["t"])} ev', flush=True)
print('RESUME DONE', len(index))
