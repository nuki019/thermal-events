"""Attach event streams to STED frames (GPU stage, run after gen_sted_frames)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from thermal_events.sim.eventsim import SimConfig
from thermal_events.pipeline.convert import convert_frames, save_events

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES = os.path.join(ROOT, 'data', 'sted_frames')
OUT = os.path.join(ROOT, 'data', 'sted')
SIM = SimConfig(mode='v2e')

def main():
    index = json.load(open(os.path.join(FRAMES, 'index.json')))
    out_index = []
    if os.path.exists(os.path.join(OUT, 'index.json')):
        out_index = json.load(open(os.path.join(OUT, 'index.json')))
    done = {e['seq'] for e in out_index}
    for e in index:
        sid = e['seq']
        if sid in done:
            continue
        import gc; gc.collect()
        frames = np.load(os.path.join(FRAMES, sid, 'frames.npy'))
        ev, meta = convert_frames(frames, e['scene']['fps'], SIM, interp_k=4,
                                  interp_method='linear')
        sdir = os.path.join(OUT, sid)
        os.makedirs(sdir, exist_ok=True)
        np.save(os.path.join(sdir, 'frames.npy'), frames)
        save_events(os.path.join(sdir, 'events.npz'), ev, meta)
        import shutil
        shutil.copy(os.path.join(FRAMES, sid, 'boxes.json'), os.path.join(sdir, 'boxes.json'))
        smeta = dict(e, n_events=int(len(ev['t'])))
        json.dump(smeta, open(os.path.join(sdir, 'meta.json'), 'w'), indent=1)
        out_index.append(smeta)
        json.dump(out_index, open(os.path.join(OUT, 'index.json'), 'w'), indent=1)
        print(f'{sid}: {len(ev["t"])} ev', flush=True)
    print('EVENTS_ATTACHED', len(out_index))

if __name__ == '__main__':
    main()
