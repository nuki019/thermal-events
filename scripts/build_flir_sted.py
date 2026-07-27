"""Build FLIR-STED: real thermal video benchmark with events + COCO boxes.

Converts 8 continuous FLIR video sequences to event streams and packages
frames + events + annotations in the STED layout for the shared trainer.
"""
import sys, os, re, json, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from thermal_events.sim.eventsim import SimConfig
from thermal_events.pipeline.convert import convert_frames, save_events

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLIR = os.path.join(ROOT, 'data', 'flir')
OUT = os.path.join(ROOT, 'data', 'flir_sted')
FPS = 30.0

CONT_VIDEOS = ['4FRnNpmSmwktFJKjg', '5RSrbWYu9eokv5bvR', '6tLtjdkv5K5BuhB37',
               'SCiKdG3MqZfiE292B', 'ZAtDSNuZZjkZFvMAo', 'ePoikf5LyTTfqchga',
               't3f7QC8hZr6zYXpEZ', 'vbrSzr4vFTm5QwuGH']
TRAIN_VIDS = {'ZAtDSNuZZjkZFvMAo', 't3f7QC8hZr6zYXpEZ', 'ePoikf5LyTTfqchga',
              'SCiKdG3MqZfiE292B', '4FRnNpmSmwktFJKjg'}
# keep person/bike/car/motor/bus/truck; drop rare classes for a clean benchmark
KEEP_CATS = {1: 0, 2: 1, 3: 2, 4: 3, 6: 4, 8: 5}   # person,bike,car,motor,bus,truck
CAT_NAMES = ['person', 'bike', 'car', 'motor', 'bus', 'truck']


def norm14_to_u8(arr):
    """uint16 radiometric -> uint8 display via per-sequence 1-99 percentile."""
    if arr.dtype == np.uint8:
        return arr
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    return np.clip((arr.astype(np.float32) - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)


def main():
    os.makedirs(OUT, exist_ok=True)
    coco = json.load(open(os.path.join(FLIR, 'video_thermal_test_coco.json')))
    # map (video_id, frame_num) -> boxes
    im2boxes = collections.defaultdict(list)
    id2img = {im['id']: im for im in coco['images']}
    for ann in coco['annotations']:
        cid = ann['category_id']
        if cid not in KEEP_CATS:
            continue
        im = id2img[ann['image_id']]
        vid = im['extra_info']['video_id']
        m = re.search(r'frame-(\d+)-', im['file_name'])
        if not m:
            continue
        fnum = int(m.group(1))
        x, y, w, h = ann['bbox']
        im2boxes[(vid, fnum)].append(dict(cx=x + w / 2, cy=y + h / 2, w=w, h=h,
                                          cls=KEEP_CATS[cid], dT=0.0, speed=0.0,
                                          track=ann.get('track_id', -1)))
    index = []
    sim = SimConfig(mode='v2e')
    for vid in CONT_VIDEOS:
        arr = np.load(os.path.join(FLIR, f'video_{vid}.npy'))
        man = json.load(open(os.path.join(FLIR, 'video_manifest.json')))[vid]
        u8 = norm14_to_u8(arr)
        ev, meta = convert_frames(u8, FPS, sim, interp_k=4, interp_method='linear',
                                  agc=True, agc_smooth=True, chunk_s=30.0)
        split = 'train' if vid in TRAIN_VIDS else 'val'
        sid = f'flir_{vid[:8]}'
        sdir = os.path.join(OUT, sid)
        os.makedirs(sdir, exist_ok=True)
        np.save(os.path.join(sdir, 'frames.npy'), u8)
        save_events(os.path.join(sdir, 'events.npz'), ev, meta)
        boxes = []
        for i in range(arr.shape[0]):
            fnum = man['first'] + i
            boxes.append(im2boxes.get((vid, fnum), []))
        json.dump(boxes, open(os.path.join(sdir, 'boxes.json'), 'w'))
        n_boxes = sum(len(b) for b in boxes)
        smeta = dict(seq=sid, split=split, profile='real', source='FLIR_ADAS_v2',
                     video_id=vid, night=bool(np.median(u8) < 110),
                     scene=dict(duration_s=arr.shape[0] / FPS, fps=FPS,
                                width=int(arr.shape[2]), height=int(arr.shape[1])),
                     n_events=int(len(ev['t'])), n_boxes=n_boxes,
                     classes=CAT_NAMES)
        json.dump(smeta, open(os.path.join(sdir, 'meta.json'), 'w'), indent=1)
        index.append(smeta)
        print(f'{sid} [{split}]: {arr.shape[0]} frames, {len(ev["t"])} ev, {n_boxes} boxes', flush=True)
    json.dump(index, open(os.path.join(OUT, 'index.json'), 'w'), indent=1)
    print('DONE')


if __name__ == '__main__':
    main()
