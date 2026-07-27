"""Extract FLIR ADAS v2: continuous video sequences + val annotations."""
import sys, os, re, json, zipfile, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP = os.path.join(ROOT, 'data', 'flir_adas_v2.zip')
OUT = os.path.join(ROOT, 'data', 'flir')
os.makedirs(OUT, exist_ok=True)

CONT_VIDEOS = ['4FRnNpmSmwktFJKjg', '5RSrbWYu9eokv5bvR', '6tLtjdkv5K5BuhB37',
               'SCiKdG3MqZfiE292B', 'ZAtDSNuZZjkZFvMAo', 'ePoikf5LyTTfqchga',
               't3f7QC8hZr6zYXpEZ', 'vbrSzr4vFTm5QwuGH']


def main():
    import io
    from PIL import Image
    z = zipfile.ZipFile(ZIP)
    names = z.namelist()
    byvideo = collections.defaultdict(list)
    for n in names:
        m = re.search(r'video-([A-Za-z0-9]+)-frame-(\d+)-[^/]+\.tiff$', n)
        if m and m.group(1) in CONT_VIDEOS:
            byvideo[m.group(1)].append((int(m.group(2)), n))
    manifest = {}
    for vid, lst in byvideo.items():
        lst.sort()
        frames = []
        for _, n in lst:
            data = z.read(n)
            im = Image.open(io.BytesIO(data))
            frames.append(np.array(im))
        arr = np.stack(frames)
        print(f'{vid}: {arr.shape} dtype={arr.dtype} range=[{arr.min()},{arr.max()}]')
        np.save(os.path.join(OUT, f'video_{vid}.npy'), arr)
        manifest[vid] = dict(n_frames=int(arr.shape[0]), shape=list(arr.shape[1:]),
                             dtype=str(arr.dtype),
                             first=int(lst[0][0]), last=int(lst[-1][0]))
    json.dump(manifest, open(os.path.join(OUT, 'video_manifest.json'), 'w'), indent=1)
    # annotations
    for jf in ['FLIR_ADAS_v2/video_thermal_test/coco.json',
               'FLIR_ADAS_v2/images_thermal_val/coco.json',
               'FLIR_ADAS_v2/images_thermal_train/coco.json']:
        try:
            data = z.read(jf)
            out_name = jf.split('/')[-2] + '_coco.json'
            open(os.path.join(OUT, out_name), 'wb').write(data)
            print('saved', out_name, len(data), 'bytes')
        except KeyError:
            print('missing', jf)
    print('DONE')


if __name__ == '__main__':
    main()
