#!/usr/bin/env python3
"""Catchlight: places a light on the eyes of a drawn character, frame by frame, on a sprite strip.

  pip install mediapipe==0.10.14 opencv-python numpy   (0.10.14 carries Face Mesh, the model of the web page)
  python3 catchlight.py sprite.webp --frames 8 --k 4 --name auvergne -o catchlights.json
  python3 catchlight.py sprite.webp --frames 8 --k 4 --name croisiere_noire --face-model off --ref catchlights.json -o catchlights.json

MediaPipe Face Landmarker gives the centre of each iris (points 468 and 473) and the eyelids. Each frame goes
to the model on a grey margin, at three scales and mirrored. A frame turns dark when the mean opening of its two
eyes falls under 55 % of the character's median (floor 0.10). With the face model off (beaks, muzzles) or on frames
the model misses, a template follows the eyes from the nearest frame that has them. Output: the entry format of
catchlights.json read by Insert Koin, in cell units (strip pixels divided by --k).
Copyright © Mauk Tenieb & Korhogo. Code under the MIT licence of this repository.
"""
import argparse, json, os, sys, urllib.request
import numpy as np, cv2

MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
CACHE = os.path.join(os.path.expanduser('~'), '.cache', 'detour')
EYES = [dict(iris=468, up=159, lo=145, a=33, b=133), dict(iris=473, up=386, lo=374, a=362, b=263)]

def model_path():
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, 'face_landmarker.task')
    if not os.path.exists(p):
        print('downloading the face model (3.6 MB) to', p, file=sys.stderr)
        urllib.request.urlretrieve(MODEL_URL, p)
    return p

def landmarker():
    """MediaPipe Face Mesh, the same model as catchlight.html (refined iris points, bundled with mediapipe <= 0.10.14);
    the newer Face Landmarker task as a fallback"""
    import mediapipe as mp
    if hasattr(mp, 'solutions'):
        fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=.3)
        def f(rgb):
            r = fm.process(np.ascontiguousarray(rgb)).multi_face_landmarks
            return [r[0].landmark] if r else None
        f.close = fm.close
        return f
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    opts = vision.FaceLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=model_path()), num_faces=1,
                                        min_face_detection_confidence=.15, min_face_presence_confidence=.15)
    lm = vision.FaceLandmarker.create_from_options(opts)
    f = lambda rgb: lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))).face_landmarks
    f.close = lm.close
    return f

def load(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None: sys.exit('cannot read ' + path)
    if im.ndim == 2: im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
    if im.shape[2] == 3: im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
    return im

def on_grey(cell):
    a = cell[:, :, 3:4].astype(np.float32) / 255
    return (cell[:, :, :3].astype(np.float32) * a + 128 * (1 - a)).astype(np.uint8)

def face(detect, cell):
    """iris centres and eye openings for one frame, in cell pixels, or None"""
    rgb = cv2.cvtColor(on_grey(cell), cv2.COLOR_BGR2RGB); H, W = rgb.shape[:2]; pad = 100
    for z in (1.6, 2.4, 1.1, 3.2, 0.8):
        big = cv2.resize(rgb, (int(W * z), int(H * z)), interpolation=cv2.INTER_CUBIC)
        canvas = np.full((big.shape[0] + 2 * pad, big.shape[1] + 2 * pad, 3), 128, np.uint8)
        canvas[pad:pad + big.shape[0], pad:pad + big.shape[1]] = big
        for flip in (False, True):
            img = canvas[:, ::-1] if flip else canvas
            res = detect(img)
            if not res: continue
            L = res[0]; cw, ch = img.shape[1], img.shape[0]
            def P(i):
                x = L[i].x * cw
                if flip: x = cw - x
                return np.array([(x - pad) / z, (L[i].y * ch - pad) / z])
            out = []
            for e in EYES:
                gap = np.linalg.norm(P(e['up']) - P(e['lo'])); width = max(1e-3, np.linalg.norm(P(e['a']) - P(e['b'])))
                out.append({'p': P(e['iris']), 'open': gap / width})
            return out
    return None

def follow(grey, w, fa, p, fb):
    """where the 11x11 patch around p in frame fa lands in frame fb (normalised cross-correlation), or None"""
    R, S = 5, max(8, int(round(w * .12))); H = grey.shape[0]
    ax, ay = int(round(fa * w + p[0])), int(round(p[1]))
    if not (R <= ax < grey.shape[1] - R and R <= ay < H - R): return None
    tpl = grey[ay - R:ay + R + 1, ax - R:ax + R + 1]
    x0, x1 = max(fb * w, int(fb * w + p[0]) - S - R), min((fb + 1) * w, int(fb * w + p[0]) + S + R + 1)
    y0, y1 = max(0, ay - S - R), min(H, ay + S + R + 1)
    win = grey[y0:y1, x0:x1]
    if win.shape[0] <= 2 * R or win.shape[1] <= 2 * R: return None
    m = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED); _, best, _, loc = cv2.minMaxLoc(m)
    if best < .7: return None
    return np.array([x0 + loc[0] + R - fb * w, y0 + loc[1] + R])

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('strip'); ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--frames', type=int, default=8); ap.add_argument('--k', type=float, default=1, help='strip pixels per cell unit (Insert Koin sprites: 4)')
    ap.add_argument('--name', help='entry name in the JSON (default: file name)'); ap.add_argument('--eyes', type=int, choices=(1, 2), default=2)
    ap.add_argument('--face-model', choices=('on', 'off'), default='on'); ap.add_argument('--ref', help='existing catchlights.json: its points guide tracking and the one-eye choice')
    ap.add_argument('--color', default='#ffffff'); ap.add_argument('--kind', choices=('pulse', 'eyes'), default='pulse')
    ap.add_argument('--r', type=float, default=2); ap.add_argument('--sp', type=float, default=1)
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()
    name = a.name or os.path.splitext(os.path.basename(a.strip))[0].replace('sprite-', '')
    im = load(a.strip); n = a.frames; w = im.shape[1] // n
    db = json.load(open(a.ref)) if a.ref and os.path.exists(a.ref) else {'version': 1, 'masks': {}}
    prev = [[np.array(p) * a.k for p in f] for f in (db['masks'].get(name, {}).get('frames') or [[] for _ in range(n)])]
    prev += [[] for _ in range(n - len(prev))]
    frames, src = [list(f) for f in prev], ['kept' if f else 'dark' for f in prev]
    if a.face_model == 'on':
        detect = landmarker(); R = [face(detect, im[:, f * w:(f + 1) * w]) for f in range(n)]; detect.close()
        if a.verbose: print('openings', [None if r is None else [round(e['open'], 3) for e in r] for r in R], file=sys.stderr)
        if sum(r is not None for r in R) >= n / 2:
            mean = lambda r: sum(e['open'] for e in r) / len(r)
            med = float(np.median([mean(r) for r in R if r])); lim = max(.1, .55 * med)
            for f, r in enumerate(R):
                if r is None: frames[f], src[f] = [], 'miss'; continue
                if mean(r) < lim: frames[f], src[f] = [], 'dark'; continue
                P = [e['p'] for e in r if e['open'] >= .08]
                if a.eyes == 1 and len(P) > 1:
                    ref = (prev[f] or [p for q in prev for p in q] or [P[0]])[0]
                    P = [min(P, key=lambda p: np.linalg.norm(p - ref))]
                frames[f], src[f] = P, 'face'
        else:
            print('face model unsure on this character: points kept and followed', file=sys.stderr)
    grey = cv2.cvtColor(on_grey(im), cv2.COLOR_BGR2GRAY)
    refs = [f for f in range(n) if frames[f] and src[f] != 'track']
    for f in range(n):
        if frames[f] or src[f] == 'dark' or not refs: continue
        r = min(refs, key=lambda x: abs(x - f))
        P = [q for q in (follow(grey, w, r, p, f) for p in frames[r]) if q is not None]
        if P: frames[f], src[f] = P, 'track'
    entry = dict(db['masks'].get(name, {}))
    entry.update({'name': entry.get('name', name), 'kind': entry.get('kind', a.kind), 'color': entry.get('color', a.color),
                  'r': entry.get('r', a.r), 'sp': entry.get('sp', a.sp), 'eyes': a.eyes, 'model': a.face_model == 'on',
                  'sheet': entry.get('sheet', {'src': os.path.basename(a.strip), 'w': round(w / a.k), 'h': round(im.shape[0] / a.k), 'k': a.k, 'n': n}),
                  'frames': [[[round(float(p[0]) / a.k, 1), round(float(p[1]) / a.k, 1)] for p in f] for f in frames]})
    db['masks'][name] = entry
    json.dump(db, open(a.out, 'w'), ensure_ascii=False, indent=1)
    print(name, ' '.join(src))

if __name__ == '__main__':
    main()
