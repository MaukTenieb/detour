#!/usr/bin/env python3
"""Export a Detour sprite strip for game engines: Phaser / TexturePacker JSON, Godot SpriteFrames, plain JSON.

  python3 export.py sprite.webp --frames 8 --name auvergne --fps 6 --out build/
writes build/auvergne.atlas.json (Phaser 3 load.atlas, TexturePacker "JSON hash"),
       build/auvergne.tres (Godot 4 SpriteFrames, animation "default"),
       build/auvergne.frames.json (x, y, w, h of each frame).
Copyright © Mauk Tenieb & Korhogo. Code under the MIT licence of this repository.
"""
import argparse, json, os
import cv2

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('strip'); ap.add_argument('--frames', type=int, default=8); ap.add_argument('--name')
    ap.add_argument('--fps', type=float, default=6); ap.add_argument('--out', default='.')
    ap.add_argument('--res-path', help='path of the strip inside the Godot project (default res://<file name>)')
    a = ap.parse_args()
    im = cv2.imread(a.strip, cv2.IMREAD_UNCHANGED); H, W = im.shape[:2]; n = a.frames; w = W // n
    name = a.name or os.path.splitext(os.path.basename(a.strip))[0]; img = os.path.basename(a.strip)
    os.makedirs(a.out, exist_ok=True)
    rects = [{'x': i * w, 'y': 0, 'w': w, 'h': H} for i in range(n)]
    atlas = {'frames': {f'{name}_{i:02d}': {'frame': r, 'rotated': False, 'trimmed': False,
                                             'spriteSourceSize': {'x': 0, 'y': 0, 'w': w, 'h': H}, 'sourceSize': {'w': w, 'h': H}}
                        for i, r in enumerate(rects)},
             'meta': {'app': 'Detour', 'image': img, 'format': 'RGBA8888', 'size': {'w': W, 'h': H}, 'scale': '1'}}
    json.dump(atlas, open(os.path.join(a.out, name + '.atlas.json'), 'w'), indent=1)
    json.dump({'image': img, 'frames': rects, 'fps': a.fps}, open(os.path.join(a.out, name + '.frames.json'), 'w'), indent=1)
    res = a.res_path or 'res://' + img
    sub = '\n'.join(f'[sub_resource type="AtlasTexture" id="f{i}"]\natlas = ExtResource("1")\nregion = Rect2({r["x"]}, 0, {w}, {H})\n' for i, r in enumerate(rects))
    frames = ', '.join('{"duration": 1.0, "texture": SubResource("f%d")}' % i for i in range(n))
    tres = (f'[gd_resource type="SpriteFrames" load_steps={n + 2} format=3]\n\n[ext_resource type="Texture2D" path="{res}" id="1"]\n\n{sub}\n'
            f'[resource]\nanimations = [{{"frames": [{frames}], "loop": true, "name": &"default", "speed": {a.fps}}}]\n')
    open(os.path.join(a.out, name + '.tres'), 'w').write(tres)
    print('wrote', name + '.atlas.json', name + '.tres', name + '.frames.json')

if __name__ == '__main__':
    main()
