# Detour

Cuts a sheet of eight portraits (one figure, eight expressions, 4×2 grid) into a transparent sprite strip, with no per-image settings.

```
pip install opencv-python numpy onnxruntime
python3 detour.py sheet.jpg -o sprite.webp --proof proof.png --report report.json
python3 detour.py sheet.jpg -o sprite.webp --glow 230,245,255,.55,5    # optional light halo
```

The two models (U²-Net, IS-Net, about 350 MB) are downloaded once to `~/.cache/detour`. Everything then runs offline, on the CPU, in about 20 s per sheet.

The story of the method, with its failures and fixes: [Detour, three judges for one outline](https://mauktenieb.github.io/insertkoin/notes/detour.html) · [Catchlight, lights in the eyes of eighteen Masks](https://mauktenieb.github.io/insertkoin/notes/catchlight.html).

## How it works

- **Panels found by themselves:** gutters of any colour or none, drawn frames stepped over, and a table in front of the figure cut away at its top edge.
- **Three judges:** a smooth backdrop model, removed only where it is reached from the edges, plus two neural mattes. Two against one decides.
- **What each kind of judge misses, corrected by the other:**
  - The neural mattes are trained on people. Drawn accessories they drop (wings, a cloud for a head, a receiver) come back when the backdrop model sees texture.
  - Plain, smooth backdrop shut in under an arm or between legs goes, even when every network keeps it.
- **Outline decided at pixel level:** near the edge, each pixel is compared with the local figure colour and the local backdrop colour. Painted drop shadows go, while ink lines and pixels both networks are sure of stay.
- **Smooth edges:** the outline becomes a distance field, scaled to the final size, then antialiased there. There is no staircase from enlarging, and the colours at the edge are the figure's own, never the backdrop's.
- **Consistent size:** each row is matched to the first on the silhouette of the head and shoulders. The top of the head sits on one line.
- **The program checks itself:** the proof shows every frame at full size on magenta and on white. It outlines in yellow whatever disagrees with the seven other frames, and the report counts these doubts.

## Catchlight

The second stage, after the cut: the light that lives in a character's eye, placed on every frame of the strip.

**Online: https://mauktenieb.github.io/detour/catchlight.html**, one HTML page, nothing to install.

- **The eyes found on each frame:** MediaPipe Face Mesh, run in the browser, gives the centre of each iris. Each frame is tried at three scales and mirrored before it counts as missed.
- **Shut lids stay dark:** the opening of each eye is measured against the character's usual opening; a frame with the lids closed gets no light.
- **Faces the model does not know** (a beak, a muzzle, a helmet): the face model is switched off for that character, and the lights the artist placed are followed from frame to frame by template matching.
- **One lit eye** when the character has an odd eye (dogXim): the eye nearest to where it was is kept.
- **By hand:** click to add a light, drag to move it, double-click or long-press to remove it. Colour, size, rhythm, and a live preview drawn as in Insert Koin.
- **Command line:** `python3 catchlight.py sprite.webp --frames 8 --k 4 --name auvergne -o catchlights.json` does the same work without a browser (MediaPipe Face Landmarker, downloaded once to `~/.cache/detour`). `--face-model off --ref catchlights.json` follows the points already placed; `--eyes 1` keeps one lit eye.
- **For game engines:** `python3 export.py sprite.webp --frames 8 --name auvergne --out build/` writes a Phaser / TexturePacker atlas, a Godot 4 SpriteFrames resource and a plain JSON of the frames.
- **Out:** `catchlights.json`, in sprite-cell units. Insert Koin reads it at start-up: put the file in the insertkoin repository and every Mask takes its new lights.

Tested on the 18 Masks: the face model finds every frame of the human faces and masks (Auvergne, dogXim, C7H5N3O6, Cadaver Synod…); Honey Buzzard and Croisière Noire are placed by hand and followed.

<!-- network:start -->
## Provenance

Built for [Insert Koin](https://mauktenieb.github.io/insertkoin/), the browser arcade of the Korhogo Fauna: it cut the sprites of the 18 Masks.

Other tools by Mauk Tenieb: [FaunaTor](https://github.com/MaukTenieb/faunator): Tor inside a web page · [KorhoTube](https://github.com/MaukTenieb/korhotube): a YouTube channel as JSON · [Reporhogo](https://github.com/MaukTenieb/reporhogo): one topic, forty-one code forges · [Photofauna](https://github.com/MaukTenieb/Photofauna): the pocket photo emulator · [TorView](https://github.com/MaukTenieb/torview): a minimal Tor browser, Go.

Start here: https://mauktenieb.github.io/
<!-- network:end -->

## Blind test

`blind-test.png`: four sheets Detour had never seen, made for the test, on a dark and a white background:

1. long loose red hair on a pale cream backdrop;
2. a lit candle held in hand, black tuxedo on a dark backdrop;
3. a wide figure touching the panel sides, hand-drawn panel borders, gradient backdrop;
4. a yellow oilskin and a white beard on grey.

In each, panel 7 leans on the table: the hands resting on it stay, the wood goes.

## Rights

The code (detour.py, catchlight.py, catchlight.html, export.py) is under the MIT licence: see LICENSE.

The images, and the characters, names and lore of Korhogo and the Korhogo Fauna, stay the property of Mauk Tenieb & Korhogo, all rights reserved, including for training or use by artificial-intelligence systems.

The neural models are third-party works, downloaded at first run and kept outside this repository: U²-Net (Qin et al.) and IS-Net (DIS, Qin et al.), Apache-2.0, through the rembg releases; MediaPipe Face Landmarker and Face Mesh (Google), Apache-2.0.
