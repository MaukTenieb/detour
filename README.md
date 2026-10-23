# Detour

By Mauk Tenieb & Korhogo. Build guide: [MaukTenieb/method](https://github.com/MaukTenieb/method).

Detour cuts a sheet of eight portraits (one figure, eight expressions, 4×2 grid) into a transparent sprite strip. It uses one set of settings for every sheet.

```
pip install opencv-python numpy onnxruntime
python3 detour.py sheet.jpg -o sprite.webp --proof proof.png --report report.json
python3 detour.py sheet.jpg -o sprite.webp --glow 230,245,255,.55,5    # optional light halo
```

On the first run, Detour downloads the two models (U²-Net, IS-Net, about 350 MB) to `~/.cache/detour`. After that it runs offline, on the CPU, in about 20 s per sheet.

Notes on the method, its failures and its fixes: [Detour: cutting Midjourney portrait sheets into sprite strips](https://mauktenieb.github.io/insertkoin/notes/detour.html) · [Catchlight: placing eye lights on sprite frames](https://mauktenieb.github.io/insertkoin/notes/catchlight.html).

## How it works

- Detour finds the panels by itself. It handles gutters of any colour or zero gutter, steps over drawn frames, and cuts a table in front of the figure away at its top edge.
- It makes three estimates of the figure: two matting networks, and a smooth model of the backdrop that marks backdrop only where it reaches the panel edges. A pixel stays in the figure when two of the three keep it.
- The two kinds of estimate correct each other. The neural mattes come from training on people and drop drawn accessories (wings, a cloud for a head, a receiver); the backdrop model sees texture there and brings them back. In the other direction, Detour removes plain, smooth backdrop shut in under an arm or between the legs, even when both networks keep it.
- At the edge, Detour compares each pixel with the local figure colour and the local backdrop colour. It removes painted drop shadows. Ink lines stay, and so do the pixels both networks rate as sure.
- Detour turns the outline into a distance field, scales it to the final size, then antialiases it there. The edges stay smooth after enlarging, and the colours at the edge come from the figure.
- Detour matches each row to the first on the silhouette of the head and shoulders, and puts the top of the head on one line.
- The program checks itself. The proof shows every frame at full size, on magenta and on white, and outlines in yellow each area that differs from the seven other frames. The report counts these doubts.

`--glow B,G,R,strength,radius` adds a light halo behind the figure. `halo-compare.png` shows one frame on a dark and on a light background: default on the left, `--glow` on the right.

## Catchlight

Catchlight is the second stage, after the cut. It places a catchlight on each eye, frame by frame, and runs as one HTML page in the browser: https://mauktenieb.github.io/detour/catchlight.html

- MediaPipe Face Mesh, run in the browser, gives the centre of each iris. Catchlight tries each frame at three scales, and mirrored, before it counts the frame as missed.
- Catchlight measures the opening of each eye against the character's usual opening. A frame with the lids shut stays dark.
- For faces outside the face model's range (a beak, a muzzle, a helmet), the artist switches the face model off for that character and places the lights; Catchlight follows them from frame to frame by template matching.
- When a character has one odd eye (dogXim), Catchlight lights one eye and keeps the one nearest to its last position.
- By hand: click to add a light, drag to move it, double-click or long-press to remove it. The page sets colour, size, rhythm, with a live preview drawn as in Insert Koin.
- Command line: `python3 catchlight.py sprite.webp --frames 8 --k 4 --name auvergne -o catchlights.json` does the same work in a terminal. It uses MediaPipe Face Mesh, the model of the web page (`pip install mediapipe==0.10.14`); on Linux it needs the libegl1 and libgles2 packages. `--face-model off --ref catchlights.json` follows the points already in place; `--eyes 1` keeps one lit eye.
- For game engines, `python3 export.py sprite.webp --frames 8 --name auvergne --out build/` writes a Phaser / TexturePacker atlas, a Godot 4 SpriteFrames resource and a plain JSON of the frames.
- Output: `catchlights.json`, in sprite-cell units. Insert Koin reads it at start-up. Put the file in the insertkoin repository and every Mask takes its new lights.

Tests on the 18 Masks: the face model finds every frame of the human faces and masks (Auvergne, dogXim, C7H5N3O6, Cadaver Synod…). The artist places the lights of Honey Buzzard and Croisière Noire by hand, and Catchlight follows them. On each push of `catchlight.py` or `export.py`, the workflow `catchlight-test.yml` runs Catchlight on four Masks, compares its lights with the `catchlights.json` of Insert Koin (2 cell units at most) and keeps the log in the branch `ci-results`.

## Blind test

Four sheets, new to Detour and made for the test, on a dark and on a white background:

1. long loose red hair on a pale cream backdrop;
2. a lit candle held in hand, black tuxedo on a dark backdrop;
3. a wide figure touching the panel sides, hand-drawn panel borders, gradient backdrop;
4. a yellow oilskin and a white beard on grey.

In each, panel 7 leans on the table. The hands resting on it stay. The wood goes.

## Third-party models

Detour and Catchlight download the neural models at first run and keep them outside this repository. U²-Net and IS-Net (DIS), by Qin et al., come under Apache-2.0, through the rembg releases. MediaPipe Face Landmarker and Face Mesh (Google) come under Apache-2.0.

<!-- network:start -->
## Provenance

Built for [Insert Koin](https://mauktenieb.github.io/insertkoin/), Mauk Tenieb's Korhogo world, games and machines in a browser: it cuts the sprites of the 18 Masks.

Other tools by Mauk Tenieb: [FaunaTor](https://github.com/MaukTenieb/faunator): reads web and .onion pages through Tor clients that run in the page · [KorhoTube](https://github.com/MaukTenieb/korhotube): lists the public videos of a YouTube channel as JSON · [Reporhogo](https://github.com/MaukTenieb/reporhogo): searches forty-one code forges for a topic and lists the repositories it finds · [Photofauna](https://github.com/MaukTenieb/Photofauna): a photo editor with 44 looks, part of Insert Koin · [TorView](https://github.com/MaukTenieb/torview): a desktop browser in Go that routes its traffic through the official Tor daemon.

Index of Mauk Tenieb's sites: https://mauktenieb.github.io/
<!-- network:end -->

Related: [mauktenieb.github.io](https://mauktenieb.github.io/) · [MaukTenieb/method](https://github.com/MaukTenieb/method)

## Rights

Code under MIT, method under CC BY 4.0, Korhogo material reserved; credit Mauk Tenieb & Korhogo. Details: [LICENSE](LICENSE).
