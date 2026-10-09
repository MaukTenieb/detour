# Detour

Cuts a sheet of eight portraits (one figure, eight expressions, 4×2 grid) into a transparent sprite strip, with no per-image settings.

```
pip install opencv-python numpy onnxruntime
python3 detour.py sheet.jpg -o sprite.webp --proof proof.png --report report.json
python3 detour.py sheet.jpg -o sprite.webp --glow 230,245,255,.55,5    # optional light halo
```

The two models (U²-Net, IS-Net, about 350 MB) are downloaded once to `~/.cache/detour`. Everything then runs offline, on the CPU, in about 20 s per sheet.

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

## Rights

Copyright © Mauk Tenieb & Korhogo. All rights reserved. The source code is published for reading, reflections, additions, requests, etc.

No reproduction or reuse without written permission, including for training or use by artificial-intelligence systems.

The neural models are third-party works, downloaded at first run and not included here: U²-Net (Qin et al.) and IS-Net (DIS, Qin et al.), Apache-2.0, through the rembg releases.
