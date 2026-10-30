#!/usr/bin/env python3
"""
digit_touch: visualize the 28x28 images produced by the board.

For each drawing, the board writes a line to the console:
    IMG:<prediction>:<784 bytes in hexadecimal>
This script extracts them from a log file and produces train/out/drawings.png:
    - top row: your drawings as the model sees them (28x28);
    - bottom row: real MNIST digits, to compare the look (size, stroke
      thickness, centering).
It also re-runs the int8 model on the PC to check that the board's
prediction is the same.

Usage: python view_digits.py <log_file>
"""

import os
import re
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quantize import make_interpreter

OUT = Path(__file__).resolve().parent / "out"
LINE = re.compile(r"IMG:(\d):([0-9a-f]{1568})")


def load_board_images(log_path):
    text = re.sub(r"\x1b\[[0-9;]*m", "", Path(log_path).read_text(errors="replace"))
    imgs, preds = [], []
    for m in LINE.finditer(text):
        preds.append(int(m.group(1)))
        imgs.append(np.frombuffer(bytes.fromhex(m.group(2)), dtype=np.uint8).reshape(28, 28))
    return imgs, preds


def pc_predict(imgs):
    interp = make_interpreter((OUT / "model_int8.tflite").read_bytes())
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    s, zp = inp["quantization"]
    res = []
    for img in imgs:
        q = np.clip(np.round(img / 255.0 / s) + zp, -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], q.reshape(1, 28, 28, 1))
        interp.invoke()
        res.append(int(interp.get_tensor(out["index"])[0].argmax()))
    return res


def stats(img):
    """Ink height/width and center of mass, to compare with MNIST."""
    ys, xs = np.nonzero(img)
    w = img.astype(np.float64)
    yy, xx = np.mgrid[0:28, 0:28]
    return (np.ptp(ys) + 1, np.ptp(xs) + 1,
            (w * xx).sum() / w.sum(), (w * yy).sum() / w.sum())


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    imgs, preds = load_board_images(sys.argv[1])
    if not imgs:
        sys.exit("No IMG: line found in the log.")

    pcs = pc_predict(imgs)
    print(f"{len(imgs)} drawings found")
    for i, (img, p, pc) in enumerate(zip(imgs, preds, pcs)):
        h, w, cx, cy = stats(img)
        print(f"  drawing {i + 1:2d}: board {p} / PC {pc} {'OK' if p == pc else 'DIFFERENT'}"
              f" | ink {h}x{w} px, center of mass ({cx:.1f}, {cy:.1f})")

    d = np.load(Path.home() / ".keras/datasets/mnist.npz")
    x_mnist, y_mnist = d["x_test"], d["y_test"]
    n = len(imgs)
    cols = max(n, 10)
    fig, axes = plt.subplots(2, cols, figsize=(cols * 1.3, 3.2))
    for ax in axes.ravel():
        ax.axis("off")
    for i, (img, p) in enumerate(zip(imgs, preds)):
        axes[0, i].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[0, i].set_title(f"read: {p}", fontsize=9)
    for k in range(min(cols, 10)):
        idx = int(np.where(y_mnist == k)[0][0])
        axes[1, k].imshow(x_mnist[idx], cmap="gray", vmin=0, vmax=255)
        axes[1, k].set_title(f"MNIST {k}", fontsize=9)
    fig.suptitle("Top: your drawings as seen by the model (28x28) — bottom: real MNIST digits")
    fig.tight_layout()
    out_path = OUT / "drawings.png"
    fig.savefig(out_path, dpi=110)
    print(f"Image written: {out_path}")


if __name__ == "__main__":
    main()
