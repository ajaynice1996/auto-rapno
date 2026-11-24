"""Draw the encoder-decoder architecture of the final model (MRI + segmentation variant).

    python plot_architecture.py --out figures/architecture.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

COL = {"in": "#dbeafe", "enc": "#dcfce7", "tok": "#fef9c3", "dec": "#fee2e2", "aux": "#ede9fe", "out": "#e5e7eb", "seg": "#ffedd5"}


def box(ax, x, y, w, h, text, kind, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=COL[kind], ec="#374151", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x1, y1, x2, y2, text=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#111827", lw=1.2))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, text, ha="center", fontsize=7.5, color="#374151")


def line(ax, pts, end):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color="#111827", lw=1.2)
    arrow(ax, pts[-1][0], pts[-1][1], end[0], end[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/architecture.png")
    args = ap.parse_args()
    fig, ax = plt.subplots(figsize=(17, 7.5))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    ax.text(8.5, 7.2, "image_2_report: 3D CNN encoder + cross-attention transformer decoder (MRI + segmentation variant)",
            ha="center", fontsize=12, weight="bold")

    # inputs
    box(ax, 0.2, 4.9, 2.2, 1.0, "4 MRI modalities\nT1n, T1c, T2f, T2w\n(4, 128, 128, 128)", "in")
    box(ax, 0.2, 3.4, 2.2, 1.0, "Segmentation\nET / NET / CC / ED\none-hot (4, 128^3)", "seg")
    box(ax, 0.2, 1.9, 2.2, 1.0, "Tumor measurements\n19 features (volumes,\nextent, centroid, midline)", "seg")
    box(ax, 0.2, 0.4, 2.2, 1.0, "Modality mask\n(4) - missing MRI\nzero-filled", "in")

    # encoder
    box(ax, 3.1, 4.0, 1.2, 1.6, "concat\n8 channels", "enc")
    box(ax, 4.7, 4.0, 1.5, 1.6, "Stem\nConv3d s2\nGN + SiLU\n32 ch, 64^3", "enc")
    box(ax, 6.5, 4.0, 2.4, 1.6, "4 stages x 2 ResBlock3D\n(first block stride 2)\n32 > 64 > 128 > 192 ch\n64^3 > 4^3", "enc")
    box(ax, 9.2, 4.0, 1.5, 1.6, "1x1x1 conv\nto d=256\nadaptive pool\n4x4x4 grid\n+ pos. embed", "enc")
    ax.text(6.4, 5.85, "3D residual CNN encoder", ha="center", fontsize=9.5, weight="bold", color="#166534")

    arrow(ax, 2.4, 5.4, 3.1, 5.0)
    arrow(ax, 2.4, 3.9, 3.1, 4.7)
    arrow(ax, 4.3, 4.8, 4.7, 4.8)
    arrow(ax, 6.2, 4.8, 6.5, 4.8)
    arrow(ax, 8.9, 4.8, 9.2, 4.8)

    # tokens
    box(ax, 9.2, 2.6, 1.5, 1.0, "+ modality\nembedding\n(Linear 4>256)", "tok")
    line(ax, [(2.4, 0.9), (2.4, 1.65), (9.95, 1.65)], end=(9.95, 2.6))
    arrow(ax, 9.95, 4.0, 9.95, 3.6)
    box(ax, 11.1, 3.1, 2.0, 1.7, "Visual tokens\n64 image tokens\n+ 1 tumor token\n(MLP 19>256)\n65 x 256", "tok")
    arrow(ax, 10.7, 3.1, 11.1, 3.5)
    line(ax, [(2.4, 2.4), (11.6, 2.4)], end=(11.6, 3.1))
    ax.text(5.5, 2.5, "tumor-feature MLP token", fontsize=7.5, color="#374151")
    ax.text(5.5, 1.75, "modality mask", fontsize=7.5, color="#374151")

    # aux heads
    box(ax, 11.1, 5.4, 2.0, 1.3, "Auxiliary heads\n(mean-pooled tokens)\ntumor type, grade,\nlaterality, findings,\nlocations", "aux", 8)
    arrow(ax, 12.1, 4.8, 12.1, 5.4, "pool")

    # decoder
    box(ax, 13.8, 1.0, 3.0, 4.6, "", "dec")
    ax.text(15.3, 5.35, "Decoder (from scratch)", ha="center", fontsize=9.5, weight="bold", color="#991b1b")
    box(ax, 14.0, 4.5, 2.6, 0.55, "Token + position embedding\n(BPE vocab 4000, d=256)", "dec", 7.5)
    box(ax, 14.0, 3.2, 2.6, 1.1, "4 x [ masked self-attention\n+ CROSS-ATTENTION to\nvisual tokens + FFN 1024 ]\n8 heads, pre-LN", "dec", 7.5)
    box(ax, 14.0, 2.3, 2.6, 0.7, "LayerNorm + tied output\nlinear > softmax", "dec", 7.5)
    box(ax, 14.0, 1.2, 2.6, 0.8, "Autoregressive generation\n(up to 448 tokens)", "dec", 7.5)
    arrow(ax, 13.1, 3.8, 14.0, 3.7)
    ax.text(13.55, 3.95, "K, V", fontsize=7.5, ha="center")
    arrow(ax, 15.3, 4.5, 15.3, 4.3)
    arrow(ax, 15.3, 3.2, 15.3, 3.0)
    arrow(ax, 15.3, 2.3, 15.3, 2.0)

    # outputs
    box(ax, 11.1, 0.3, 2.3, 1.2, "Report JSON text\nfindings, impression, type,\ncharacterization, grade,\nsummary, any_other", "out", 7.5)
    arrow(ax, 14.0, 1.5, 13.4, 0.95)
    box(ax, 8.3, 0.3, 2.4, 1.2, "Schema validation\nvalid / repaired / partial /\ninvalid (raw text kept)", "out", 7.5)
    arrow(ax, 11.1, 0.9, 10.7, 0.9)
    box(ax, 13.8, 5.9, 3.0, 0.9, "Loss = LM loss (label smooth 0.1)\n+ 0.5 x aux attribute loss", "out", 7.5)
    arrow(ax, 13.1, 6.0, 13.8, 6.3)

    ax.text(0.2, 7.0, "MRI-only variant: no segmentation channels (4 input channels) and no tumor token.",
            fontsize=8.5, style="italic", color="#374151")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
