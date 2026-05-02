"""
Generate a visual showcase of all Arête matplotlib styles and colormaps.
Outputs: arete_colormaps_demo.png, arete_styles_demo.png, arete_perceptual_demo.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import arete_mpl

OUT = "/home/claude/arete_mpl"

# =========================================================================
#  1.  Colormap gallery  (sequential + perceptual blends)
# =========================================================================
def colormap_gallery():
    seq_names = ["arete_blues", "arete_reds", "arete_greens",
                 "arete_purples", "arete_oranges"]
    blend_names = ["arete_blue2earth", "arete_blue2green",
                   "arete_blue2purple", "arete_blue2red", "arete_blue2white"]

    all_names = seq_names + blend_names
    n = len(all_names)

    fig, axes = plt.subplots(n, 1, figsize=(10, 0.55 * n + 1.2))
    fig.suptitle("Arête Colormaps", fontsize=14, fontweight="bold", y=0.98,
                 color="#031D3A")
    gradient = np.linspace(0, 1, 512).reshape(1, -1)

    for ax, name in zip(axes, all_names):
        cmap = arete_mpl.get_cmap(name)
        ax.imshow(gradient, aspect="auto", cmap=cmap)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylabel(name.replace("arete_", ""), fontsize=9,
                       rotation=0, ha="right", va="center", labelpad=90)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout(rect=[0.18, 0, 1, 0.94])

    # Divider labels positioned after layout
    ypos_seq = axes[0].get_position().y1 + 0.01
    ypos_blend = axes[5].get_position().y1 + 0.01
    fig.text(0.02, ypos_seq, "Sequential", fontsize=10,
             fontweight="bold", color="#07456C", va="bottom")
    fig.text(0.02, ypos_blend, "Perceptual blends",
             fontsize=10, fontweight="bold", color="#07456C", va="bottom")

    fig.savefig(f"{OUT}/arete_colormaps_demo.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  -> arete_colormaps_demo.png")


# =========================================================================
#  2.  Style comparison across contexts
# =========================================================================
def style_comparison():
    np.random.seed(42)
    x = np.linspace(0, 4 * np.pi, 120)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Arête Style Contexts", fontsize=16, fontweight="bold",
                 color="#031D3A", y=0.98)
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    contexts = ["paper", "notebook", "talk", "poster"]
    for idx, ctx in enumerate(contexts):
        ax = fig.add_subplot(gs[idx])
        # Temporarily apply style for visual rendering
        with plt.style.context(["arete-base", f"arete-{ctx}"]):
            # Re-get the prop_cycle colors from the style
            colors = ["#036C9A", "#AB6638", "#2A9D8F", "#C44E52", "#72A3C3", "#D4A24E"]
            for i, c in enumerate(colors[:4]):
                y = np.sin(x + i * 0.5) + np.random.normal(0, 0.08, len(x))
                ax.plot(x, y, color=c, linewidth=1.4 + idx * 0.3,
                        label=f"Series {i+1}")
            ax.set_title(f"arete-{ctx}", fontsize=12, fontweight="bold", color="#07456C")
            ax.set_xlabel("Phase")
            ax.set_ylabel("Amplitude")
            ax.legend(fontsize=7 + idx, loc="upper right", frameon=False)
            ax.tick_params(direction="out")

    fig.savefig(f"{OUT}/arete_styles_demo.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  -> arete_styles_demo.png")


# =========================================================================
#  3.  Perceptual-uniformity verification
#      Show L* (lightness) profiles for each blend cmap
# =========================================================================
def perceptual_verification():
    blend_names = ["arete_blue2earth", "arete_blue2green",
                   "arete_blue2purple", "arete_blue2red", "arete_blue2white"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 2]})
    fig.suptitle("Perceptual Uniformity Verification  (L* lightness profiles)",
                 fontsize=13, fontweight="bold", color="#031D3A")

    t = np.linspace(0, 1, 256)
    lstar_colors = ["#031D3A", "#0B3530", "#4E3175", "#8B1A1E", "#72A3C3"]

    ax_top = axes[0]
    for name, lc in zip(blend_names, lstar_colors):
        cmap = arete_mpl.get_cmap(name)
        rgb = cmap(t)[:, :3]
        lab = arete_mpl._rgb_to_lab(rgb)
        label = name.replace("arete_", "")
        ax_top.plot(t, lab[:, 0], color=lc, linewidth=2, label=label)

    ax_top.set_ylabel("L*  (CIELAB lightness)")
    ax_top.set_xlabel("Colormap position  t")
    ax_top.set_ylim(0, 105)
    ax_top.legend(fontsize=9, loc="upper left", frameon=False)
    ax_top.axhline(50, color=".7", ls="--", lw=0.8)
    ax_top.tick_params(direction="out")

    # Bottom panel: show the sequential maps L* for comparison
    seq_names = ["arete_blues", "arete_reds", "arete_greens",
                 "arete_purples", "arete_oranges"]
    seq_colors = ["#036C9A", "#C44E52", "#2A9D8F", "#7B5EA7", EARTH]
    ax_bot = axes[1]
    for name, lc in zip(seq_names, seq_colors):
        cmap = arete_mpl.get_cmap(name)
        rgb = cmap(t)[:, :3]
        lab = arete_mpl._rgb_to_lab(rgb)
        label = name.replace("arete_", "")
        ax_bot.plot(t, lab[:, 0], color=lc, linewidth=2, label=label)

    ax_bot.set_ylabel("L*")
    ax_bot.set_xlabel("Colormap position  t")
    ax_bot.set_ylim(0, 105)
    ax_bot.legend(fontsize=9, loc="upper right", frameon=False, ncol=2)
    ax_bot.axhline(50, color=".7", ls="--", lw=0.8)
    ax_bot.tick_params(direction="out")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{OUT}/arete_perceptual_demo.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  -> arete_perceptual_demo.png")


# =========================================================================
#  4.  Practical usage example  (fake glaciology data)
# =========================================================================
def usage_example():
    arete_mpl.use("notebook")

    np.random.seed(7)
    # Simulate a 2D velocity field (Gaussian peaks)
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = (np.exp(-((X - 1)**2 + Y**2)) * 3
         + np.exp(-((X + 1.5)**2 + (Y - 0.5)**2) / 0.6) * 2
         + np.random.normal(0, 0.05, X.shape))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # Panel 1: velocity with arete_blues
    im0 = axes[0].pcolormesh(X, Y, Z, cmap="arete_blues", shading="auto")
    axes[0].set_title("Surface Velocity")
    fig.colorbar(im0, ax=axes[0], label="m/yr", shrink=0.85)

    # Panel 2: temperature anomaly with arete_blue2red
    T = np.sin(X) * np.cos(Y) * 2
    im1 = axes[1].pcolormesh(X, Y, T, cmap="arete_blue2red", shading="auto")
    axes[1].set_title("Basal Temperature Anomaly")
    fig.colorbar(im1, ax=axes[1], label="°C", shrink=0.85)

    # Panel 3: bed elevation with arete_blue2earth
    B = -2 + X * 0.3 + np.sin(Y * 2) * 0.5
    im2 = axes[2].pcolormesh(X, Y, B, cmap="arete_blue2earth", shading="auto")
    axes[2].set_title("Bed Elevation")
    fig.colorbar(im2, ax=axes[2], label="m a.s.l.", shrink=0.85)

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("x  (km)")
        ax.set_ylabel("y  (km)")

    fig.tight_layout()
    fig.savefig(f"{OUT}/arete_usage_example.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  -> arete_usage_example.png")


# =========================================================================
EARTH = "#AB6638"

if __name__ == "__main__":
    print("Generating Arête matplotlib demos …")
    colormap_gallery()
    style_comparison()
    perceptual_verification()
    usage_example()
    print("Done.")
