#!/usr/bin/env python3
"""Recreate the two-panel figure using model probability densities.

Dependencies: numpy, matplotlib
Usage:
    python plot_sea_level_two_panel.py distributions.csv --output figure.png
    python plot_sea_level_two_panel.py distributions.csv --output figure.pdf \
        --lower-label '2°C warming' --higher-label '4°C warming' \
        --reference-label '3°C warming' --baseline 2000

CSV header (followed by actual model data, no synthetic data supplied):
sea_level_m,lower_warming_stable,higher_warming_stable,reference_warming_stable,reference_warming_unstable

All four density columns must be probability densities in m^-1, on the same
strictly increasing sea-level grid, with the same year and sea-level baseline.
Panel A holds Thwaites stable and varies warming. Panel B holds warming fixed
and varies Thwaites stability. Do not use blended scenarios for these inputs.

Supply the full support, including the unstable distribution's long tail.
Each curve must integrate to one within 1%; densities are never silently
renormalized. Probability mass per bin is NOT a density: for histogram output,
divide each bin probability by its width and evaluate/export an appropriate
common-grid density before using this script. Percentages must first be
converted to fractions. No KDE, curve fitting, or smoothing is performed.

The same function can be called directly from your model with NumPy arrays:
    fig = make_figure(x_m, low_stable, high_stable, ref_stable, ref_unstable)
    fig.savefig('figure.pdf', bbox_inches='tight')
"""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

COLUMNS = ('lower_warming_stable', 'higher_warming_stable',
           'reference_warming_stable', 'reference_warming_unstable')
INK, MUTED = '#173646', '#5D7280'
BLUE, SLATE, RED = '#197E9D', '#8196A4', '#C4483F'


def validate(x, curves):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 3 or not np.all(np.isfinite(x)):
        raise ValueError('Sea-level grid must contain at least 3 finite values.')
    if np.any(np.diff(x) <= 0):
        raise ValueError('sea_level_m must be strictly increasing.')
    checked = []
    for name, values in zip(COLUMNS, curves):
        p = np.asarray(values, dtype=float)
        if p.shape != x.shape or not np.all(np.isfinite(p)) or np.any(p < 0):
            raise ValueError(f'{name}: require finite, nonnegative densities matching x.')
        area = np.sum((p[:-1] + p[1:]) * np.diff(x) / 2)
        if abs(area - 1) > 0.01:
            raise ValueError(f'{name}: integral is {area:.6g}, expected 1 ± 0.01. '
                             'Check density units, grid resolution, and tail coverage.')
        checked.append(p)
    return x, checked


def quantile(x, p, q):
    """Numerical quantile, used only to position arrows and tail shading."""
    cdf = np.r_[0, np.cumsum((p[:-1] + p[1:]) * np.diff(x) / 2)]
    # Normalize the numerical CDF only; plotted densities remain unchanged.
    return float(np.interp(q, cdf / cdf[-1], x))


def make_figure(x, low_stable, high_stable, ref_stable, ref_unstable,
                lower_label='Lower warming', higher_label='Higher warming',
                reference_label='Same warming trajectory', baseline='2000'):
    x, curves = validate(x, [low_stable, high_stable, ref_stable, ref_unstable])
    low, high, stable, unstable = curves
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'text.color': INK, 'axes.labelcolor': MUTED,
                         'xtick.color': MUTED, 'ytick.color': MUTED})
    fig = plt.figure(figsize=(15, 9), facecolor='white')
    fig.text(.06, .95, 'SPECTRUM  /  TWO CONTROLS ON SEA-LEVEL RISK',
             fontsize=10, weight='bold', color=MUTED)
    fig.text(.06, .878, 'What emissions cuts change—and what they leave uncertain',
             fontsize=23, weight='bold')
    fig.text(.06, .834, f'Global sea-level rise by 2100  •  Relative to {baseline}',
             fontsize=12, color=MUTED)
    fig.text(.06, .756, 'A   Emissions shift the likely outcome', fontsize=16, weight='bold')
    fig.text(.06, .719, 'Hold Thwaites stable; vary future warming.', color=MUTED)
    fig.text(.55, .756, 'B   Instability extends the upper tail', fontsize=16, weight='bold')
    fig.text(.55, .719, f'Hold warming fixed ({reference_label}); vary Thwaites stability.',
             fontsize=10.5, color=MUTED)
    axes = [fig.add_axes([.075, .355, .385, .29]),
            fig.add_axes([.565, .355, .385, .29])]
    ymax = max(p.max() for p in curves) * 1.30
    for ax in axes:
        ax.set(xlim=(x[0], x[-1]), ylim=(0, ymax))
        ax.spines[['top', 'right']].set_visible(False)
        for side in ['left', 'bottom']:
            ax.spines[side].set_color('#BAC8CF')
        ax.tick_params(length=3, pad=7, labelsize=9)
        ax.locator_params(axis='x', nbins=6)
        ax.locator_params(axis='y', nbins=4)
        ax.set_xlabel('Sea-level rise by 2100 (m)', labelpad=10)
        ax.text(0, 1.07, 'Probability density (m⁻¹)', transform=ax.transAxes,
                fontsize=9, color=MUTED)

    def draw(ax, p, color, label, style='-'):
        ax.fill_between(x, 0, p, color=color, alpha=.10, linewidth=0)
        ax.plot(x, p, color=color, lw=2.8, ls=style, label=label)

    draw(axes[0], low, BLUE, lower_label)
    draw(axes[0], high, SLATE, higher_label, '--')
    draw(axes[1], stable, BLUE, 'Stable Thwaites')
    draw(axes[1], unstable, RED, 'Unstable Thwaites')
    for ax in axes:
        ax.legend(loc='upper right', frameon=False, fontsize=10)

    # Arrow connects model-derived medians; it does not alter the curves.
    axes[0].annotate('', xy=(quantile(x, low, .5), .88*ymax),
                     xytext=(quantile(x, high, .5), .88*ymax),
                     arrowprops={'arrowstyle': '->', 'lw': 2, 'color': BLUE})

    # Highlight the unstable distribution above the stable 95th percentile.
    threshold = quantile(x, stable, .95)
    axes[1].fill_between(x, 0, unstable, where=x >= threshold,
                         interpolate=True, color=RED, alpha=.20, linewidth=0)
    tail_x = quantile(x, unstable, .95)
    tail_y = float(np.interp(tail_x, x, unstable))
    axes[1].annotate('High-end risk is largely\nindependent of future emissions.',
                     xy=(tail_x, tail_y), xycoords='data',
                     xytext=(.98, .60), textcoords='axes fraction',
                     ha='right', va='top', color=RED, fontsize=10, linespacing=1.5,
                     bbox={'facecolor': 'white', 'alpha': .9, 'edgecolor': 'none', 'pad': 3},
                     arrowprops={'arrowstyle': '->', 'lw': 1.5, 'color': RED,
                                 'connectionstyle': 'arc3,rad=-.12'})

    cards = [(.06, .41, BLUE, 'CUT EMISSIONS',
              'Reduce warming and expected sea-level rise.'),
             (.55, .40, RED, 'OBSERVE THWAITES',
              'Narrow uncertainty about the high-end risk.')]
    for left, width, color, heading, body in cards:
        fig.add_artist(FancyBboxPatch((left, .16), width, .095,
                       boxstyle='round,pad=.014,rounding_size=.01',
                       transform=fig.transFigure, facecolor=color, alpha=.065,
                       edgecolor='none'))
        fig.text(left+.012, .221, heading, fontsize=10, weight='bold', color=color)
        fig.text(left+.012, .185, body, fontsize=11)
    fig.text(.06, .071,
             'Model probability densities on shared axes. Darker red shading: '
             'unstable outcomes above the stable-scenario 95th percentile.',
             fontsize=9, color=MUTED)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--output', type=Path, default=Path('sea_level_two_panel.png'))
    parser.add_argument('--lower-label', default='Lower warming')
    parser.add_argument('--higher-label', default='Higher warming')
    parser.add_argument('--reference-label', default='same temperature path')
    parser.add_argument('--baseline', default='2000')
    args = parser.parse_args()
    data = np.genfromtxt(args.csv, delimiter=',', names=True, dtype=float, encoding='utf-8-sig')
    required = ('sea_level_m',) + COLUMNS
    missing = set(required) - set(data.dtype.names or ())
    if missing:
        parser.error('Missing CSV columns: ' + ', '.join(sorted(missing)))
    try:
        fig = make_figure(data['sea_level_m'], *(data[name] for name in COLUMNS),
                          lower_label=args.lower_label, higher_label=args.higher_label,
                          reference_label=args.reference_label, baseline=args.baseline)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
