"""Simplified sea-level figures for donors and general audiences.

Kept deliberately separate from the manuscript notebooks and scripts. The
figures here are accurate, but they omit detail that a scientific reader wants
(model diagnostics, per-component decomposition, calibration, secondary
scenarios) in favour of a single readable message. Nothing is recomputed: every
curve is read from the trajectory ensembles already written to
`data/processed/component_results.h5` by the analysis pipeline.

Figures produced
----------------
`donor_wais_fan_pdf.png`
    Copy of the manuscript's Figure 2v4: GMSL fan plot 2000-2100 at 2/3/4 degC
    with the 2100 probability distributions alongside. Written under a donor
    name so this script never competes with `notebooks/results_figures.ipynb`
    for the manuscript output file.

`wais_probability2100_simple.png`
    The 2100 distribution panel alone, for SSP2-4.5 only, as two conditional
    worlds: Thwaites remains stable and Thwaites is unstable.

A note on which quantity is plotted where
-----------------------------------------
The two figures show different things on purpose. In the fan figure the
`fast WAIS` and `slow WAIS` curves are mixtures over the two WAIS scenarios,
at p(S1) = 10% and p(S1) = 90% respectively; the paper presents them that way
because the observational record cannot retire either branch. The standalone
distribution figure instead shows the two *pure conditionals*, p(S1) = 100% and
p(S1) = 0%, which is the question a donor audience is actually asking: what
happens if Thwaites holds, and what happens if it does not. The medians
therefore do not match between the two figures, and should not.

The unstable branch is the A4 framework's S2 scenario, one parameterized
accelerating trajectory, so `Thwaites is unstable` is a curve conditional on
that scenario rather than a marginal over every possible collapse future.

Run from the repository root:  python scripts/donor_figures.py
"""

import sys
import textwrap
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks'))
sys.path.insert(0, str(ROOT / 'notebooks' / 'arete_mpl'))
sys.path.insert(0, str(ROOT / 'src'))

import arete_mpl  # noqa: E402
from slr_data_readers import (  # noqa: E402
    people_displaced_kulpstrauss2019, slr_cost_jevrejeva2018,
)
from slr_forecast.config import BASELINE_YEAR, FIG_DIR, PROCESSED_DATA_DIR  # noqa: E402

H5_COMP = PROCESSED_DATA_DIR / 'component_results.h5'
H5_OBS = PROCESSED_DATA_DIR / 'slr_processed_data.h5'

# Matches the notebook's plotting environment (results_figures.ipynb cell 2).
arete_mpl.use('poster')
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 10,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
})

# Warming-level labelling follows the manuscript: the SSPs stand in for the
# approximate 2100 warming they deliver.
FAN_SSPS = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0']
FAN_COLORS = {'SSP1-2.6': '#08306b', 'SSP2-4.5': '#4292c6', 'SSP3-7.0': '#d6604d'}
FAN_LABELS = {'SSP1-2.6': r'2$^\circ$C', 'SSP2-4.5': r'3$^\circ$C',
              'SSP3-7.0': r'4$^\circ$C'}
DONOR_SSP = 'SSP2-4.5'

PDF_YEAR = 2100
OBS_COLOR = '0.25'
YLIM = (-0.05, 2.0)          # covers the 90% bands at 4 degC (max ~1.91 m)

# Probability axis: density x grid spacing x 100, i.e. probability (%) per grid
# bin, on the 400-point / 2.5 m grid the manuscript ridge plots use. Held fixed
# so the two donor figures share an x-axis definition.
DY = 2.5 / 399
YGRID = np.arange(YLIM[0], YLIM[1] + DY, DY)

# Impact axes (Kulp & Strauss 2019 population, Tiggeloven adaptation capital
# annualized at 3% over 75 years, Jevrejeva et al. 2018 flood damage).
ADAPT_CAPITAL_PER_M = 4000
_CRF = 0.03 * 1.03 ** 75 / (1.03 ** 75 - 1)
ADAPT_COST_PER_M = ADAPT_CAPITAL_PER_M * _CRF


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
def load_fan():
    """Forecast ensembles for the fan figure, plus the altimetry record."""
    with h5py.File(str(H5_COMP), 'r') as hf:
        years = hf['blended/forecast_years'][:]
        t0 = float(hf['blended'].attrs['t_origin'])
        h0 = float(hf['blended'].attrs['h_origin_m'])
        fast = {s: hf[f'blended/{s}/samples'][:] for s in FAN_SSPS}
        slow_years = hf['blended_slow90/forecast_years'][:]
        slow = {s: hf[f'blended_slow90/{s}/samples'][:] for s in FAN_SSPS}
    assert np.array_equal(years, slow_years)

    nasa = pd.read_hdf(H5_OBS, key='harmonized/df_nasa_gmsl_h')
    obs_t = nasa['decimal_year'].values
    obs_h = nasa['gmsl'].values
    obs_h = obs_h - obs_h[np.argmin(np.abs(obs_t - BASELINE_YEAR))]
    m = obs_t >= 2000
    return years, t0, h0, fast, slow, obs_t[m], obs_h[m]


def load_conditionals(ssp=DONOR_SSP, year=PDF_YEAR):
    """2100 totals under the two pure WAIS conditionals.

    `blended_stable` is the S1 world (no marine ice-sheet instability), i.e.
    p(S1) = 100%. The S2 members of `blended`, selected by the scenario index
    the forecast was built with, are the S2 world, i.e. p(S1) = 0%. Both are
    already rate-space blended forecasts; nothing is recombined here.
    """
    with h5py.File(str(H5_COMP), 'r') as hf:
        years = hf['blended/forecast_years'][:]
        i = int(np.argmin(np.abs(years - year)))
        idx = hf['blended/wais_scenario_idx'][:].astype(int)
        stable = hf[f'blended_stable/{ssp}/samples'][:, i]
        unstable = hf[f'blended/{ssp}/samples'][:][idx == 1][:, i]
    return stable, unstable


def pdf_percent(samples, bandwidth_m):
    """KDE on YGRID, expressed as probability (%) per grid bin."""
    kde = gaussian_kde(samples, bw_method='scott')
    kde.set_bandwidth(bandwidth_m / np.std(samples, ddof=1))
    return kde(YGRID) * DY * 100


def robust_bandwidth(samples, factor):
    """Silverman's outlier-resistant rule, h = 0.9 min(sd, IQR/1.34) n^(-1/5).

    The plain rule leaves visible sampling noise in the heavy S2 tail, so it is
    scaled by a common `factor` for every curve. The factor is fixed (below) so
    that the S2 curve lands on the 0.08 m bandwidth the manuscript figure uses;
    applying the same factor to the other curves smooths each one in proportion
    to its own width. A single fixed bandwidth cannot serve both conditionals:
    the whole S1 distribution has sd ~0.07 m, so 0.08 m would flatten it.
    """
    sd = np.std(samples, ddof=1)
    iqr = np.subtract(*np.percentile(samples, [75, 25]))
    return factor * 0.9 * min(sd, iqr / 1.34) * len(samples) ** (-0.2)


BW_FACTOR = 1.75   # see robust_bandwidth: gives ~0.08 m on the S2 branch
# Annotations on the standalone distribution figure, one per curve:
#   (curve index, text, (left edge, top edge) of the text in metres of sea
#    level, sea level the arrow points to on that curve, arrow curvature)
ANNOTATIONS = [
    (0, 'Responds to atmospheric temperature and is predictable because '
        'Thwaites remains stable.', (0.25, 0.40), 0.58, -0.2),
    (1, 'High-end risk and virtually all uncertainty is due to instabilities '
        'in Thwaites, whose fate is insensitive to future warming.',
     (0.75, 1.75), 1.25, 0.2),
]
FAN_BW_M = 0.08    # manuscript Figure 2v4 value, so figure 1 reproduces it exactly


# ------------------------------------------------------------------
# Shared styling
# ------------------------------------------------------------------
GREY = '#666666'


def style_axis(ax, hide=('top', 'right')):
    for sp in hide:
        ax.spines[sp].set_visible(False)
    for sp in ax.spines.values():
        if sp.get_visible():
            sp.set_position(('outward', 0))   # the style offsets spines
            sp.set_color(GREY)
    ax.tick_params(axis='both', colors=GREY, labelcolor=GREY)
    ax.grid(axis='y', alpha=0.2, lw=0.5)


def add_impact_axes(ax, spacing, tick_fs, label_fs, tick_slr):
    """Right-hand people / cost axes, sharing the sea-level scale of `ax`."""
    pop0 = people_displaced_kulpstrauss2019(0.0)
    specs = [
        ([people_displaced_kulpstrauss2019(v) - pop0 for v in tick_slr],
         lambda v: f'{int(10 * round(v / 10))}', '#666666',
         'Additional People Below\nFlood Level [Millions]'),
        ([ADAPT_COST_PER_M * v for v in tick_slr],
         lambda v: f'{int(round(v))}', '#888888',
         'Global Coastal Adaptation\n' r'Costs [Billions US\$/yr]'),
        ([1e-3 * slr_cost_jevrejeva2018(v) for v in tick_slr],
         lambda v: f'{int(round(v))}', '#999999',
         'Global Flood Damage Without\n' r'Adaptation [Trillions US\$/yr]'),
    ]
    for k, (vals, fmt, clr, lbl) in enumerate(specs):
        axi = ax.twinx()
        axi.set_ylim(YLIM)
        for sp in axi.spines.values():
            sp.set_visible(False)
        axi.spines['right'].set_visible(True)
        axi.spines['right'].set_position(('outward', k * spacing))
        axi.spines['right'].set_color(clr)
        axi.set_yticks(tick_slr)
        axi.set_yticklabels([fmt(v) for v in vals], color=clr, fontsize=tick_fs)
        axi.set_ylabel(lbl, color=clr, fontsize=label_fs)
        axi.tick_params(right=True, left=False, length=3, pad=4, colors=clr)


# ------------------------------------------------------------------
# Figure 1 — fan plot + 2100 distributions
# ------------------------------------------------------------------
def figure_fan_pdf(outfile=FIG_DIR / 'donor_wais_fan_pdf.png'):
    years, t0, h0, fast, slow, obs_t, obs_h = load_fan()
    fs = {'legend': 8, 'label': 10, 'tick': 9, 'impact_tick': 7, 'impact_label': 8}

    # Prepend the forecast origin (end of the altimetry record) so the fan
    # joins the observations.
    mask = years <= 2100
    t = np.concatenate([[t0], years[mask]])

    def stats(samples):
        s = samples[:, mask]
        pad = lambda a: np.concatenate([[h0], a])   # noqa: E731
        return (pad(np.median(s, axis=0)), pad(np.percentile(s, 5, axis=0)),
                pad(np.percentile(s, 95, axis=0)))

    fig = plt.figure(figsize=(6.5, 2.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.08,
                          left=0.10, right=0.72, top=0.97, bottom=0.08)
    ax_fan = fig.add_subplot(gs[0])
    ax_pdf = fig.add_subplot(gs[1], sharey=ax_fan)

    ax_fan.plot(obs_t, obs_h, color=OBS_COLOR, lw=1.0, zorder=10)
    for ssp in FAN_SSPS:
        c = FAN_COLORS[ssp]
        f_med, f_p5, f_p95 = stats(fast[ssp])
        s_med, _, _ = stats(slow[ssp])
        ax_fan.fill_between(t, f_p5, f_p95, color=c, alpha=0.15, lw=0)
        ax_fan.plot(t, f_med, color=c, lw=1.8, ls='-')
        ax_fan.plot(t, s_med, color=c, lw=1.4, ls='--')

    ax_fan.set_xlim(2000, 2100)
    ax_fan.set_ylim(YLIM)
    ax_fan.set_xlabel('Year', fontsize=fs['label'], color=GREY)
    ax_fan.set_ylabel('Sea level rise (m)', fontsize=fs['label'], color=GREY)
    style_axis(ax_fan)
    ax_fan.tick_params(labelsize=fs['tick'], right=False)
    ax_fan.legend(
        handles=[Line2D([0], [0], color=FAN_COLORS[s], lw=1.8, label=FAN_LABELS[s])
                 for s in FAN_SSPS]
        + [Line2D([0], [0], color='0.4', lw=1.8, ls='-', label='Fast WAIS'),
           Line2D([0], [0], color='0.4', lw=1.4, ls='--', label='Slow WAIS'),
           Line2D([0], [0], color=OBS_COLOR, lw=1.0, label='Observations')],
        fontsize=fs['legend'], loc='upper left', ncol=2, frameon=True,
        edgecolor='0.8', facecolor='white', framealpha=1.0)

    i_pdf = int(np.argmin(np.abs(years - PDF_YEAR)))
    curves = {}
    for ssp in FAN_SSPS:
        s = fast[ssp][:, i_pdf]
        curves[ssp] = pdf_percent(s, FAN_BW_M)
        ax_pdf.fill_betweenx(YGRID, 0, curves[ssp], color=FAN_COLORS[ssp],
                             alpha=0.15, lw=0)
        ax_pdf.plot(curves[ssp], YGRID, color=FAN_COLORS[ssp], lw=1.8)

    ax_pdf.set_xlim(0, 1.08 * max(v.max() for v in curves.values()))
    ax_pdf.set_xlabel('Probability in 2100 (%)', fontsize=fs['label'], color=GREY)
    style_axis(ax_pdf)
    ax_pdf.tick_params(axis='x', labelsize=fs['tick'])
    ax_pdf.xaxis.set_major_locator(plt.MaxNLocator(3, prune='lower'))

    add_impact_axes(ax_pdf, spacing=42, tick_fs=fs['impact_tick'],
                    label_fs=fs['impact_label'],
                    tick_slr=np.arange(0.0, YLIM[1] + 1e-9, 0.5))
    # twinx() resets the parent's y ticks: ticks stay on the shared axis, the
    # labels only on the left panel.
    ax_pdf.tick_params(axis='y', left=True, labelleft=False, colors=GREY)

    fig.canvas.draw()   # right-align 2100 so it clears the right panel
    ax_fan.xaxis.get_major_ticks()[-1].label1.set_ha('right')
    fig.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f'{outfile.name}')
    for ssp in FAN_SSPS:
        f_med, f_p5, f_p95 = stats(fast[ssp])
        s_med, _, _ = stats(slow[ssp])
        print(f'  {ssp} 2100: fast {f_med[-1]:.2f} [{f_p5[-1]:.2f}, '
              f'{f_p95[-1]:.2f}] m, slow median {s_med[-1]:.2f} m, '
              f'visible mass {curves[ssp].sum():.0f}%')


# ------------------------------------------------------------------
# Figure 2 — 2100 distribution, stable vs unstable Thwaites
# ------------------------------------------------------------------
def figure_probability_simple(outfile=FIG_DIR / 'wais_probability2100_simple.png'):
    stable, unstable = load_conditionals()
    fs = {'legend': 9, 'label': 11, 'tick': 10, 'impact_tick': 8, 'impact_label': 9}

    cases = [
        ('Thwaites remains stable', stable, 'tab:blue'),
        ('Thwaites is unstable', unstable, 'tab:red'),
    ]

    fig = plt.figure(figsize=(5.4, 3.6))
    gs = fig.add_gridspec(1, 1, left=0.12, right=0.62, top=0.97, bottom=0.14)
    ax = fig.add_subplot(gs[0])

    curves = {}
    for label, s, color in cases:
        h = robust_bandwidth(s, BW_FACTOR)
        curves[label] = (pdf_percent(s, h), s, h)
        ax.fill_betweenx(YGRID, 0, curves[label][0], color=color, alpha=0.15, lw=0)
        ax.plot(curves[label][0], YGRID, color=color, lw=2.0, label=label)
        # Median of the samples themselves, not of the smoothed curve, so the
        # line is unaffected by the KDE bandwidth. Clipped at the curve: it runs
        # from the axis to wherever the curve sits at that sea level.
        med = np.median(s)
        ax.plot([0, curves[label][0][int(np.argmin(np.abs(YGRID - med)))]],
                [med, med], color=color, lw=1.2, ls='--', zorder=1)

    ax.set_ylim(YLIM)
    ax.set_xlim(0, 1.08 * max(v[0].max() for v in curves.values()))
    ax.set_xlabel('Probability in 2100 (%)', fontsize=fs['label'], color=GREY)
    ax.set_ylabel('Sea level rise (m)', fontsize=fs['label'], color=GREY)
    style_axis(ax)
    ax.tick_params(labelsize=fs['tick'])
    ax.xaxis.set_major_locator(plt.MaxNLocator(4, prune='lower'))
    # Annotations in place of a legend: one per curve, each hanging from the
    # stated top edge into empty space, with an arrow onto its own curve.
    for k, text, (x_text, y_top), y_target, rad in ANNOTATIONS:
        label, _, color = cases[k]
        pdf = curves[label][0]
        j = int(np.argmin(np.abs(YGRID - y_target)))
        ax.annotate(
            '\n'.join(textwrap.wrap(text, width=30)),
            xy=(pdf[j], YGRID[j]), xytext=(x_text, y_top),
            ha='left', va='top', fontsize=fs['tick'], color=color,
            arrowprops=dict(arrowstyle='->', color=color, lw=1.2,
                            shrinkA=4, shrinkB=2,
                            connectionstyle=f'arc3,rad={rad}'))

    # The sea-level ticks and the impact ticks must share one array, or the
    # right-hand columns stop corresponding to the left axis.
    tick_slr = np.arange(0.0, YLIM[1] + 1e-9, 0.5)
    add_impact_axes(ax, spacing=46, tick_fs=fs['impact_tick'],
                    label_fs=fs['impact_label'], tick_slr=tick_slr)
    # twinx() resets the parent's y ticks; restore them on the standalone axis.
    ax.set_yticks(tick_slr)
    ax.tick_params(axis='y', left=True, labelleft=True, right=False,
                   colors=GREY, labelcolor=GREY, labelsize=fs['tick'])

    fig.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f'\n{outfile.name}  ({DONOR_SSP}, {PDF_YEAR})')
    for label, s, _ in cases:
        pdf, _, h = curves[label]
        print(f'  {label:24s}: median {np.median(s):.2f} m, '
              f'90% [{np.percentile(s, 5):.2f}, {np.percentile(s, 95):.2f}] m, '
              f'n = {len(s)}, KDE bandwidth {h:.3f} m, '
              f'visible mass {pdf.sum():.0f}%')


if __name__ == '__main__':
    figure_fan_pdf()
    figure_probability_simple()
