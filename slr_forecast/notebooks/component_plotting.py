"""
Component decomposition plotting functions.

Extracted from component_decomposition.ipynb to keep notebooks free of
function definitions.  All plotting functions accept data in internal
units (meters, °C, yr) and convert to display units (mm) internally.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
try:
    from slr_forecast.config import Z_90
    from slr_forecast import M_TO_MM
except ImportError:
    M_TO_MM = 1000.0
    Z_90 = 1.645

# ---------------------------------------------------------------------------
# Standard colour palettes
# ---------------------------------------------------------------------------
SSP_COLORS = {
    'SSP1-2.6': '#2166ac', 'SSP2-4.5': '#4393c3',
    'SSP3-7.0': '#d6604d', 'SSP5-8.5': '#b2182b',
}

COMP_COLORS = {
    'Thermosteric': '#B22222',  # firebrick red (complementary warm)
    'Glaciers':     '#808080',  # grey
    'Greenland':    '#2E8B57',  # sea green (complementary to red)
    'TWS':          '#AB6638',  # Arete Earth (brown/copper)
    'WAIS':         '#036C9A',  # Glacier Blue 400 (darkest blue)
    'EAIS':         '#B5D5E9',  # Glacier Blue 200 (lightest blue)
    'Peninsula':    '#72A3C3',  # Glacier Blue 300 (mid blue)
}

ARETE_COLORS = {
    'blues': ['#D8E7F1', '#B5D5E9', '#72A3C3', '#036C9A', '#07456C', '#031D3A'],
    'greys': ['#FFFFFF', '#D9D9D9', '#000000'],
    'brown': ['#AB6638'],
}

PANEL_COLORS = {
    'Total GMSL': '#333333',
    'Thermosteric': 'C0', 'Glaciers': 'C1', 'Greenland': 'C2',
    'EAIS': 'C3', 'Peninsula': 'C6', 'WAIS': 'C4', 'TWS': 'C5',
}

TAPER_BLUES = {1: '#a6cee3', 2: '#4a90d9', 3: '#08306b'}
TAPER_REDS = {1: '#fcbba1', 2: '#d94801', 3: '#67000d'}


# =========================================================================
# Frederikse overview
# =========================================================================

def plot_frederikse_overview(fred_year, fred_gmsl, fred_gmsl_sigma,
                             fred_steric, fred_steric_sigma,
                             fred_glaciers, fred_glaciers_sigma,
                             fred_greenland, fred_greenland_sigma,
                             fred_antarctica, fred_antarctica_sigma,
                             fred_tws, fred_tws_sigma,
                             fred_barystatic, fred_barystatic_sigma,
                             fred_thermo_gmsl, save_path=None):
    """Two-panel figure: component decomposition + thermo vs barystatic."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # Panel A: cumulative components
    ax = axes[0]
    components = [
        ('Steric (thermal expansion)', fred_steric, fred_steric_sigma, 'C0'),
        ('Glaciers', fred_glaciers, fred_glaciers_sigma, 'C1'),
        ('Greenland', fred_greenland, fred_greenland_sigma, 'C2'),
        ('Antarctica', fred_antarctica, fred_antarctica_sigma, 'C3'),
        ('Terrestrial water storage', fred_tws, fred_tws_sigma, 'C4'),
    ]
    for label, val, sig, color in components:
        ax.plot(fred_year, val * M_TO_MM, color=color, label=label)
        ax.fill_between(fred_year,
                        (val - 2 * sig) * M_TO_MM,
                        (val + 2 * sig) * M_TO_MM,
                        color=color, alpha=0.15)
    ax.plot(fred_year, fred_gmsl * M_TO_MM, 'k-', lw=2, label='Total GMSL')
    ax.fill_between(fred_year,
                    (fred_gmsl - 2 * fred_gmsl_sigma) * M_TO_MM,
                    (fred_gmsl + 2 * fred_gmsl_sigma) * M_TO_MM,
                    color='k', alpha=0.1)
    ax.set_ylabel('Cumulative sea level change (mm)')
    ax.set_title('Frederikse et al. (2020) — Component Decomposition of GMSL')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.axhline(0, color='gray', ls=':', lw=0.5)
    ax.grid(True, alpha=0.3)

    # Panel B: thermo vs barystatic
    ax = axes[1]
    ax.plot(fred_year, fred_steric * M_TO_MM, 'C0-', lw=2,
            label='Thermosteric (steric obs)')
    ax.fill_between(fred_year,
                    (fred_steric - 2 * fred_steric_sigma) * M_TO_MM,
                    (fred_steric + 2 * fred_steric_sigma) * M_TO_MM,
                    color='C0', alpha=0.15)
    ax.plot(fred_year, fred_barystatic * M_TO_MM, 'C3-', lw=2,
            label='Barystatic (glaciers + ice sheets + TWS)')
    ax.fill_between(fred_year,
                    (fred_barystatic - 2 * fred_barystatic_sigma) * M_TO_MM,
                    (fred_barystatic + 2 * fred_barystatic_sigma) * M_TO_MM,
                    color='C3', alpha=0.15)
    ax.plot(fred_year, fred_thermo_gmsl * M_TO_MM, 'C0--', lw=1.5,
            label='GMSL − barystatic (residual thermosteric)')
    ax.set_ylabel('Cumulative sea level change (mm)')
    ax.set_xlabel('Year')
    ax.set_title('Thermosteric vs Barystatic Partition')
    ax.legend(loc='upper left', fontsize=8)
    ax.axhline(0, color='gray', ls=':', lw=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# =========================================================================
# Model fit visualisation
# =========================================================================

def plot_model_fits(fred_year, steric_rebase, steric_sigma,
                    gmsl_rebase, gmsl_sigma,
                    H_ens_thermo, H_ens_total, r2_thermo,
                    save_path=None):
    """Two-panel: thermosteric + total GMSL model fits.

    Parameters
    ----------
    H_ens_thermo, H_ens_total : ndarray (n_draw, n_time)
        Ensemble model predictions in meters.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # Panel A: Thermosteric
    ax = axes[0]
    ax.errorbar(fred_year, steric_rebase * M_TO_MM,
                yerr=2 * steric_sigma * M_TO_MM,
                fmt='o', ms=2, color='C0', alpha=0.5, label='Frederikse steric obs')
    p5, p50, p95 = np.percentile(H_ens_thermo * M_TO_MM, [5, 50, 95], axis=0)
    ax.plot(fred_year, p50, 'C0-', lw=2, label='Bayesian fit (median)')
    ax.fill_between(fred_year, p5, p95, color='C0', alpha=0.2, label='90% CI')
    ax.set_ylabel('Steric sea level (mm, 2005 baseline)')
    ax.set_title(f'Thermosteric Component — R² = {r2_thermo:.4f}')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: Total GMSL
    ax = axes[1]
    ax.errorbar(fred_year, gmsl_rebase * M_TO_MM,
                yerr=2 * gmsl_sigma * M_TO_MM,
                fmt='o', ms=2, color='k', alpha=0.5, label='Frederikse GMSL obs')
    p5t, p50t, p95t = np.percentile(H_ens_total * M_TO_MM, [5, 50, 95], axis=0)
    ax.plot(fred_year, p50t, 'k-', lw=2, label='Bayesian fit (median)')
    ax.fill_between(fred_year, p5t, p95t, color='gray', alpha=0.2, label='90% CI')
    ax.plot(fred_year, p50, 'C0--', lw=1.5, label='Thermosteric component')
    ax.set_ylabel('Sea level (mm, 2005 baseline)')
    ax.set_xlabel('Year')
    ax.set_title('Total GMSL with Thermosteric Component')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# =========================================================================
# Residual validation
# =========================================================================

def plot_residual_validation(fred_year, barystatic_rebase, fred_barystatic_sigma,
                             resid_model, ice_years, ice_slr,
                             glac_years, glac_cumul_mm,
                             save_path=None):
    """Cryospheric residual vs direct observations."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(fred_year, barystatic_rebase * M_TO_MM, 'k-', lw=2,
            label='Frederikse barystatic (glaciers+ice sheets+TWS)')
    ax.fill_between(fred_year,
                    (barystatic_rebase - 2 * fred_barystatic_sigma) * M_TO_MM,
                    (barystatic_rebase + 2 * fred_barystatic_sigma) * M_TO_MM,
                    color='gray', alpha=0.15)
    ax.plot(fred_year, resid_model * M_TO_MM, 'C3--', lw=2,
            label='GMSL − fitted thermosteric (model residual)')
    ax.plot(ice_years, ice_slr * M_TO_MM, 'C2-', lw=1.5,
            label='IMBIE ice sheets (GrIS + AIS)')
    ax.plot(glac_years, glac_cumul_mm, 'C1-', lw=1.5,
            label='GlaMBIE glaciers (global)')

    ax.set_ylabel('Cumulative sea level change (mm, 2005 baseline)')
    ax.set_xlabel('Year')
    ax.set_title('Cryospheric + TWS Residual — Validation against Direct Observations')
    ax.legend(loc='upper left', fontsize=8)
    ax.axhline(0, color='gray', ls=':', lw=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1900, 2023)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# =========================================================================
# Rate vs temperature (8-panel figure)
# =========================================================================

def plot_rate_vs_temperature(panel_order, rates, T_for_rate, model_rates,
                             T_annual, obs_windows=None,
                             save_path=None):
    """Eight-panel rate vs temperature figure.

    Parameters
    ----------
    panel_order : list of str
        Component names in order.
    rates : dict
        ``{name: rate_array_mm_yr}`` — kernel-smoothed observed rates.
    T_for_rate : dict
        ``{name: T_array}`` — temperature at each rate point.
    model_rates : dict
        ``{name: {'a': ndarray, 'b': ndarray, 'c': ndarray}}`` — posterior samples (meters).
    T_annual : ndarray
        Annual GMST.
    obs_windows : dict or None
        ``{name: (start_yr, end_yr)}`` for marker styling.
    save_path : str or None
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    T_grid = np.linspace(T_annual.min() - 0.05, T_annual.max() + 0.05, 200)

    for ax, name in zip(axes.flat, panel_order):
        color = PANEL_COLORS.get(name, 'gray')

        if name in rates:
            r = rates[name]
            T_r = T_for_rate[name]
            valid = np.isfinite(r)
            ax.scatter(T_r[valid], r[valid], s=8, alpha=0.4, color=color, zorder=2)

        if name in model_rates:
            mr = model_rates[name]
            a_med = np.median(mr['a']) * M_TO_MM
            b_med = np.median(mr['b']) * M_TO_MM
            c_med = np.median(mr['c']) * M_TO_MM

            T_comp = T_for_rate.get(name, T_grid)
            if hasattr(T_comp, '__len__') and len(T_comp) > 0:
                T_fit = np.linspace(np.nanmin(T_comp) - 0.02,
                                    np.nanmax(T_comp) + 0.02, 200)
            else:
                T_fit = T_grid

            rate_quad = a_med * T_fit ** 2 + b_med * T_fit + c_med
            ax.plot(T_fit, rate_quad, '-', color='k', lw=2.0, alpha=0.8, zorder=4)

            n_draw = min(500, len(mr['a']))
            rate_draws = np.array([
                mr['a'][k] * M_TO_MM * T_fit ** 2
                + mr['b'][k] * M_TO_MM * T_fit
                + mr['c'][k] * M_TO_MM
                for k in range(n_draw)
            ])
            ax.fill_between(T_fit,
                            np.percentile(rate_draws, 5, axis=0),
                            np.percentile(rate_draws, 95, axis=0),
                            color='k', alpha=0.08, zorder=3)

            rate_lin = b_med * T_fit + c_med
            ax.plot(T_fit, rate_lin, '--', color='C5', lw=1.5, alpha=0.8, zorder=4)

            a_p5, a_p95 = np.percentile(mr['a'] * M_TO_MM, [5, 95])
            p_a_positive = np.mean(mr['a'] > 0) * 100
            ax.text(0.03, 0.97,
                    f'a = {a_med:.3f} [{a_p5:.3f}, {a_p95:.3f}]\n'
                    f'P(a>0) = {p_a_positive:.0f}%',
                    transform=ax.transAxes, fontsize=7,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Temperature (°C)', fontsize=9)
        ax.set_ylabel('Rate (mm/yr)', fontsize=9)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# =========================================================================
# Component projections
# =========================================================================

def plot_component_projections(comp_projections, proj_years, ipcc_total,
                               proj_ssps, save_path=None):
    """Four-panel: component fans for SSP2-4.5, SSP5-8.5, total vs IPCC, budget table.

    Parameters
    ----------
    comp_projections : dict
        ``{ssp: {component: {'median': ..., 'p17': ..., 'p83': ..., 'p5': ..., 'p95': ..., 'samples': ...}}}``
    proj_years : ndarray
    ipcc_total : dict
        ``{ssp: {'years': ..., 'quantiles': ..., 'slc': ...}}``
    proj_ssps : list of str
    save_path : str or None
    """
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.30, wspace=0.25)

    proj_mask = proj_years >= 2005
    yr_plot = proj_years[proj_mask]

    comp_order = ['Thermosteric', 'Glaciers', 'Greenland', 'WAIS',
                  'Peninsula', 'EAIS', 'TWS']

    for panel_idx, ssp_show in enumerate(['SSP2-4.5', 'SSP5-8.5']):
        ax = fig.add_subplot(gs[0, panel_idx])
        for cname in comp_order:
            if cname not in comp_projections.get(ssp_show, {}):
                continue
            p = comp_projections[ssp_show][cname]
            med = p['median'][proj_mask] * M_TO_MM
            lo = p['p17'][proj_mask] * M_TO_MM
            hi = p['p83'][proj_mask] * M_TO_MM
            ax.plot(yr_plot, med, color=COMP_COLORS.get(cname, 'gray'),
                    lw=1.5, label=cname)
            ax.fill_between(yr_plot, lo, hi,
                            color=COMP_COLORS.get(cname, 'gray'), alpha=0.15)
        ax.set_xlabel('Year')
        ax.set_ylabel('Sea-level contribution (mm)')
        ax.set_title(f'Component projections — {ssp_show}')
        ax.legend(fontsize=8, loc='upper left', ncol=2)
        ax.set_xlim(2005, 2150)
        ax.axhline(0, color='k', lw=0.5, ls='--')

    # Bottom-left: total sum vs IPCC
    ax2 = fig.add_subplot(gs[1, 0])
    for ssp in proj_ssps:
        if ssp not in comp_projections or 'Total_sum' not in comp_projections[ssp]:
            continue
        p = comp_projections[ssp]['Total_sum']
        med = p['median'][proj_mask] * M_TO_MM
        lo = p['p5'][proj_mask] * M_TO_MM
        hi = p['p95'][proj_mask] * M_TO_MM
        ax2.plot(yr_plot, med, color=SSP_COLORS.get(ssp, 'gray'), lw=2,
                 label=f'{ssp} (component sum)')
        ax2.fill_between(yr_plot, lo, hi,
                         color=SSP_COLORS.get(ssp, 'gray'), alpha=0.12)
        if ssp in ipcc_total:
            ipcc_d = ipcc_total[ssp]
            q_med = np.argmin(np.abs(ipcc_d['quantiles'] - 0.5))
            ax2.plot(ipcc_d['years'], ipcc_d['slc'][q_med],
                     color=SSP_COLORS.get(ssp, 'gray'), lw=1.5, ls='--', alpha=0.7)

    ax2.set_xlabel('Year')
    ax2.set_ylabel('Total GMSL (mm, rel. to 2005)')
    ax2.set_title('Component sum (solid) vs IPCC total (dashed)')
    ax2.set_xlim(2005, 2150)
    ax2.axhline(0, color='k', lw=0.5, ls='--')
    ax2.legend(fontsize=8, loc='upper left')

    # Bottom-right: summary table
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    rows = []
    for yr in [2050, 2100, 2150]:
        idx_yr = np.argmin(np.abs(proj_years - yr))
        for ssp in proj_ssps:
            if ssp not in comp_projections or 'Total_sum' not in comp_projections[ssp]:
                continue
            total_med = comp_projections[ssp]['Total_sum']['median'][idx_yr] * M_TO_MM
            total_lo = comp_projections[ssp]['Total_sum']['p5'][idx_yr] * M_TO_MM
            total_hi = comp_projections[ssp]['Total_sum']['p95'][idx_yr] * M_TO_MM
            rows.append([yr, ssp, f'{total_med:.0f}',
                         f'[{total_lo:.0f}, {total_hi:.0f}]'])
    if rows:
        table = ax3.table(
            cellText=rows,
            colLabels=['Year', 'SSP', 'Median (mm)', '90% CI'],
            loc='center', cellLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# =========================================================================
# Variance decomposition
# =========================================================================

def plot_variance_decomposition(var_fracs_wc, var_fracs_full,
                                proj_years, ssp_label,
                                well_constrained, component_order,
                                save_path=None):
    """Two-panel stacked-area variance decomposition.

    Parameters
    ----------
    var_fracs_wc, var_fracs_full : dict of ndarray
        Variance fractions from ``compute_variance_fractions``.
    proj_years : ndarray
    ssp_label : str
    well_constrained, component_order : list of str
    save_path : str or None
    """
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: well-constrained
    bottoms = np.zeros(len(proj_years))
    for cname in well_constrained:
        frac = var_fracs_wc[cname]
        ax_a.fill_between(proj_years, bottoms, bottoms + frac,
                          color=COMP_COLORS.get(cname, '#999999'),
                          alpha=0.7, label=cname)
        bottoms += frac
    ax_a.set_xlim(2020, 2150)
    ax_a.set_ylim(0, 1.05)
    ax_a.set_xlabel('Year')
    ax_a.set_ylabel('Fraction of variance (normalised)')
    ax_a.set_title(f'Panel A: Well-constrained components — {ssp_label}')
    ax_a.legend(fontsize=9, loc='center right')

    # Panel B: all components
    bottoms = np.zeros(len(proj_years))
    for cname in component_order:
        frac = var_fracs_full[cname]
        ax_b.fill_between(proj_years, bottoms, bottoms + frac,
                          color=COMP_COLORS.get(cname, '#999999'),
                          alpha=0.7, label=cname)
        bottoms += frac
    ax_b.set_xlim(2020, 2150)
    ax_b.set_ylim(0, 1.05)
    ax_b.set_xlabel('Year')
    ax_b.set_ylabel('Fraction of variance (normalised)')
    ax_b.set_title(f'Panel B: All components (incl. WAIS) — {ssp_label}')
    ax_b.legend(fontsize=8, loc='center right', ncol=1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# =========================================================================
# Historical budget closure
# =========================================================================

def plot_budget_closure(fred_year, fred_gmsl_rebase,
                        H_thermo_model, H_glacier_model, H_greenland_model,
                        fred_antarctica_rebase, fred_tws_rebase,
                        r2_budget, rms_residual,
                        save_path=None):
    """Two-panel: component lines vs GMSL, and residual."""
    valid = ~np.isnan(H_greenland_model)
    H_sum = (H_thermo_model + H_glacier_model + H_greenland_model
             + fred_antarctica_rebase + fred_tws_rebase)

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(14, 9),
                                      height_ratios=[3, 1], sharex=True)

    ax_a.plot(fred_year, fred_gmsl_rebase * M_TO_MM, 'k-', lw=2.5,
              label='Observed GMSL', zorder=10)
    ax_a.plot(fred_year[valid], H_sum[valid] * M_TO_MM, '--', color='dimgray',
              lw=2, label='Sum of components', zorder=9)
    for name, vals, color in [
        ('Thermosteric (model)', H_thermo_model, COMP_COLORS['Thermosteric']),
        ('Glaciers (model)', H_glacier_model, COMP_COLORS['Glaciers']),
        ('Greenland (model)', H_greenland_model, COMP_COLORS['Greenland']),
        ('Antarctica (obs)', fred_antarctica_rebase, 'C3'),
        ('TWS (obs)', fred_tws_rebase, COMP_COLORS['TWS']),
    ]:
        v = ~np.isnan(vals)
        ax_a.plot(fred_year[v], vals[v] * M_TO_MM, color=color, lw=1.5, label=name)

    ax_a.set_ylabel('Sea level (mm, 2005 baseline)')
    ax_a.set_title(f'Historical Budget Closure  (R² = {r2_budget:.4f}, '
                    f'RMS = {rms_residual:.1f} mm)')
    ax_a.legend(fontsize=8, loc='upper left', ncol=2)
    ax_a.grid(True, alpha=0.2)

    residual = fred_gmsl_rebase[valid] - H_sum[valid]
    ax_b.plot(fred_year[valid], residual * M_TO_MM, 'k-', lw=1.5)
    ax_b.fill_between(fred_year[valid], 0, residual * M_TO_MM,
                       color='gray', alpha=0.2)
    ax_b.axhline(0, color='k', lw=0.5, ls='--')
    ax_b.set_xlabel('Year')
    ax_b.set_ylabel('Residual (mm)')
    ax_b.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# =========================================================================
# IPCC comparison (6-panel)
# =========================================================================

def plot_ipcc_comparison(comp_projections, ipcc_components, proj_years,
                         comp_ssps, n_samples, ipcc_extract_fn,
                         save_path=None):
    """Six-panel: each component vs IPCC counterpart.

    Parameters
    ----------
    comp_projections : dict
    ipcc_components : dict
    proj_years : ndarray
    comp_ssps : list of str
    n_samples : int
    ipcc_extract_fn : callable
        The ``ipcc_extract`` function.
    save_path : str or None
    """
    panel_config = [
        ('Thermosteric', 'Thermosteric', 'Thermosteric vs IPCC Ocean Dynamics'),
        ('Glaciers', 'Glaciers', 'Glaciers vs IPCC Glaciers'),
        ('Greenland', 'Greenland', 'Greenland vs IPCC GIS'),
        ('AIS', 'AIS', 'Total AIS vs IPCC AIS'),
        ('TWS', 'TWS', 'TWS vs IPCC Land Water Storage'),
        ('Total', 'Total_sum', 'Total GMSL vs IPCC Total'),
    ]

    proj_mask = proj_years >= 2005
    yr_plot = proj_years[proj_mask]

    fig, axes = plt.subplots(3, 2, figsize=(16, 16))

    for idx_panel, (ipcc_key, our_key, title) in enumerate(panel_config):
        ax = axes.flat[idx_panel]
        for ssp in comp_ssps:
            color = SSP_COLORS.get(ssp, 'gray')

            if our_key == 'AIS':
                our_samples = np.zeros((n_samples, len(proj_years)))
                for cname in ['EAIS', 'Peninsula', 'WAIS']:
                    if cname in comp_projections.get(ssp, {}):
                        our_samples += comp_projections[ssp][cname]['samples']
                our_med = np.median(our_samples, axis=0)[proj_mask] * M_TO_MM
                our_lo = np.percentile(our_samples, 5, axis=0)[proj_mask] * M_TO_MM
                our_hi = np.percentile(our_samples, 95, axis=0)[proj_mask] * M_TO_MM
            elif our_key in comp_projections.get(ssp, {}):
                p = comp_projections[ssp][our_key]
                our_med = p['median'][proj_mask] * M_TO_MM
                our_lo = p['p5'][proj_mask] * M_TO_MM
                our_hi = p['p95'][proj_mask] * M_TO_MM
            else:
                continue

            ax.plot(yr_plot, our_med, color=color, lw=2,
                    label=f'{ssp} (this study)')
            ax.fill_between(yr_plot, our_lo, our_hi, color=color, alpha=0.12)

            if ipcc_key in ipcc_components.get(ssp, {}):
                ipcc_d = ipcc_extract_fn(ipcc_components[ssp][ipcc_key])
                ax.plot(ipcc_d['years'], ipcc_d['q50'], color=color, lw=1.5,
                        ls='--', label=f'{ssp} (IPCC AR6)')
                ax.fill_between(ipcc_d['years'], ipcc_d['q05'], ipcc_d['q95'],
                                color=color, alpha=0.06)

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Sea-level contribution (mm)')
        ax.set_xlim(2005, 2150)
        ax.axhline(0, color='k', lw=0.5, ls='--')
        ax.legend(fontsize=7, loc='upper left')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# =========================================================================
# Summary table (print-based)
# =========================================================================

def format_summary_table(comp_projections, ipcc_components, proj_years,
                         comp_ssps, n_samples,
                         get_our_stats_fn, get_ipcc_stats_fn):
    """Print formatted comparison table at 2050 and 2100.

    Parameters
    ----------
    comp_projections, ipcc_components : dict
    proj_years : ndarray
    comp_ssps : list of str
    n_samples : int
    get_our_stats_fn, get_ipcc_stats_fn : callable
    """
    components_table = [
        ('Thermosteric', 'Thermosteric', 'Thermosteric'),
        ('Glaciers', 'Glaciers', 'Glaciers'),
        ('Greenland', 'Greenland', 'Greenland'),
        ('AIS', 'AIS', 'Total AIS'),
        ('TWS', 'TWS', 'TWS'),
        ('Total_sum', 'Total', 'Total GMSL'),
    ]

    for year in [2050, 2100]:
        print(f'\n{"=" * 90}')
        print(f'PROJECTION COMPARISON AT {year}: This Study vs IPCC AR6 (medium confidence)')
        print('=' * 90)
        for ssp in comp_ssps:
            print(f'\n{ssp}:')
            print(f'  {"Component":<18s} {"This study":>25s} {"IPCC AR6":>25s} {"Ratio":>8s}')
            print(f'  {"-" * 76}')
            for our_key, ipcc_key, label in components_table:
                our = get_our_stats_fn(comp_projections, proj_years, ssp,
                                       our_key, year=year, n_samples=n_samples)
                ipcc = get_ipcc_stats_fn(ipcc_components, ssp, ipcc_key, year=year)
                our_str = (f'{our[1]:.0f} [{our[0]:.0f}, {our[2]:.0f}]'
                           if our else '—')
                ipcc_str = (f'{ipcc[1]:.0f} [{ipcc[0]:.0f}, {ipcc[2]:.0f}]'
                            if ipcc else '—')
                ratio_str = '—'
                if our and ipcc and ipcc[1] != 0:
                    ratio_str = f'{our[1] / ipcc[1]:.2f}'
                print(f'  {label:<18s} {our_str:>25s} {ipcc_str:>25s} {ratio_str:>8s}')


# =========================================================================
# Taper sensitivity figure (for sensitivities notebook)
# =========================================================================

def plot_taper_sensitivity(taper_results, all_components, F_MAX_VALUES,
                           taper_restricted_data,
                           eais_year=None, pen_year=None,
                           save_path=None):
    """Two-panel: a coefficient vs f_max, and ΔBIC vs f_max.

    Parameters
    ----------
    taper_results : dict
        ``{f_max: {component: {'quad': result, 'linear': result}}}``
    all_components : list of str
    F_MAX_VALUES : list of int
    taper_restricted_data : dict
        ``{component: {'years': ..., 'window': ...}}``
    save_path : str or None
    """
    def _get_n_obs(name):
        if name in taper_restricted_data:
            return len(taper_restricted_data[name]['years'])
        elif name == 'EAIS' and eais_year is not None:
            return len(eais_year)
        elif name == 'Peninsula' and pen_year is not None:
            return len(pen_year)
        return 100  # fallback

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: a coefficient with 90% CI
    ax = axes[0]
    offsets = np.linspace(-0.15, 0.15, len(all_components))
    for j, name in enumerate(all_components):
        a_meds, a_lows, a_highs = [], [], []
        for f_max in F_MAX_VALUES:
            a_s = taper_results[f_max][name]['quad'].posterior_samples[:, 0] * M_TO_MM
            a_meds.append(np.median(a_s))
            a_lows.append(np.percentile(a_s, 5))
            a_highs.append(np.percentile(a_s, 95))

        x = np.array(F_MAX_VALUES) + offsets[j]
        ax.errorbar(x, a_meds,
                    yerr=[np.array(a_meds) - np.array(a_lows),
                          np.array(a_highs) - np.array(a_meds)],
                    fmt='o-', capsize=3, label=name, ms=5)

    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.set_xlabel('f_max (σ inflation factor)')
    ax.set_ylabel('a (mm/yr/°C²)')
    ax.set_title('Quadratic coefficient sensitivity to σ taper')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.2)

    # Panel 2: ΔBIC
    ax = axes[1]
    for j, name in enumerate(all_components):
        dbics = []
        n = _get_n_obs(name)
        for f_max in F_MAX_VALUES:
            tr = taper_results[f_max][name]
            rss_q = np.sum(tr['quad'].residuals ** 2)
            rss_l = np.sum(tr['linear'].residuals ** 2)
            bic_q = n * np.log(rss_q / n) + 5 * np.log(n)
            bic_l = n * np.log(rss_l / n) + 4 * np.log(n)
            dbics.append(bic_l - bic_q)
        x = np.array(F_MAX_VALUES) + offsets[j]
        ax.plot(x, dbics, 'o-', label=name, ms=5)

    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.axhline(2, color='gray', ls=':', lw=0.5)
    ax.axhline(-2, color='gray', ls=':', lw=0.5)
    ax.set_xlabel('f_max (σ inflation factor)')
    ax.set_ylabel('ΔBIC (positive = quadratic preferred)')
    ax.set_title('Model selection sensitivity to σ taper')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# =========================================================================
# Greenland local-T vs GMST figure (for sensitivities notebook)
# =========================================================================

def plot_greenland_local_vs_gmst(gr_local_taper_results, gr_gmst_taper_results,
                                  rates_greenland, fred_year,
                                  T_gr_on_fred, T_annual,
                                  yrs_gr_local, F_MAX_VALUES,
                                  save_path=None):
    """Side-by-side: Greenland rate vs T with local-T and GMST.

    Parameters
    ----------
    gr_local_taper_results, gr_gmst_taper_results : dict
        ``{f_max: {'quad': result, 'linear': result}}``
    rates_greenland : ndarray
        Kernel-smoothed Greenland rates (mm/yr).
    fred_year : ndarray
    T_gr_on_fred, T_annual : ndarray
        Greenland T and GMST on fred_year grid.
    yrs_gr_local : ndarray
    F_MAX_VALUES : list of int
    save_path : str or None
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ow_start, ow_end = 1972, 2015

    for ax_idx, (ax, label, taper_dict, T_scatter) in enumerate(zip(
        axes,
        ['Greenland Temperature', 'GMST'],
        [gr_local_taper_results, gr_gmst_taper_results],
        [T_gr_on_fred, T_annual],
    )):
        r_gr = rates_greenland
        yr = fred_year
        valid = np.isfinite(r_gr) & np.isfinite(T_scatter)

        obs_mask = valid & (yr >= ow_start) & (yr <= ow_end)
        pre_mask = valid & (yr < ow_start)
        norm = plt.Normalize(yr[valid].min(), yr[valid].max())

        if pre_mask.any():
            ax.scatter(T_scatter[pre_mask], r_gr[pre_mask], c=yr[pre_mask],
                       s=18, alpha=0.35, cmap='viridis', norm=norm,
                       marker='^', edgecolors='dimgray', linewidths=0.4, zorder=2)
        if obs_mask.any():
            ax.scatter(T_scatter[obs_mask], r_gr[obs_mask], c=yr[obs_mask],
                       s=18, alpha=0.55, cmap='viridis', norm=norm,
                       marker='o', edgecolors='dimgray', linewidths=0.4, zorder=2)

        T_obs_vals = T_scatter[obs_mask & np.isfinite(T_scatter)]
        if len(T_obs_vals) > 0:
            T_fit = np.linspace(T_obs_vals.min() - 0.05, T_obs_vals.max() + 0.05, 200)
        else:
            T_fit = np.linspace(-1, 2, 200)

        for f_max in F_MAX_VALUES:
            tr = taper_dict[f_max]
            ps_q = tr['quad'].posterior_samples
            a_s, b_s, c_s = ps_q[:, 0], ps_q[:, 1], ps_q[:, 2]

            a_med = np.median(a_s) * M_TO_MM
            b_med = np.median(b_s) * M_TO_MM
            c_med = np.median(c_s) * M_TO_MM
            rate_quad = a_med * T_fit ** 2 + b_med * T_fit + c_med

            qcol = TAPER_BLUES[f_max]
            ax.plot(T_fit, rate_quad, '-', color=qcol, lw=2.0,
                    label=f'Quad f={f_max}', zorder=5 + f_max)

            b_lin = tr['linear'].posterior_samples[:, 1]
            c_lin = tr['linear'].posterior_samples[:, 2]
            rate_lin = np.median(b_lin) * M_TO_MM * T_fit + np.median(c_lin) * M_TO_MM

            lcol = TAPER_REDS[f_max]
            ax.plot(T_fit, rate_lin, '--', color=lcol, lw=1.5,
                    label=f'Lin f={f_max}', zorder=4 + f_max)

        ax.set_title(f'Greenland — {label}', fontsize=11, fontweight='bold')
        ax.set_xlabel(f'{label} anomaly (°C)', fontsize=10)
        ax.set_ylabel('Rate (mm/yr)', fontsize=10)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# =========================================================================
# Component projection two-panel (SLR + temperature forcing)
# =========================================================================

def plot_component_projection_twopanel(comp_proj, proj_years, component_name,
                                        temperature_scenarios=None,
                                        temp_label='GMST anomaly (°C)',
                                        ssps=None, xlim=(2005, 2150),
                                        ipcc_data=None, ipcc_key=None,
                                        obs_years=None, obs_vals=None,
                                        obs_sigma=None, obs_label=None,
                                        temp_obs_years=None, temp_obs_vals=None,
                                        temp_obs_label=None,
                                        units='m', save_path=None):
    """Two-panel figure: upper panel shows component SLR projections under
    multiple SSPs with CI bands; lower panel shows the temperature forcing.

    Parameters
    ----------
    comp_proj : dict
        ``{ssp: {'samples': ..., 'median': ..., 'p5': ..., 'p17': ...,
        'p83': ..., 'p95': ...}}`` — projection data in meters.
    proj_years : ndarray
        Projection year array.
    component_name : str
        Component name for titles.
    temperature_scenarios : dict or None
        ``{ssp: {'years': ndarray, 'temperature': ndarray}}`` for the
        lower panel. If None, lower panel is omitted.
    temp_label : str
        Y-axis label for temperature panel.
    ssps : list of str or None
        SSPs to plot. Defaults to sorted keys of comp_proj.
    xlim : tuple of (float, float)
        X-axis limits.
    ipcc_data : dict or None
        ``{ssp: ipcc_component_dict}`` for IPCC comparison (dashed lines).
    ipcc_key : str or None
        Key name for labelling IPCC data.
    obs_years, obs_vals, obs_sigma : ndarray or None
        Observational data to overlay on the upper panel (meters).
    obs_label : str or None
        Label for observational data.
    units : str
        Display units: 'm' (default), 'cm', or 'mm'.  All SLR data
        (comp_proj, obs_vals, obs_sigma) are assumed to be in meters
        and are scaled accordingly.
    save_path : str or None
    """
    has_temp = temperature_scenarios is not None
    n_panels = 2 if has_temp else 1
    height_ratios = [2, 1] if has_temp else [1]

    # Unit conversion: data is in meters
    _unit_scale = {'m': 1.0, 'cm': 100.0, 'mm': M_TO_MM}
    scale = _unit_scale.get(units, 1.0)

    fig = plt.figure(figsize=(10, 5 + 2.5 * has_temp))
    gs = gridspec.GridSpec(n_panels, 1, height_ratios=height_ratios, hspace=0.28)
    ax_sl = fig.add_subplot(gs[0])

    if ssps is None:
        ssps = sorted(comp_proj.keys())

    proj_mask = (proj_years >= xlim[0]) & (proj_years <= xlim[1])
    yr_plot = proj_years[proj_mask]

    # Observations
    if obs_years is not None and obs_vals is not None:
        ax_sl.plot(obs_years, obs_vals * scale, color='#444444', lw=2,
                   label=obs_label or 'Observed', zorder=5)
        if obs_sigma is not None:
            ax_sl.fill_between(obs_years,
                               (obs_vals - Z_90 * obs_sigma) * scale,
                               (obs_vals + Z_90 * obs_sigma) * scale,
                               color='#444444', alpha=0.12, zorder=4)

    # Projections per SSP
    for ssp in ssps:
        if ssp not in comp_proj:
            continue
        p = comp_proj[ssp]
        color = SSP_COLORS.get(ssp, 'gray')
        med = p['median'][proj_mask] * scale
        lo17 = p['p17'][proj_mask] * scale
        hi83 = p['p83'][proj_mask] * scale
        lo5 = p['p5'][proj_mask] * scale
        hi95 = p['p95'][proj_mask] * scale

        ax_sl.plot(yr_plot, med, color=color, lw=2, label=ssp)
        ax_sl.fill_between(yr_plot, lo17, hi83, color=color, alpha=0.20)
        ax_sl.fill_between(yr_plot, lo5, hi95, color=color, alpha=0.08)

        # IPCC overlay
        if ipcc_data is not None and ssp in ipcc_data:
            ipcc_d = ipcc_data[ssp]
            slc = ipcc_d['slc'].squeeze()  # drop trailing dim if present
            q50_idx = np.argmin(np.abs(ipcc_d['quantiles'] - 0.5))
            q05_idx = np.argmin(np.abs(ipcc_d['quantiles'] - 0.05))
            q95_idx = np.argmin(np.abs(ipcc_d['quantiles'] - 0.95))
            ax_sl.plot(ipcc_d['years'], slc[q50_idx],
                       color=color, lw=1.5, ls='--', alpha=0.7)
            ax_sl.fill_between(ipcc_d['years'],
                               slc[q05_idx], slc[q95_idx],
                               color=color, alpha=0.05)

    ax_sl.set_ylabel(f'{component_name} SLR ({units})')
    ax_sl.set_title(f'{component_name} — Projections')
    ax_sl.legend(fontsize=8, loc='upper left', ncol=2)
    ax_sl.axhline(0, color='k', lw=0.5, ls='--')
    ax_sl.set_xlim(*xlim)
    ax_sl.grid(True, alpha=0.2)

    # Lower panel: temperature
    if has_temp:
        ax_t = fig.add_subplot(gs[1], sharex=ax_sl)
        for ssp in ssps:
            if ssp not in temperature_scenarios:
                continue
            ts = temperature_scenarios[ssp]
            color = SSP_COLORS.get(ssp, 'gray')
            ax_t.plot(ts['years'], ts['temperature'], color=color, lw=1.5)
        if temp_obs_years is not None and temp_obs_vals is not None:
            ax_t.plot(temp_obs_years, temp_obs_vals, color='#444444', lw=2,
                      label=temp_obs_label or 'Observed', zorder=5)
            ax_t.legend(fontsize=8, loc='upper left')
        ax_t.set_ylabel(temp_label)
        ax_t.set_xlabel('Year')
        ax_t.axhline(0, color='k', lw=0.5, ls='--')
        ax_t.grid(True, alpha=0.2)
    else:
        ax_sl.set_xlabel('Year')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# =========================================================================
# A4 scenario decomposition plot
# =========================================================================

def plot_a4_scenario_pdfs(scenario_samples, scenario_labels, scenario_colors,
                           mixture_samples=None, component_name='WAIS',
                           year=2100, xlabel=None, xlim=None, save_path=None):
    """Two-panel figure: per-scenario PDFs (left) and mixture (right).

    Parameters
    ----------
    scenario_samples : dict
        ``{scenario_name: ndarray}`` — MC endpoint samples (meters) per scenario.
    scenario_labels : dict
        ``{scenario_name: str}`` — legend label per scenario.
    scenario_colors : dict
        ``{scenario_name: str}`` — color per scenario.
    mixture_samples : ndarray or None
        Mixture endpoint samples (meters).  If None, panel B is omitted.
    """
    n_panels = 2 if mixture_samples is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    if xlabel is None:
        xlabel = f'{component_name} SLR at {year} (m)'

    if xlim is not None:
        x_lo, x_hi = xlim
    else:
        all_s = np.concatenate(list(scenario_samples.values()))
        x_lo = np.percentile(all_s, 0.1) - 0.01
        x_hi = np.percentile(all_s, 99.9) + 0.01
    x_grid = np.linspace(x_lo, x_hi, 600)
    dx = x_grid[1] - x_grid[0]

    # Panel A: per-scenario
    ax = axes[0]
    for sname in scenario_samples:
        s = scenario_samples[sname]
        kde = gaussian_kde(s, bw_method='scott')
        prob = kde(x_grid) * dx * 100
        ax.plot(x_grid, prob, lw=2, color=scenario_colors[sname],
                label=scenario_labels[sname])
        ax.fill_between(x_grid, prob, alpha=0.15, color=scenario_colors[sname])
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability (%)')
    ax.set_title(f'A4 Scenario PDFs at {year}')
    ax.legend(fontsize=7)
    ax.set_xlim(x_lo, x_hi)
    ax.grid(True, alpha=0.2)

    # Panel B: mixture
    if mixture_samples is not None:
        ax = axes[1]
        kde_mix = gaussian_kde(mixture_samples, bw_method='scott')
        prob_mix = kde_mix(x_grid) * dx * 100
        ax.plot(x_grid, prob_mix, 'k-', lw=2, label='A4 mixture')
        ax.fill_between(x_grid, prob_mix, alpha=0.3, color='gray')
        med = np.median(mixture_samples)
        ax.axvline(med, color='k', ls='--', lw=1, label=f'Median: {med:.2f} m')
        p5, p95 = np.percentile(mixture_samples, [5, 95])
        ax.axvline(p5, color='gray', ls=':', lw=1)
        ax.axvline(p95, color='gray', ls=':', lw=1,
                   label=f'90% CI: [{p5:.2f}, {p95:.2f}] m')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Probability (%)')
        ax.set_title(f'A4 Mixture Distribution')
        ax.legend(fontsize=8)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# =========================================================================
# Component histogram / KDE overlay
# =========================================================================

def plot_component_histogram(sample_sets, labels, colors, component_name,
                              year=2100, title=None, xlabel=None, ylabel=None,
                              xlim=None, probability=False, fontsize=None,
                              save_path=None):
    """KDE overlays at a user-specified year for multiple projection sources.

    Parameters
    ----------
    sample_sets : list of ndarray
        Each array contains MC samples at the target year (mm).
    labels : list of str
        Label for each sample set.
    colors : list of str
        Color for each sample set.
    component_name : str
        Component name for title.
    year : int
        Target year (for title only).
    xlabel : str or None
        X-axis label. Defaults to '{component_name} SLR at {year} (mm)'.
    xlim : tuple or None
        X-axis limits. Auto-determined if None.
    probability : bool
        If True, scale the KDE by dx so the y-axis shows probability
        (fraction of samples per bin) instead of density.
    save_path : str or None
    """
    # Font sizes: accept int (uniform) or dict with keys title/xlabel/ylabel/legend/xtick/ytick
    _fs_defaults = {'title': None, 'xlabel': None, 'ylabel': None,
                    'legend': 9, 'xtick': None, 'ytick': None}
    if fontsize is None:
        _fs = _fs_defaults
    elif isinstance(fontsize, (int, float)):
        _fs = {k: fontsize for k in _fs_defaults}
    else:
        _fs = {**_fs_defaults, **fontsize}

    fig, ax = plt.subplots(figsize=(8, 5))

    if xlabel is None:
        xlabel = f'{component_name} SLR at {year} (mm)'

    # Determine shared x range — use xlim if provided so KDE covers
    # the full requested range; otherwise auto from data percentiles.
    all_vals = np.concatenate([s for s in sample_sets if len(s) > 0])
    if xlim is not None:
        x_lo, x_hi = xlim
    else:
        x_lo = np.percentile(all_vals, 0.5) - 5
        x_hi = np.percentile(all_vals, 99.5) + 5
    x_grid = np.linspace(x_lo, x_hi, 300)

    dx = x_grid[1] - x_grid[0]

    for samples, label, color in zip(sample_sets, labels, colors):
        if len(samples) < 10:
            continue
        kde = gaussian_kde(samples, bw_method='scott')
        y = kde(x_grid)
        if probability:
            y = y * dx * 100
        ax.fill_between(x_grid, 0, y, color=color, alpha=0.35)
        ax.plot(x_grid, y, color=color, lw=2, label=label)
        # Median line
        med = np.median(samples)
        ax.axvline(med, color=color, ls='--', lw=1, alpha=0.7)

    ax.set_xlabel(xlabel, fontsize=_fs['xlabel'])
    _ylabel = ylabel if ylabel is not None else ('Probability (%)' if probability else 'Probability density')
    ax.set_ylabel(_ylabel, fontsize=_fs['ylabel'])
    _title = title if title is not None else f'{component_name} — Distribution at {year}'
    ax.set_title(_title, fontsize=_fs['title'])
    ax.legend(fontsize=_fs['legend'])
    if _fs['xtick'] is not None:
        ax.tick_params(axis='x', labelsize=_fs['xtick'])
    if _fs['ytick'] is not None:
        ax.tick_params(axis='y', labelsize=_fs['ytick'])
    ax.set_ylim(bottom=0)
    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(x_lo, x_hi)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# =========================================================================
# Component PDF + exceedance plot
# =========================================================================

def plot_component_pdf_exceedance(sample_sets, labels, colors, component_name,
                                   year=2100, xlabel=None, xlim=None,
                                   save_path=None):
    """Two-panel figure: PDF (left) and exceedance probability (right).

    Parameters
    ----------
    sample_sets : list of ndarray
        Each array contains MC samples at the target year (mm).
    labels : list of str
    colors : list of str
    component_name : str
    year : int
    xlabel : str or None
    xlim : tuple or None
        View limits for x-axis. Does not affect the calculation.
    save_path : str or None
    """
    fig, (ax_pdf, ax_exc) = plt.subplots(1, 2, figsize=(14, 5))

    if xlabel is None:
        xlabel = f'{component_name} SLR at {year} (mm)'

    # KDE x range: extend well beyond the data so the PDF tails are
    # fully resolved and the CDF integrates to ~1.  xlim only controls
    # the view window.
    all_vals = np.concatenate([s for s in sample_sets if len(s) > 0])
    data_range = np.percentile(all_vals, 99.9) - np.percentile(all_vals, 0.1)
    x_lo = np.percentile(all_vals, 0.1) - 0.5 * data_range
    x_hi = np.percentile(all_vals, 99.9) + 0.5 * data_range
    x_grid = np.linspace(x_lo, x_hi, 1000)
    dx = x_grid[1] - x_grid[0]

    for samples, label, color in zip(sample_sets, labels, colors):
        if len(samples) < 10:
            continue
        kde = gaussian_kde(samples, bw_method='scott')
        density = kde(x_grid)

        # PDF panel (probability %)
        prob = density * dx * 100
        ax_pdf.fill_between(x_grid, 0, prob, color=color, alpha=0.35)
        ax_pdf.plot(x_grid, prob, color=color, lw=2, label=label)
        med = np.median(samples)
        ax_pdf.axvline(med, color=color, ls='--', lw=1, alpha=0.7)

        # Exceedance = 1 - CDF, where CDF = integral of PDF from -inf to x
        cdf = np.cumsum(density) * dx
        ax_exc.plot(x_grid, (1.0 - cdf) * 100, color=color, lw=2, label=label)

    # PDF formatting
    ax_pdf.set_xlabel(xlabel)
    ax_pdf.set_ylabel('Probability (%)')
    ax_pdf.set_title(f'{component_name} — Distribution at {year}')
    ax_pdf.legend(fontsize=9)
    ax_pdf.set_ylim(bottom=0)
    ax_pdf.grid(True, alpha=0.2)

    # Exceedance formatting
    ax_exc.set_xlabel(xlabel)
    ax_exc.set_ylabel('P(exceedance) (%)')
    ax_exc.set_title(f'{component_name} — Exceedance at {year}')
    ax_exc.legend(fontsize=9)
    ax_exc.set_ylim(0, 100)
    ax_exc.grid(True, alpha=0.2)

    # View limits
    if xlim is not None:
        ax_pdf.set_xlim(*xlim)
        ax_exc.set_xlim(*xlim)
    else:
        view_lo = np.percentile(all_vals, 0.5) - 5
        view_hi = np.percentile(all_vals, 99.5) + 5
        ax_pdf.set_xlim(view_lo, view_hi)
        ax_exc.set_xlim(view_lo, view_hi)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# =========================================================================
# Component ridge plot (density evolution across decades)
# =========================================================================

def plot_component_ridge(samples_by_year, component_name, ssp_label,
                          years=None, source_labels=None, legend_labels=None,
                          source_colors=None, bw_factor=0.9,
                          xlabel=None, xlim=None, title=None,
                          legend_loc=None, legend_bbox=None,
                          top=0.90, fontsize=None,
                          units='m', hspace=-0.4, show_median=True,
                          show_impact_pop=False, show_impact_cost=False,
                          impact_spacing=25, figsize=None, save_path=None):
    """Ridge plot showing density evolution across decades for one or more
    projection sources at a single SSP.

    Parameters
    ----------
    samples_by_year : dict
        ``{year: {source_label: ndarray}}`` — MC samples in **meters** for
        each (year, source) combination.  Converted to display units via
        the *units* parameter.
    component_name : str
        Component name for title.
    ssp_label : str
        SSP name for title.
    years : list of int or None
        Years to plot, in order (bottom to top). Defaults to sorted keys.
    source_labels : list of str or None
        Source names. Defaults to keys of first year entry.
    source_colors : dict or None
        ``{source_label: color}``. Auto-assigned if None.
    bw_factor : float
        Bandwidth scaling factor for KDE.
    xlabel : str or None
        X-axis label. Defaults to '{component_name} SLR ({units})'.
    units : str
        Display units: 'm' (default), 'cm', or 'mm'.  Samples are assumed
        to be in meters and are scaled accordingly.
    save_path : str or None
    """
    if years is None:
        years = sorted(samples_by_year.keys())
    if source_labels is None:
        source_labels = list(samples_by_year[years[0]].keys())
    if source_colors is None:
        default_colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple']
        source_colors = {s: default_colors[i % len(default_colors)]
                         for i, s in enumerate(source_labels)}
    if xlabel is None:
        xlabel = f'{component_name} SLR ({units})'

    # Font sizes: accept int (uniform) or dict with keys year/title/legend/xlabel
    _fs_defaults = {'year': 10, 'title': 12, 'legend': 8, 'xlabel': None, 'xtick': None}
    if fontsize is None:
        _fs = _fs_defaults
    elif isinstance(fontsize, (int, float)):
        _fs = {k: fontsize for k in _fs_defaults}
    else:
        _fs = {**_fs_defaults, **fontsize}

    # Unit conversion: samples assumed to be in meters
    _unit_scale = {'m': 1.0, 'cm': 100.0, 'mm': M_TO_MM}
    scale = _unit_scale.get(units, 1.0)

    n_yrs = len(years)

    # Determine shared x range
    if xlim is not None:
        x_lo, x_hi = xlim
    else:
        all_vals = np.concatenate([np.asarray(samples_by_year[yr][src]) * scale
                                   for yr in years for src in source_labels
                                   if src in samples_by_year.get(yr, {})])
        x_lo = float(np.percentile(all_vals, 0.5)) - 5
        x_hi = float(np.percentile(all_vals, 99.5)) + 5
    x_grid = np.linspace(x_lo, x_hi, 400)

    # Precompute KDEs
    kde_data = {}
    for yr in years:
        kde_data[yr] = {}
        for src in source_labels:
            if src not in samples_by_year.get(yr, {}):
                continue
            vals = np.asarray(samples_by_year[yr][src]) * scale
            if len(vals) < 10:
                continue
            kde = gaussian_kde(vals, bw_method='scott')
            kde.set_bandwidth(kde.factor * bw_factor)
            kde_data[yr][src] = kde(x_grid)

    _figsize = figsize if figsize is not None else (8, n_yrs * 0.6 + 1)
    fig, axes = plt.subplots(n_yrs, 1, figsize=_figsize,
                              sharex=True)
    if n_yrs == 1:
        axes = [axes]

    # Remove auto x-margins so ridge axes xlim matches impact axes exactly
    for ax in axes:
        ax.margins(x=0)
    axes[-1].set_xlim(x_lo, x_hi)

    for i, yr in enumerate(years):
        ax = axes[i]
        ax.set_facecolor((0, 0, 0, 0))
        for src in source_labels:
            if src not in kde_data.get(yr, {}):
                continue
            density = kde_data[yr][src]
            color = source_colors.get(src, 'gray')
            ax.fill_between(x_grid, density, alpha=0.4, color=color,
                            clip_on=False)
            ax.plot(x_grid, density, color=color, lw=1.0, clip_on=False)
            if show_median:
                vals = np.asarray(samples_by_year[yr][src]) * scale
                med = float(np.median(vals))
                med_height = float(np.interp(med, x_grid, density))
                ax.plot([med, med], [0, med_height], ls='--', lw=0.8,
                        color=color, alpha=0.7, clip_on=False)
        ax.text(0.0, 0.2, str(yr), fontweight='bold', color='0.5',
                ha='left', va='center', transform=ax.transAxes, fontsize=_fs['year'])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_color('#666666')
        ax.tick_params(axis='x', length=3, pad=4, colors='#666666',
                       labelcolor='#666666')

    # Perspective scaling (later years = taller)
    for i, ax in enumerate(axes):
        p = i / max(n_yrs - 1, 1)
        perspective = 0.25 + 0.75 * p
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(0, ymax / perspective)

    # xlabel only on the bottom panel
    axes[-1].set_xlabel(xlabel, fontsize=_fs['xlabel'], color='#666666')
    if _fs['xtick'] is not None:
        for ax in axes:
            ax.tick_params(axis='x', labelsize=_fs['xtick'])
    _title = title if title is not None else f'{component_name} — Density evolution ({ssp_label})'
    fig.suptitle(_title, fontsize=_fs['title'], fontweight='bold', y=0.98)

    # Legend
    from matplotlib.patches import Patch
    _legend_names = legend_labels if legend_labels is not None else source_labels
    legend_elements = [Patch(facecolor=source_colors.get(s, 'gray'),
                             alpha=0.4, label=leg)
                       for s, leg in zip(source_labels, _legend_names)]
    _loc = legend_loc if legend_loc is not None else 'upper right'
    _legend_kw = dict(handles=legend_elements, fontsize=_fs['legend'], loc=_loc,
                      framealpha=1.0, edgecolor='0.8')
    if legend_bbox is not None:
        _legend_kw['bbox_to_anchor'] = legend_bbox
    axes[0].legend(**_legend_kw)

    # --- Impact axes on the bottom panel ---
    # Input samples are in meters; impact functions expect meters.
    # xlim is in display units, so convert back to meters for the functions.
    if show_impact_pop or show_impact_cost:
        from slr_data_readers import people_displaced_kulpstrauss2019, slr_cost_jevrejeva2018
        _xlim_m = (x_lo / scale, x_hi / scale)  # display units -> meters
        n_impact_ticks = 6
        _tick_slr_display = np.linspace(x_lo, x_hi, n_impact_ticks)
        _tick_slr_m = _tick_slr_display / scale

    # Impact axes use smaller fonts to stay compact
    _fs_impact_tick = 7
    _fs_impact_label = 8

    _bottom = 0.08
    if show_impact_pop:
        _pop_baseline = people_displaced_kulpstrauss2019(0.0)
        _pop_targets = np.array([people_displaced_kulpstrauss2019(v) - _pop_baseline
                                 for v in _tick_slr_m])
        ax_pop = axes[-1].twiny()
        ax_pop.xaxis.set_ticks_position('bottom')
        ax_pop.xaxis.set_label_position('bottom')
        for sp in ax_pop.spines.values():
            sp.set_visible(False)
        ax_pop.spines['bottom'].set_visible(True)
        ax_pop.spines['bottom'].set_position(('outward', impact_spacing))
        ax_pop.spines['bottom'].set_color('#666666')
        ax_pop.set_xlim(x_lo, x_hi)
        ax_pop.set_xticks(_tick_slr_display)
        ax_pop.set_xticklabels([f'{int(10 * round(v / 10))}' for v in _pop_targets],
                               color='#666666', fontsize=_fs_impact_tick)
        ax_pop.set_xlabel('Additional People on Land Below Flood Level [Millions]',
                          color='#666666', fontsize=_fs_impact_label)
        ax_pop.tick_params(bottom=True, top=False, length=3, pad=4, colors='#666666')
        _bottom = 0.10

    if show_impact_cost:
        _cost_targets = np.array([1e-3 * slr_cost_jevrejeva2018(v)
                                  for v in _tick_slr_m])
        ax_cost = axes[-1].twiny()
        ax_cost.xaxis.set_ticks_position('bottom')
        ax_cost.xaxis.set_label_position('bottom')
        for sp in ax_cost.spines.values():
            sp.set_visible(False)
        _offset_cost = 2 * impact_spacing if show_impact_pop else impact_spacing
        ax_cost.spines['bottom'].set_visible(True)
        ax_cost.spines['bottom'].set_position(('outward', _offset_cost))
        ax_cost.spines['bottom'].set_color('#999999')
        ax_cost.set_xlim(x_lo, x_hi)
        ax_cost.set_xticks(_tick_slr_display)
        ax_cost.set_xticklabels([f'{int(round(v))}' for v in _cost_targets],
                                color='#999999', fontsize=_fs_impact_tick)
        ax_cost.set_xlabel('Global Annual Flood Costs [Trillions US\\$]',
                           color='#999999', fontsize=_fs_impact_label)
        ax_cost.tick_params(bottom=True, top=False, length=3, pad=4, colors='#999999')
        _bottom = 0.12

    fig.subplots_adjust(hspace=hspace, bottom=_bottom, top=top)

    # Align edge tick labels inward so they don't protrude past the axes,
    # preventing bbox_inches='tight' from cropping asymmetrically.
    fig.canvas.draw()
    _axes_with_ticks = [axes[-1]]
    if show_impact_pop:
        _axes_with_ticks.append(ax_pop)
    if show_impact_cost:
        _axes_with_ticks.append(ax_cost)
    for _ax in _axes_with_ticks:
        _ticks = _ax.xaxis.get_major_ticks()
        if len(_ticks) >= 2:
            _ticks[0].label1.set_ha('left')
            _ticks[-1].label1.set_ha('right')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# =========================================================================
# Component rates bar chart
# =========================================================================

def plot_component_rates(components_rate, save_path=None):
    """Bar chart + pie chart of component rates (mm/yr).

    Parameters
    ----------
    components_rate : dict
        ``{name: rate_mm_yr}``.
    save_path : str or None
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    comp_names = ['Steric', 'Glaciers', 'Greenland', 'Antarctica', 'TWS']
    comp_rates = [components_rate[n] for n in comp_names]
    comp_colors = [ARETE_COLORS['greys'][1], ARETE_COLORS['blues'][1],
                   ARETE_COLORS['blues'][3], ARETE_COLORS['blues'][5],
                   ARETE_COLORS['brown'][0]]

    ax = axes[0]
    bars = ax.barh(comp_names, comp_rates, color=comp_colors,
                   edgecolor='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel('Rate (mm/yr), 2002-2018')
    ax.set_title('Component Rates (Frederikse Budget)')
    for bar, rate in zip(bars, comp_rates):
        ax.text(rate + 0.05, bar.get_y() + bar.get_height() / 2,
                f'{rate:.2f}', va='center', fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')

    ax = axes[1]
    positive_rates = {n: max(r, 0) for n, r in zip(comp_names, comp_rates)}
    total_pos = sum(positive_rates.values())
    sizes = [positive_rates[n] / total_pos * 100 for n in comp_names]
    ax.pie(sizes, labels=comp_names, autopct='%1.0f%%',
           colors=comp_colors, startangle=90)
    ax.set_title('Percentage of SLR by Component\n(2002-2018, positive contributions)')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
