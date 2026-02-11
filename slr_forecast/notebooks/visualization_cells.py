# =============================================================================
# VISUALIZATION CELLS FOR GMSL ANALYSIS
# =============================================================================
# Copy these cells into slr_analysis_notebook.ipynb
# =============================================================================


# %% [markdown]
# ---
# ## 8. Thermodynamic Signal Analysis
# 
# Compute the thermodynamic signal (climate-driven sea level change):
# 
# $$\text{Thermodynamic} = \text{Steric} + \text{Barystatic} - \text{TWS}$$
# 
# where Barystatic = Glaciers + Greenland + Antarctica
# 
# **Note:** Additional datasets for thermosteric and TWS constraints to be incorporated later.


# %% [code]
# -----------------------------------------------------------------------------
# Compute thermodynamic signal for Frederikse and Dangendorf
# -----------------------------------------------------------------------------

from slr_analysis import compute_thermodynamic_signal

# Frederikse: has full budget decomposition
df_frederikse_thermo = compute_thermodynamic_signal(df_frederikse)
print("Frederikse thermodynamic signal computed")
print(f"  Components: {[c for c in df_frederikse_thermo.columns if not c.endswith('_sigma')]}")

# Dangendorf: thermodynamic = steric + barystatic (TWS already excluded from barystatic)
df_dangendorf_thermo = pd.DataFrame({
    'thermodynamic': df_dangendorf['steric'] + df_dangendorf['barystatic'],
    'thermodynamic_sigma': np.sqrt(df_dangendorf['steric_sigma']**2 + df_dangendorf['barystatic_sigma']**2),
    'steric': df_dangendorf['steric'],
    'steric_sigma': df_dangendorf['steric_sigma'],
    'barystatic': df_dangendorf['barystatic'],
    'barystatic_sigma': df_dangendorf['barystatic_sigma'],
}, index=df_dangendorf.index)
df_dangendorf_thermo['decimal_year'] = df_dangendorf['decimal_year']
print("\nDangendorf thermodynamic signal computed")
print(f"  Components: steric, barystatic (TWS not available separately)")

# Harmonize thermodynamic signals to baseline
df_frederikse_thermo_h = harmonize_baseline(df_frederikse_thermo, 'thermodynamic', BASELINE)
df_dangendorf_thermo_h = harmonize_baseline(df_dangendorf_thermo, 'thermodynamic', BASELINE)


# %% [code]
# -----------------------------------------------------------------------------
# Compute kinematics for thermodynamic signal and components
# -----------------------------------------------------------------------------

# Frederikse thermodynamic and components
thermo_kinematics = {}

thermo_kinematics['frederikse_thermodynamic'] = compute_kinematics(
    time=get_decimal_year(df_frederikse_thermo),
    value=df_frederikse_thermo['thermodynamic'].values,
    sigma=df_frederikse_thermo['thermodynamic_sigma'].values,
    span_years=SPAN_YEARS
)

thermo_kinematics['frederikse_steric'] = compute_kinematics(
    time=get_decimal_year(df_frederikse_thermo),
    value=df_frederikse_thermo['steric'].values,
    sigma=df_frederikse_thermo['steric_sigma'].values,
    span_years=SPAN_YEARS
)

thermo_kinematics['frederikse_barystatic'] = compute_kinematics(
    time=get_decimal_year(df_frederikse_thermo),
    value=df_frederikse_thermo['barystatic'].values,
    sigma=df_frederikse_thermo['barystatic_sigma'].values,
    span_years=SPAN_YEARS
)

thermo_kinematics['frederikse_glaciers'] = compute_kinematics(
    time=get_decimal_year(df_frederikse_thermo),
    value=df_frederikse_thermo['glaciers'].values,
    sigma=df_frederikse_thermo['glaciers_sigma'].values,
    span_years=SPAN_YEARS
)

thermo_kinematics['frederikse_greenland'] = compute_kinematics(
    time=get_decimal_year(df_frederikse_thermo),
    value=df_frederikse_thermo['greenland'].values,
    sigma=df_frederikse_thermo['greenland_sigma'].values,
    span_years=SPAN_YEARS
)

thermo_kinematics['frederikse_antarctica'] = compute_kinematics(
    time=get_decimal_year(df_frederikse_thermo),
    value=df_frederikse_thermo['antarctica'].values,
    sigma=df_frederikse_thermo['antarctica_sigma'].values,
    span_years=SPAN_YEARS
)

# Dangendorf thermodynamic and components
thermo_kinematics['dangendorf_thermodynamic'] = compute_kinematics(
    time=get_decimal_year(df_dangendorf_thermo),
    value=df_dangendorf_thermo['thermodynamic'].values,
    sigma=df_dangendorf_thermo['thermodynamic_sigma'].values,
    span_years=SPAN_YEARS
)

thermo_kinematics['dangendorf_steric'] = compute_kinematics(
    time=get_decimal_year(df_dangendorf_thermo),
    value=df_dangendorf_thermo['steric'].values,
    sigma=df_dangendorf_thermo['steric_sigma'].values,
    span_years=SPAN_YEARS
)

thermo_kinematics['dangendorf_barystatic'] = compute_kinematics(
    time=get_decimal_year(df_dangendorf_thermo),
    value=df_dangendorf_thermo['barystatic'].values,
    sigma=df_dangendorf_thermo['barystatic_sigma'].values,
    span_years=SPAN_YEARS
)

# Convert to DataFrames
thermo_kinematics_dfs = {name: res.to_dataframe() for name, res in thermo_kinematics.items()}

print(f"Thermodynamic kinematics computed for {len(thermo_kinematics)} components")


# %% [markdown]
# ---
# ## 9. Visualization
# 
# ### 9.1 Main Overview: GMSL, Rates, and Accelerations


# %% [code]
# -----------------------------------------------------------------------------
# PLOT 1: Main overview (3 subplots: GMSL, Rate, Acceleration)
# -----------------------------------------------------------------------------

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

colors = {
    'frederikse': 'C0',
    'nasa': 'C1', 
    'horwath': 'C2',
    'ipcc': 'C3',
    'dangendorf': 'C4'
}

# Get original dataframes for each dataset
orig_data = {
    'frederikse': df_frederikse,
    'nasa': df_nasa_gmsl,
    'horwath': df_horwath_valid,
    'ipcc': df_ipcc_observed_gmsl,
    'dangendorf': df_dangendorf
}

# ax1: GMSL data
for name, res in kinematics.items():
    df = kinematics_dfs[name]
    orig_df = orig_data[name]
    valid = ~np.isnan(df['rate'])
    
    ax1.plot(df['decimal_year'], orig_df['gmsl'].values * 1000, 
             '-', color=colors[name], alpha=0.7, linewidth=1.5, label=name)

ax1.set_ylabel('GMSL (mm)')
ax1.set_title(f'Global Mean Sea Level and Kinematics ({SPAN_YEARS}-year bandwidth)')
ax1.legend(loc='upper left', ncol=2)
ax1.grid(True, alpha=0.3)

# ax2: Rates
for name, res in kinematics.items():
    df = kinematics_dfs[name]
    valid = ~np.isnan(df['rate'])
    rate_mm = df['rate'] * 1000
    rate_se_mm = df['rate_se'] * 1000
    
    ax2.plot(df.loc[valid, 'decimal_year'], rate_mm[valid], 
             '-', color=colors[name], linewidth=2, label=name)
    ax2.fill_between(df.loc[valid, 'decimal_year'], 
                     (rate_mm - 1.96*rate_se_mm)[valid],
                     (rate_mm + 1.96*rate_se_mm)[valid],
                     color=colors[name], alpha=0.2)

ax2.set_ylabel('Rate (mm/yr)')
ax2.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax2.legend(loc='upper left', ncol=2)
ax2.grid(True, alpha=0.3)

# ax3: Accelerations
for name, res in kinematics.items():
    df = kinematics_dfs[name]
    valid = ~np.isnan(df['accel'])
    accel_mm = df['accel'] * 1000
    accel_se_mm = df['accel_se'] * 1000
    
    ax3.plot(df.loc[valid, 'decimal_year'], accel_mm[valid],
             '-', color=colors[name], linewidth=2, label=name)
    ax3.fill_between(df.loc[valid, 'decimal_year'],
                     (accel_mm - 1.96*accel_se_mm)[valid],
                     (accel_mm + 1.96*accel_se_mm)[valid],
                     color=colors[name], alpha=0.2)

ax3.set_ylabel('Acceleration (mm/yr²)')
ax3.set_xlabel('Year')
ax3.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax3.legend(loc='upper left', ncol=2)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/gmsl_kinematics_overview.png', dpi=150, bbox_inches='tight')
plt.show()


# %% [markdown]
# ### 9.2 Faceted Small Multiples (by dataset)


# %% [code]
# -----------------------------------------------------------------------------
# PLOT 2: Faceted by dataset (small multiples)
# -----------------------------------------------------------------------------

n_datasets = len(kinematics)
fig, axes = plt.subplots(n_datasets, 3, figsize=(16, 3*n_datasets), sharex=True)

for i, (name, res) in enumerate(kinematics.items()):
    df = kinematics_dfs[name]
    orig_df = orig_data[name]
    valid = ~np.isnan(df['rate'])
    color = colors[name]
    
    # Column 1: GMSL
    axes[i, 0].plot(df['decimal_year'], orig_df['gmsl'].values * 1000, 
                    '-', color=color, linewidth=1.5)
    axes[i, 0].set_ylabel(f'{name}\nGMSL (mm)')
    axes[i, 0].grid(True, alpha=0.3)
    
    # Column 2: Rate
    rate_mm = df['rate'] * 1000
    rate_se_mm = df['rate_se'] * 1000
    axes[i, 1].plot(df.loc[valid, 'decimal_year'], rate_mm[valid], 
                    '-', color=color, linewidth=2)
    axes[i, 1].fill_between(df.loc[valid, 'decimal_year'],
                            (rate_mm - 1.96*rate_se_mm)[valid],
                            (rate_mm + 1.96*rate_se_mm)[valid],
                            color=color, alpha=0.2)
    axes[i, 1].axhline(0, color='k', linestyle='--', linewidth=0.5)
    axes[i, 1].set_ylabel('Rate (mm/yr)')
    axes[i, 1].grid(True, alpha=0.3)
    
    # Column 3: Acceleration
    accel_mm = df['accel'] * 1000
    accel_se_mm = df['accel_se'] * 1000
    axes[i, 2].plot(df.loc[valid, 'decimal_year'], accel_mm[valid],
                    '-', color=color, linewidth=2)
    axes[i, 2].fill_between(df.loc[valid, 'decimal_year'],
                            (accel_mm - 1.96*accel_se_mm)[valid],
                            (accel_mm + 1.96*accel_se_mm)[valid],
                            color=color, alpha=0.2)
    axes[i, 2].axhline(0, color='k', linestyle='--', linewidth=0.5)
    axes[i, 2].set_ylabel('Accel (mm/yr²)')
    axes[i, 2].grid(True, alpha=0.3)

# Column titles
axes[0, 0].set_title('Sea Level')
axes[0, 1].set_title('Rate of Change')
axes[0, 2].set_title('Acceleration')

# X-axis labels
for j in range(3):
    axes[-1, j].set_xlabel('Year')

plt.tight_layout()
plt.savefig('figures/gmsl_kinematics_faceted.png', dpi=150, bbox_inches='tight')
plt.show()


# %% [markdown]
# ### 9.3 Rate vs. Temperature (Phase Plot)
# 
# The slope of this relationship approximates the climate sensitivity parameter α.


# %% [code]
# -----------------------------------------------------------------------------
# PLOT 2: Rate vs Temperature (phase plot)
# -----------------------------------------------------------------------------

# Resample Berkeley Earth to annual for matching
df_berkeley_annual = df_berkeley_h.resample('Y').mean()
df_berkeley_annual['decimal_year'] = df_berkeley_annual.index.year + 0.5

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, (name, res) in enumerate(kinematics.items()):
    if i >= 6:
        break
    
    df_kin = kinematics_dfs[name]
    valid = ~np.isnan(df_kin['rate'])
    
    # Match time points with temperature
    kin_years = df_kin.loc[valid, 'decimal_year'].values
    rate_mm = df_kin.loc[valid, 'rate'].values * 1000
    
    # Find matching temperature values
    temp_matched = []
    rate_matched = []
    time_matched = []
    
    for year, rate in zip(kin_years, rate_mm):
        # Find closest temperature year
        temp_idx = np.argmin(np.abs(df_berkeley_annual['decimal_year'].values - year))
        if np.abs(df_berkeley_annual['decimal_year'].values[temp_idx] - year) < 1.0:
            temp_matched.append(df_berkeley_annual['temperature'].values[temp_idx])
            rate_matched.append(rate)
            time_matched.append(year)
    
    temp_matched = np.array(temp_matched)
    rate_matched = np.array(rate_matched)
    time_matched = np.array(time_matched)
    
    # Scatter plot colored by time
    sc = axes[i].scatter(temp_matched, rate_matched, c=time_matched, 
                         cmap='viridis', s=30, alpha=0.7)
    
    # Linear fit for α estimation
    if len(temp_matched) > 10:
        mask = ~(np.isnan(temp_matched) | np.isnan(rate_matched))
        if mask.sum() > 10:
            z = np.polyfit(temp_matched[mask], rate_matched[mask], 1)
            p = np.poly1d(z)
            temp_range = np.linspace(temp_matched[mask].min(), temp_matched[mask].max(), 100)
            axes[i].plot(temp_range, p(temp_range), 'r--', linewidth=2, 
                        label=f'α ≈ {z[0]:.1f} mm/yr/°C')
            axes[i].legend(loc='upper left')
    
    axes[i].set_xlabel('Temperature anomaly (°C)')
    axes[i].set_ylabel('GMSL rate (mm/yr)')
    axes[i].set_title(name)
    axes[i].grid(True, alpha=0.3)

# Remove empty subplot if only 5 datasets
if len(kinematics) < 6:
    axes[-1].axis('off')

plt.colorbar(sc, ax=axes, label='Year', shrink=0.6)
plt.suptitle('GMSL Rate vs Temperature (Berkeley Earth)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('figures/rate_vs_temperature_phase.png', dpi=150, bbox_inches='tight')
plt.show()


# %% [markdown]
# ### 9.4 Acceleration Heatmap (Time × Dataset)


# %% [code]
# -----------------------------------------------------------------------------
# PLOT 4: Acceleration heatmap
# -----------------------------------------------------------------------------

# Configuration - adjust time resolution here
TIME_RESOLUTION_YEARS = 5  # Annual binning for heatmap

# Create common time grid
time_min = 1900
time_max = 2025
time_grid = np.arange(time_min + TIME_RESOLUTION_YEARS/2, 
                       time_max, 
                       TIME_RESOLUTION_YEARS)

# Resample all datasets to common grid (using bin averages, not interpolation)
dataset_names = list(kinematics.keys())
accel_matrix = np.full((len(dataset_names), len(time_grid)), np.nan)

for i, name in enumerate(dataset_names):
    df = kinematics_dfs[name]
    valid = ~np.isnan(df['accel'])
    years = df.loc[valid, 'decimal_year'].values
    accels = df.loc[valid, 'accel'].values * 1000  # mm/yr²
    
    for j, t in enumerate(time_grid):
        # Bin average
        mask = (years >= t - TIME_RESOLUTION_YEARS/2) & (years < t + TIME_RESOLUTION_YEARS/2)
        if mask.sum() > 0:
            accel_matrix[i, j] = np.mean(accels[mask])

# Plot heatmap
fig, ax = plt.subplots(figsize=(14, 6))

# Use diverging colormap centered at 0
vmax = np.nanmax(np.abs(accel_matrix))
im = ax.imshow(accel_matrix, aspect='auto', cmap='RdBu_r', 
               vmin=-vmax, vmax=vmax,
               extent=[time_grid[0]-TIME_RESOLUTION_YEARS/2, 
                       time_grid[-1]+TIME_RESOLUTION_YEARS/2, 
                       len(dataset_names)-0.5, -0.5])

ax.set_yticks(range(len(dataset_names)))
ax.set_yticklabels(dataset_names)
ax.set_xlabel('Year')
ax.set_title(f'GMSL Acceleration by Dataset ({TIME_RESOLUTION_YEARS}-year bins)')

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Acceleration (mm/yr²)')

# Add vertical lines for key dates
for year, label in [(1993, 'Altimetry'), (2002, 'GRACE')]:
    ax.axvline(year, color='k', linestyle='--', alpha=0.5)
    ax.text(year+1, -0.3, label, fontsize=9, alpha=0.7)

plt.tight_layout()
plt.savefig('figures/acceleration_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()


# %% [markdown]
# ### 9.5 Satellite Era Comparison (1993+)


# %% [code]
# -----------------------------------------------------------------------------
# PLOT 5: Satellite era rates with uncertainty bands only
# -----------------------------------------------------------------------------

SATELLITE_ERA_START = 1993

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

for name, res in kinematics.items():
    df = kinematics_dfs[name]
    valid = (~np.isnan(df['rate'])) & (df['decimal_year'] >= SATELLITE_ERA_START)
    
    if valid.sum() == 0:
        continue
    
    color = colors[name]
    years = df.loc[valid, 'decimal_year']
    
    # Rate
    rate_mm = df.loc[valid, 'rate'] * 1000
    rate_se_mm = df.loc[valid, 'rate_se'] * 1000
    ax1.plot(years, rate_mm, '-', color=color, linewidth=2, label=name)
    ax1.fill_between(years, rate_mm - 1.96*rate_se_mm, rate_mm + 1.96*rate_se_mm,
                     color=color, alpha=0.15)
    
    # Acceleration
    accel_mm = df.loc[valid, 'accel'] * 1000
    accel_se_mm = df.loc[valid, 'accel_se'] * 1000
    ax2.plot(years, accel_mm, '-', color=color, linewidth=2, label=name)
    ax2.fill_between(years, accel_mm - 1.96*accel_se_mm, accel_mm + 1.96*accel_se_mm,
                     color=color, alpha=0.15)

ax1.set_ylabel('Rate (mm/yr)')
ax1.set_title(f'Satellite Era GMSL Kinematics ({SATELLITE_ERA_START}–present, {SPAN_YEARS}-year bandwidth)')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

ax2.set_ylabel('Acceleration (mm/yr²)')
ax2.set_xlabel('Year')
ax2.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/satellite_era_kinematics.png', dpi=150, bbox_inches='tight')
plt.show()


# %% [markdown]
# ### 9.6 Cumulative vs Instantaneous (Dual Axis)


# %% [code]
# -----------------------------------------------------------------------------
# PLOT 6: Cumulative sea level vs instantaneous rate (dual axis)
# -----------------------------------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, (name, res) in enumerate(kinematics.items()):
    if i >= 6:
        break
    
    df = kinematics_dfs[name]
    orig_df = orig_data[name]
    valid = ~np.isnan(df['rate'])
    
    ax1 = axes[i]
    ax2 = ax1.twinx()
    
    # Left axis: cumulative sea level
    years = df['decimal_year'].values
    gmsl_mm = orig_df['gmsl'].values * 1000
    line1, = ax1.plot(years, gmsl_mm, 'b-', linewidth=1.5, alpha=0.7, label='GMSL')
    ax1.set_ylabel('GMSL (mm)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Right axis: instantaneous rate
    rate_mm = df['rate'].values * 1000
    line2, = ax2.plot(df.loc[valid, 'decimal_year'], rate_mm[valid], 
                      'r-', linewidth=2, label='Rate')
    ax2.set_ylabel('Rate (mm/yr)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.axhline(0, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
    
    ax1.set_title(name)
    ax1.set_xlabel('Year')
    ax1.grid(True, alpha=0.3)
    
    # Legend
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

if len(kinematics) < 6:
    axes[-1].axis('off')

plt.suptitle('Sea Level (left) and Rate of Change (right)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('figures/cumulative_vs_rate.png', dpi=150, bbox_inches='tight')
plt.show()


# %% [markdown]
# ### 9.7 Interactive Exploration (Plotly)


# %% [code]
# -----------------------------------------------------------------------------
# PLOT 7: Interactive Plotly visualization
# -----------------------------------------------------------------------------

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create interactive figure with dropdown for dataset selection
fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    subplot_titles=('GMSL', 'Rate', 'Acceleration'),
                    vertical_spacing=0.08)

# Add traces for each dataset (initially all visible)
for name, res in kinematics.items():
    df = kinematics_dfs[name]
    orig_df = orig_data[name]
    valid = ~np.isnan(df['rate'])
    color = colors[name]
    
    # GMSL
    fig.add_trace(
        go.Scatter(x=df['decimal_year'], y=orig_df['gmsl'].values * 1000,
                   mode='lines', name=f'{name} GMSL',
                   line=dict(color=color), legendgroup=name,
                   hovertemplate='Year: %{x:.1f}<br>GMSL: %{y:.1f} mm'),
        row=1, col=1
    )
    
    # Rate with uncertainty
    rate_mm = df['rate'] * 1000
    rate_se_mm = df['rate_se'] * 1000
    
    fig.add_trace(
        go.Scatter(x=df.loc[valid, 'decimal_year'], y=rate_mm[valid],
                   mode='lines', name=f'{name} Rate',
                   line=dict(color=color), legendgroup=name,
                   showlegend=False,
                   hovertemplate='Year: %{x:.1f}<br>Rate: %{y:.2f} mm/yr'),
        row=2, col=1
    )
    
    # Rate uncertainty band
    fig.add_trace(
        go.Scatter(x=pd.concat([df.loc[valid, 'decimal_year'], 
                                df.loc[valid, 'decimal_year'][::-1]]),
                   y=pd.concat([(rate_mm + 1.96*rate_se_mm)[valid],
                               (rate_mm - 1.96*rate_se_mm)[valid][::-1]]),
                   fill='toself', fillcolor=color, opacity=0.2,
                   line=dict(color='rgba(0,0,0,0)'),
                   showlegend=False, legendgroup=name,
                   hoverinfo='skip'),
        row=2, col=1
    )
    
    # Acceleration with uncertainty
    accel_mm = df['accel'] * 1000
    accel_se_mm = df['accel_se'] * 1000
    
    fig.add_trace(
        go.Scatter(x=df.loc[valid, 'decimal_year'], y=accel_mm[valid],
                   mode='lines', name=f'{name} Accel',
                   line=dict(color=color), legendgroup=name,
                   showlegend=False,
                   hovertemplate='Year: %{x:.1f}<br>Accel: %{y:.4f} mm/yr²'),
        row=3, col=1
    )
    
    # Acceleration uncertainty band
    fig.add_trace(
        go.Scatter(x=pd.concat([df.loc[valid, 'decimal_year'],
                                df.loc[valid, 'decimal_year'][::-1]]),
                   y=pd.concat([(accel_mm + 1.96*accel_se_mm)[valid],
                               (accel_mm - 1.96*accel_se_mm)[valid][::-1]]),
                   fill='toself', fillcolor=color, opacity=0.2,
                   line=dict(color='rgba(0,0,0,0)'),
                   showlegend=False, legendgroup=name,
                   hoverinfo='skip'),
        row=3, col=1
    )

# Update layout
fig.update_layout(
    height=900, width=1000,
    title_text=f'GMSL Kinematics ({SPAN_YEARS}-year bandwidth) - Interactive',
    hovermode='x unified',
    legend=dict(orientation='h', yanchor='bottom', y=1.02)
)

fig.update_yaxes(title_text='GMSL (mm)', row=1, col=1)
fig.update_yaxes(title_text='Rate (mm/yr)', row=2, col=1)
fig.update_yaxes(title_text='Acceleration (mm/yr²)', row=3, col=1)
fig.update_xaxes(title_text='Year', row=3, col=1)

# Add zero lines
fig.add_hline(y=0, line_dash='dash', line_color='black', opacity=0.5, row=2, col=1)
fig.add_hline(y=0, line_dash='dash', line_color='black', opacity=0.5, row=3, col=1)

fig.show()


# %% [markdown]
# ### 9.8 Thermodynamic Signal and Components


# %% [code]
# -----------------------------------------------------------------------------
# PLOT 8: Thermodynamic signal and components
# -----------------------------------------------------------------------------

fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)

# Left column: Frederikse
# Right column: Dangendorf

component_colors = {
    'thermodynamic': 'black',
    'steric': 'C0',
    'barystatic': 'C1',
    'glaciers': 'C2',
    'greenland': 'C3',
    'antarctica': 'C4'
}

# --- Frederikse (left column) ---

# Row 1: Signal
ax = axes[0, 0]
for component in ['thermodynamic', 'steric', 'barystatic']:
    key = f'frederikse_{component}'
    if key in thermo_kinematics:
        df = thermo_kinematics_dfs[key]
        if component == 'thermodynamic':
            orig = df_frederikse_thermo[component] * 1000
        else:
            orig = df_frederikse_thermo[component] * 1000
        ax.plot(df['decimal_year'], orig.values, '-', 
                color=component_colors[component], linewidth=1.5, label=component)
ax.set_ylabel('Signal (mm)')
ax.set_title('Frederikse')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# Row 2: Rates
ax = axes[1, 0]
for component in ['thermodynamic', 'steric', 'barystatic', 'glaciers', 'greenland', 'antarctica']:
    key = f'frederikse_{component}'
    if key in thermo_kinematics:
        df = thermo_kinematics_dfs[key]
        valid = ~np.isnan(df['rate'])
        rate_mm = df['rate'] * 1000
        ax.plot(df.loc[valid, 'decimal_year'], rate_mm[valid], '-',
                color=component_colors[component], linewidth=2, label=component)
ax.set_ylabel('Rate (mm/yr)')
ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax.legend(loc='upper left', ncol=2)
ax.grid(True, alpha=0.3)

# Row 3: Accelerations
ax = axes[2, 0]
for component in ['thermodynamic', 'steric', 'barystatic', 'glaciers', 'greenland', 'antarctica']:
    key = f'frederikse_{component}'
    if key in thermo_kinematics:
        df = thermo_kinematics_dfs[key]
        valid = ~np.isnan(df['accel'])
        accel_mm = df['accel'] * 1000
        ax.plot(df.loc[valid, 'decimal_year'], accel_mm[valid], '-',
                color=component_colors[component], linewidth=2, label=component)
ax.set_ylabel('Acceleration (mm/yr²)')
ax.set_xlabel('Year')
ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax.legend(loc='upper left', ncol=2)
ax.grid(True, alpha=0.3)

# --- Dangendorf (right column) ---

# Row 1: Signal
ax = axes[0, 1]
for component in ['thermodynamic', 'steric', 'barystatic']:
    key = f'dangendorf_{component}'
    if key in thermo_kinematics:
        df = thermo_kinematics_dfs[key]
        orig = df_dangendorf_thermo[component] * 1000
        ax.plot(df['decimal_year'], orig.values, '-',
                color=component_colors[component], linewidth=1.5, label=component)
ax.set_title('Dangendorf')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# Row 2: Rates
ax = axes[1, 1]
for component in ['thermodynamic', 'steric', 'barystatic']:
    key = f'dangendorf_{component}'
    if key in thermo_kinematics:
        df = thermo_kinematics_dfs[key]
        valid = ~np.isnan(df['rate'])
        rate_mm = df['rate'] * 1000
        ax.plot(df.loc[valid, 'decimal_year'], rate_mm[valid], '-',
                color=component_colors[component], linewidth=2, label=component)
ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# Row 3: Accelerations
ax = axes[2, 1]
for component in ['thermodynamic', 'steric', 'barystatic']:
    key = f'dangendorf_{component}'
    if key in thermo_kinematics:
        df = thermo_kinematics_dfs[key]
        valid = ~np.isnan(df['accel'])
        accel_mm = df['accel'] * 1000
        ax.plot(df.loc[valid, 'decimal_year'], accel_mm[valid], '-',
                color=component_colors[component], linewidth=2, label=component)
ax.set_xlabel('Year')
ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.suptitle(f'Thermodynamic Signal and Components ({SPAN_YEARS}-year bandwidth)', 
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('figures/thermodynamic_components.png', dpi=150, bbox_inches='tight')
plt.show()


# %% [markdown]
# ---
# **TODO:** Circle back to incorporate additional thermosteric and TWS datasets to better constrain the thermodynamic signal.
