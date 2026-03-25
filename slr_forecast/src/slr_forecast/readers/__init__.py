"""
Data readers — each returns DataFrames in SLR convention (positive = rise), meters.

Sign convention: positive = sea level rise (SLR) throughout.
Internal units: meters (sea level), degC (temperature), decimal years (time).
Convert to display units (mm, cm) only at the plotting layer.
"""

# GMSL
from slr_forecast.readers.gmsl import (
    read_nasa_gmsl,
    read_frederikse2020,
    read_dangendorf2024,
    read_horwath2022,
    read_ipcc_ar6_observed_gmsl,
)

# Temperature
from slr_forecast.readers.gmst import (
    read_berkeley_earth,
    read_berkeley_earth_gridded,
    read_hadcrut5,
    read_nasa_gistemp,
    read_noaa_globaltemp,
)

# Ice sheets
from slr_forecast.readers.ice_sheets import (
    read_imbie_greenland,
    read_imbie_east_antarctica,
    read_imbie_antarctic_peninsula,
    read_imbie_antarctica,
    read_imbie_all,
    read_imbie_west_antarctica,
    read_mouginot2019_greenland,
    read_mankoff2021_greenland,
)

# Glaciers
from slr_forecast.readers.glaciers import (
    read_glambie_global,
    read_glambie_regional,
)

# IPCC projections
from slr_forecast.readers.ipcc import (
    read_ipcc_ar6_projected_temperature,
    read_ipcc_ar6_projected_gmsl,
    read_ipcc_ar6_projected_gmsl_low_confidence,
    read_ipcc_ar6_component,
    list_ipcc_ar6_components,
)

# Forcing
from slr_forecast.readers.forcing import (
    read_noaa_thermosteric,
    read_glossac,
    read_mauna_loa_transmission,
    read_noaa_oni,
    read_noaa_mei,
)

# Ocean temperature
from slr_forecast.readers.ocean_temp import (
    read_en4_regional,
)

# TWS
from slr_forecast.readers.tws import read_grace_tws_global

# Impact functions
from slr_forecast.readers.impacts import (
    people_displaced_kulpstrauss2019,
    slr_cost_jevrejeva2018,
)

# Utilities
from slr_forecast.readers._utils import (
    decimal_year_to_datetime,
    datetime_to_decimal_year,
)
