import importlib, sys, platform, io, contextlib
print("python", sys.version.replace("\n"," "), platform.machine(), platform.platform())
mods = ["numpy","scipy","pandas","matplotlib","h5py","tables","netCDF4","cftime","xarray","h5netcdf","statsmodels","seaborn","emcee","corner","arviz","openpyxl","tqdm","numexpr","blosc2","pytest","jupyter_core","nbconvert","nbclient","nbformat","ipykernel","IPython","jupyter_client","caltech_mpl","slr_forecast","sklearn","xarray_einstats","setuptools"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"{m:16s} {getattr(mod,'__version__','?'):14s} {getattr(mod,'__file__','')}")
    except Exception as e:
        print(f"{m:16s} IMPORT-FAIL {e!r}")
import h5py, tables, netCDF4
print("h5py hdf5", h5py.version.hdf5_version)
print("tables hdf5", tables.hdf5_version)
print("netCDF4 hdf5", netCDF4.__hdf5libversion__, "netcdf", netCDF4.__netcdf4libversion__)
import numpy as np, scipy
print("=== numpy config"); np.show_config()
print("=== scipy config"); scipy.show_config()
try:
    from threadpoolctl import threadpool_info
    import json; print(json.dumps(threadpool_info(), indent=1))
except Exception as e: print("threadpoolctl", e)
