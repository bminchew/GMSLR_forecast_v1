"""Check the container environment without touching data.

Imports every direct dependency of the pipeline and the slr_forecast package,
prints versions, the BLAS/OpenMP thread pools, and the HDF5 libraries seen by
h5py, PyTables and netCDF4.  Exits non-zero if anything fails to import or a
version differs from its pin.

Usage (inside the image):  python environment/check_env.py
"""
import importlib
import json
import platform
import sys
from pathlib import Path

# Pinned versions: those imported in the macOS development environment
# (python2025, osx-64).  The results of record come from the linux/arm64
# container run with these versions.
EXPECTED = {
    "numpy": "2.3.2",
    "scipy": "1.16.1",
    "pandas": "2.3.2",
    "matplotlib": "3.10.6",
    "h5py": "3.15.1",
    "tables": "3.10.2",
    "numexpr": "2.14.1",
    "netCDF4": "1.7.2",
    "cftime": "1.6.5",
    "xarray": "2026.2.0",
    "openpyxl": "3.1.5",
    "tqdm": "4.67.3",
    "statsmodels": "0.14.6",
    "emcee": "3.1.6",
    "arviz": "0.23.4",
    "corner": "2.2.3",
    "seaborn": "0.13.2",
    "jupyter_core": "5.9.1",
    "nbconvert": "7.17.0",
    "nbclient": "0.10.4",
    "nbformat": "5.10.4",
    "ipykernel": "7.2.0",
    "IPython": "9.10.0",
    "jupyter_client": "8.8.0",
    "pytest": "9.0.2",
    "threadpoolctl": "3.6.0",
    "slr_forecast": "0.1.0",
    # preprocessing and data-download scripts only
    "gsw": "3.6.23",
    "cartopy": "0.25.0",
    "shapely": "2.1.2",
    "pyproj": "3.7.2",
    "fsspec": "2026.2.0",
    "aiohttp": "3.12.15",
    "requests": "2.34.2",
    "zarr": "3.1.6",
    "dask": "2025.9.1",
    "plotly": "6.5.2",
}

failures = []
print(f"python {sys.version.split()[0]}  {platform.machine()}  {platform.platform()}")
if sys.version.split()[0] != "3.13.7":
    failures.append("python version")
for name, want in EXPECTED.items():
    try:
        mod = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL  {name:15s} import error: {exc!r}")
        failures.append(name)
        continue
    have = getattr(mod, "__version__", "?")
    flag = "ok  " if have == want else "DIFF"
    if have != want:
        failures.append(f"{name} {have} != {want}")
    print(f"  {flag}  {name:15s} {have:10s} (pinned {want})")

# Submodules the notebooks use
for sub in ["scipy.stats", "scipy.optimize", "scipy.signal", "scipy.interpolate",
            "statsmodels.api", "matplotlib.pyplot", "xarray.backends",
            "cartopy.io.shapereader", "shapely.geometry", "shapely.ops",
            "fsspec.implementations.http", "plotly.graph_objects", "dask.array"]:
    try:
        importlib.import_module(sub)
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL  {sub}: {exc!r}")
        failures.append(sub)

# The package must be an editable install rooted at the repository, because
# slr_forecast.config derives PROJECT_ROOT (and the data paths) from __file__.
import slr_forecast  # noqa: E402
from slr_forecast import config  # noqa: E402
print(f"slr_forecast file: {slr_forecast.__file__}")
print(f"PROJECT_ROOT: {config.PROJECT_ROOT}   N_SAMPLES: {config.N_SAMPLES}")
root = Path(config.PROJECT_ROOT)
if not (root / "run_pipeline.sh").exists():
    failures.append(f"PROJECT_ROOT {root} does not contain run_pipeline.sh")

# Local style module the notebooks load from notebooks/arete_mpl/
sys.path.insert(0, str(root / "notebooks" / "arete_mpl"))
try:
    import arete_mpl  # noqa: E402
    print(f"arete_mpl file: {arete_mpl.__file__}")
except Exception as exc:  # noqa: BLE001
    print(f"  FAIL  arete_mpl (notebooks/arete_mpl): {exc!r}")
    failures.append("arete_mpl")

# Kernel used by run_pipeline.sh (--ExecutePreprocessor.kernel_name=python3)
from jupyter_client.kernelspec import KernelSpecManager  # noqa: E402
specs = KernelSpecManager().find_kernel_specs()
print(f"kernelspecs: {specs}")
if "python3" not in specs:
    failures.append("python3 kernelspec")
else:
    # Start the kernel and run one cell: under Docker Desktop's Rosetta
    # emulation the linux/amd64 image starts kernels that then hang, which
    # imports alone do not reveal.
    from jupyter_client.manager import start_new_kernel  # noqa: E402
    km = kc = None
    try:
        km, kc = start_new_kernel(kernel_name="python3", startup_timeout=60)
        reply = kc.execute_interactive("1 + 1", timeout=60, output_hook=lambda msg: None)
        if reply["content"]["status"] != "ok":
            raise RuntimeError(f"cell status {reply['content']['status']}")
        print("python3 kernel: started and ran a cell")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL  python3 kernel did not run a cell: {exc!r}")
        failures.append("python3 kernel start")
    finally:
        if kc is not None:
            kc.stop_channels()
        if km is not None:
            km.shutdown_kernel(now=True)
        del km, kc   # avoid jupyter_client finalizer noise at interpreter exit

import h5py, tables, netCDF4  # noqa: E401,E402
print(f"HDF5: h5py {h5py.version.hdf5_version}, PyTables {tables.hdf5_version}, "
      f"netCDF4 {netCDF4.__hdf5libversion__} (netCDF {netCDF4.__netcdf4libversion__})")

from threadpoolctl import threadpool_info  # noqa: E402
print("thread pools:")
print(json.dumps([{k: d.get(k) for k in ("internal_api", "version", "threading_layer",
                                          "architecture", "num_threads", "filepath")}
                  for d in threadpool_info()], indent=1))

if failures:
    print("\nFAILURES:", *failures, sep="\n  ")
    sys.exit(1)
print("\nall checks passed")
