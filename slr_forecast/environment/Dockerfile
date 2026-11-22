# SPECTRUM sea-level forecast: reproducible runtime image.
#
# Build from the repository root (the directory that holds run_pipeline.sh):
#   docker build -f environment/Dockerfile -t spectrum:latest .
#
# Data are NOT in the image.  Mount them at run time:
#   /work/data/raw        read-only   (raw observational and model inputs)
#   /work/data/processed  read-write  (HDF5 intermediates and results)
# See environment/README.md for the full run commands.

# micromamba 2.10.0 on Debian 12 (bookworm) slim, pinned by multi-arch index digest.
FROM mambaorg/micromamba:2.10.0-debian12-slim@sha256:bd7f1077166c1c71e9375a0e54835f151d5f216b96301b75f888c53c46d8e78b

# Set by BuildKit (amd64 or arm64); selects the matching conda lock.
ARG TARGETARCH

# Liberation Sans (metric-compatible with Arial) for the figures drawn with
# the arete style, which resolves to Helvetica Neue (not redistributable) on
# macOS.  Figures drawn with matplotlib's defaults use its bundled DejaVu Sans.
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends fonts-liberation2=2.1.5-1 \
 && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

# 1. Conda environment from the explicit, md5-checked lock for this platform.
COPY --chown=$MAMBA_USER:$MAMBA_USER environment/conda-linux-64.lock environment/conda-linux-aarch64.lock /tmp/lock/
RUN set -eux; \
    arch="${TARGETARCH:-$(uname -m)}"; \
    case "$arch" in \
      amd64|x86_64)  plat=linux-64 ;; \
      arm64|aarch64) plat=linux-aarch64 ;; \
      *) echo "unsupported architecture: $arch" >&2; exit 1 ;; \
    esac; \
    micromamba install -y -n base -f "/tmp/lock/conda-${plat}.lock"; \
    micromamba clean --all --yes; \
    rm -rf /tmp/lock

# Activate the base env for the RUN steps below.
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Numerical threading: one thread per BLAS/OpenMP/numexpr pool, so that
# floating-point reduction order (and hence results) does not depend on the
# host core count.  Override at run time with -e OMP_NUM_THREADS=... etc.
# MPLBACKEND is not set: the notebook kernels then use the inline backend, so
# the executed notebooks keep their figures, and scripts fall back to Agg
# (no display).
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    NUMEXPR_MAX_THREADS=1 \
    PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    IPYTHONDIR=/tmp/ipython \
    JUPYTER_RUNTIME_DIR=/tmp/jupyter-runtime \
    XDG_CACHE_HOME=/tmp/cache \
    XDG_CONFIG_HOME=/tmp/config \
    XDG_DATA_HOME=/tmp/data \
    PYDEVD_DISABLE_FILE_VALIDATION=1 \
    PYTHON=python3

USER root
RUN mkdir -p /work /opt/spectrum && chown $MAMBA_USER:$MAMBA_USER /work /opt/spectrum
USER $MAMBA_USER
WORKDIR /work

# 2. The slr_forecast package, installed editable: slr_forecast.config derives
#    PROJECT_ROOT (and all data paths) from its own file location, so the
#    package must stay at /work/src.  No dependency resolution or network:
#    everything comes from the lock.
COPY --chown=$MAMBA_USER:$MAMBA_USER pyproject.toml ./
COPY --chown=$MAMBA_USER:$MAMBA_USER src ./src
RUN SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 \
    python -m pip install --no-deps --no-build-isolation --no-index -e . \
 && rm -rf src/*.egg-info

# 3. Environment files: entrypoint, check_env.py, version records.
COPY --chown=$MAMBA_USER:$MAMBA_USER environment ./environment

# 4. Pipeline code.  The notebooks are rewritten in place by nbconvert and
#    component_wais rewrites notebooks/s1_quadratic_fit.json, so /work stays
#    writable; figures/ is created empty (outputs only).
COPY --chown=$MAMBA_USER:$MAMBA_USER run_pipeline.sh ./
COPY --chown=$MAMBA_USER:$MAMBA_USER notebooks ./notebooks
COPY --chown=$MAMBA_USER:$MAMBA_USER scripts ./scripts
COPY --chown=$MAMBA_USER:$MAMBA_USER tests ./tests
# Group/other write permission lets the image run as the host user on Linux
# (docker run --user "$(id -u):$(id -g)"), which keeps files written to the
# mounted data/processed owned by that user.
RUN mkdir -p data/raw data/processed figures/diagnostics notebooks/figures \
 && chmod +x run_pipeline.sh environment/entrypoint.sh \
 && chmod -R a+rwX /work

# Record the resolved environment inside the image for provenance, outside
# /work so that a repository mounted at /work does not hide it.
RUN micromamba list -n base --explicit --md5 > /opt/spectrum/installed-explicit.txt \
 && python -m pip list --format=freeze > /opt/spectrum/installed-pip-freeze.txt

# Through bash, so a repository mounted at /work need not keep the
# executable bit of entrypoint.sh.
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "bash", "/work/environment/entrypoint.sh"]
CMD []
