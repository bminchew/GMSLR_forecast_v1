# Use an official Jupyter base image with scientific tools
FROM jupyter/scipy-notebook:latest

# Set up the internal structure
WORKDIR /home/jovyan/work
COPY --chown=jovyan:users . /home/jovyan/work

# Install specific dependencies (like h5py for Dangendorf data)
RUN mamba install --yes h5py seaborn statsmodels

# Expose the port for Jupyter
EXPOSE 8888
