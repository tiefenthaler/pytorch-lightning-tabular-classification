# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Create a docker conda image
# FROM mcr.microsoft.com/devcontainers/anaconda:0-3
FROM continuumio/miniconda3

# Copy the environment.yml file to the container. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
# COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
COPY environment_conda.yml* /tmp/conda-tmp/

# Use Conda to create/update python environment
# RUN if [ -f "/tmp/conda-tmp/environment_conda.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment_conda.yml; fi \
#     && rm -rf /tmp/conda-tmp

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

# WORKDIR /project

# Expose the port that the application listens on.
# EXPOSE 8000

# Command to run on container start.
# CMD python main.py
