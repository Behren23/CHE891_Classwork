#!/usr/bin/env bash

# Log all output from this script
exec &>>"/mnt/home/behren23/ondemand/data/sys/dashboard/batch_connect/sys/bc_icer_jupyter_ubuntu/output/d62f898d-0755-4c16-90c9-9a9403834c81/launch_wrapper.log"

# Load the required environment
module purge
module load JupyterLab/4.0.5-GCCcore-12.3.0 JupyterNotebook/7.0.2-GCCcore-12.3.0 SciPy-bundle/2023.07-gfbf-2023a jupyterlmod/4.0.3-GCCcore-12.3.0
module list

# Launch the original command
set -x
exec "${@}"
