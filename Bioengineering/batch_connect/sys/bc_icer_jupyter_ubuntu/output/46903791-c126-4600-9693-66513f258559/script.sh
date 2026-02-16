#!/usr/bin/env bash

# Get working directory's group
GROUPNAME=$(ls -ld $(readlink -f ${NOTEBOOK_ROOT}) | cut -d " " -f 4)

# If the user is not part of the working directory's group, use their primary group
if [[ ! " $(groups) " =~ " $GROUPNAME " ]]; then
  GROUPNAME=$(groups | cut -d" " -f1)
fi

newgrp ${GROUPNAME} <<'EOF'
source /etc/profile  # newgrp removes some setup, this restores it
echo "Starting main script..."
echo "TTT - $(date)"


#
# Start Jupyter server
#

# Clean the environment
module purge

# Create launcher wrapper
echo "Creating launcher wrapper script..."
(
umask 077
sed 's/^ \{2\}//' > "/mnt/home/behren23/ondemand/data/sys/dashboard/batch_connect/sys/bc_icer_jupyter_ubuntu/output/46903791-c126-4600-9693-66513f258559/launch_wrapper.sh" << EOL
  #!/usr/bin/env bash

  # Log all output from this script
  exec &>>"/mnt/home/behren23/ondemand/data/sys/dashboard/batch_connect/sys/bc_icer_jupyter_ubuntu/output/46903791-c126-4600-9693-66513f258559/launch_wrapper.log"

  # Load the required environment
  module purge
  module load JupyterLab/4.0.5-GCCcore-12.3.0 JupyterNotebook/7.0.2-GCCcore-12.3.0 SciPy-bundle/2023.07-gfbf-2023a jupyterlmod/4.0.3-GCCcore-12.3.0
  module list

  # Launch the original command
  set -x
  exec "\${@}"
EOL
)
chmod 700 "/mnt/home/behren23/ondemand/data/sys/dashboard/batch_connect/sys/bc_icer_jupyter_ubuntu/output/46903791-c126-4600-9693-66513f258559/launch_wrapper.sh"
echo "TTT - $(date)"

export jtype="notebook"

export jtype="lab"

# Create user-created Conda env Jupyter kernels
echo "Creating custom Jupyter kernels from user-created Conda environments..."
for dir in "${HOME}/.conda/envs"/*/ "${HOME}/envs"/*/ ; do
  (
  umask 077
  set -e
  KERNEL_NAME="$(basename "${dir}")"
  KERNEL_PATH="~${dir#${HOME}}"
  [[ -x "${dir}bin/activate" ]] || exit 0
  echo "Creating kernel for ${dir}..."
  source "${dir}bin/activate" "${dir}"
  set -x
  if [[ "$(conda list -f --json ipykernel)" == "[]" ]]; then
    CONDA_PKGS_DIRS="$(mktemp -d)" conda install --yes --quiet --no-update-deps ipykernel
  fi
  python \
    -m ipykernel \
      install \
      --name "conda_${KERNEL_NAME}" \
      --display-name "${KERNEL_NAME} [${KERNEL_PATH}]" \
      --prefix "${PWD}"
  ) &
done

# Set working directory to notebook root directory
cd "${NOTEBOOK_ROOT}"


# Setup Jupyter environment
      module load JupyterLab/4.0.5-GCCcore-12.3.0 JupyterNotebook/7.0.2-GCCcore-12.3.0 SciPy-bundle/2023.07-gfbf-2023a jupyterlmod/4.0.3-GCCcore-12.3.0
    module load CUDA
  module list
  echo "TTT - $(date)"

  # List available kernels for debugging purposes
  set -x
  jupyter kernelspec list
  { set +x; } 2>/dev/null
  echo "TTT - $(date)"

  # Launch the Jupyter server
  set -x
  env | sort
  module list
  jupyter ${jtype} --config="${CONFIG_FILE}"
EOF

# Check if jupyter was not found
if grep "jupyter: command not found" "/mnt/home/behren23/ondemand/data/sys/dashboard/batch_connect/sys/bc_icer_jupyter_ubuntu/output/46903791-c126-4600-9693-66513f258559/output.log" > /dev/null; then
  touch "/mnt/home/behren23/ondemand/data/sys/dashboard/batch_connect/sys/bc_icer_jupyter_ubuntu/output/46903791-c126-4600-9693-66513f258559/no-jupyter.error"
fi
