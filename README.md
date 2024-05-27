# Scanalyser

Welcome! Scanalyser is a project focused on applying Machine Learning (Unsupervised Learning via Convolutional Autoencoders) to the Raman-spectra scans of Picocavity events. The main use case: reconstructing the stable/nanocavity state of the picocavity-containing scan.

## Installation

### 0. Clone the repo

For instance:

```
git clone https://github.com/nanophotonics/scanalyser.git
```

If you already have a multi-GPU (or a single GPU) node/hardware with CUDA, cuDNN, NCCL, and TensorFlow >= 2.13 working properly, then you can skip the installation from scratch.

## Install from scratch on the Cambridge HPC
Here is the example of setting up the enviroment using venv and the modules available on the cluster.

### 1. Load Required Modules

Load the necessary modules for Python, CUDA, and cuDNN:

```
module load python-3.9.6-gcc-5.4.0-sbr552h
module load py-virtualenv-15.1.0-gcc-5.4.0-gu4wi6c
module load cuda/12.1
module load cudnn/8.9_cuda-12.1
```

### 2. Create a Virtual Environment

For instance:

```
python -m venv myenv
```

### 3. Update the Activation File

Update the activation file with the following content:

```bash
# This file must be used with "source bin/activate" *from bash*
# you cannot run it directly

# Check if the environment is already activated
#if [ -n "${VIRTUAL_ENV}" ]; then
#    echo "Environment is already activated."
#    return
#fi

# Save the current environment
export _OLD_PATH="$PATH"
export _OLD_PYTHONHOME="$PYTHONHOME"
export _OLD_PS1="$PS1"

echo "Setting the venv & loading modules.."

module load python-3.9.6-gcc-5.4.0-sbr552h
module load py-virtualenv-15.1.0-gcc-5.4.0-gu4wi6c
module load cuda/12.1
module load cudnn/8.9_cuda-12.1

# Set CUDA and cuDNN paths
export CUDA_HOME=/usr/local/software/cuda/12.1
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/include:$CPATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/software/cuda/12.1

export CUDNN_HOME=/usr/local/Cluster-Apps/cudnn/8.9_cuda-12.1
export CPATH=$CUDNN_HOME/include:$CPATH
export LIBRARY_PATH=$CUDNN_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDNN_HOME/lib64:$LD_LIBRARY_PATH

export VIRTUAL_ENV="$(cd "$(dirname "$(dirname "$BASH_SOURCE")")" && pwd)"

deactivate () {
    unset -f pydoc >/dev/null 2>&1

    # reset old environment variables
    PATH="$_OLD_PATH"
    export PATH
    PYTHONHOME="$_OLD_PYTHONHOME"
    export PYTHONHOME
    PS1="$_OLD_PS1"
    export PS1

    echo "Usetting venv & unloading modules.."
    # Unload the modules if necessary
    module unload python-3.9.6-gcc-5.4.0-sbr552h
    module unload py-virtualenv-15.1.0-gcc-5.4.0-gu4wi6c
    module unload cudnn/8.9_cuda-12.1
    module unload cuda/12.1

    # Unset variables and deactivate the function
    unset _OLD_PATH _OLD_PYTHONHOME _OLD_PS1

    # This should detect bash and zsh, which have a hash command that must
    # be called to get it to forget past commands.  Without forgetting
    # past commands the $PATH changes we made may not be respected
    if [ -n "${BASH-}" ] || [ -n "${ZSH_VERSION-}" ] ; then
        hash -r 2>/dev/null
    fi

    unset VIRTUAL_ENV
    if [ ! "${1-}" = "nondestructive" ] ; then
    # Self destruct!
        unset -f deactivate
    fi
}

export PATH="$VIRTUAL_ENV/bin:$PATH"

# unset PYTHONHOME if set
if ! [ -z "${PYTHONHOME+_}" ] ; then
    _OLD_VIRTUAL_PYTHONHOME="$PYTHONHOME"
    unset PYTHONHOME
fi

if [ -z "${VIRTUAL_ENV_DISABLE_PROMPT-}" ] ; then
    _OLD_VIRTUAL_PS1="$PS1"
    if [ "x" != x ] ; then
        PS1="$PS1"
    else
        PS1="(`basename \\\\"$VIRTUAL_ENV\\\\"`) $PS1"
    fi
    export PS1
fi

# Make sure to unalias

```

### 4. Exit and Reconnect

Exit the current session and reconnect.

### 5. Activate the Virtual Environment

Activate the virtual environment:

```
source myenv/bin/activate
```

### 6. Install TensorFlow 2.15
The set up is tested with the version of TF == 2.15. Other versions may lead to NCCL compatibility issues. (Means single-GPU training only)

Install TensorFlow:

```
pip install tensorflow==2.15
```

## Training a New Model on Cambridge HPC

### 1. Update or Create a Config File

Create or update the configuration file where you specify the model name and the number of training epochs.
See the `configs` folder.
### 2. Update the Training File Path

Update the path to the configuration file in `main_cae_train.py` or write your own training file.

### 3. Submit a Job

Update the job submission script (`.wilkes3` file in your home dir) by changing `[PATH_TO_YOUR_VENV]` and `[PATH_TO_TRAINFILE]`:

```bash
#!/bin/bash
#SBATCH -J pico_ai_train
#SBATCH -A NIJS-SL3-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=11:59:00
#SBATCH --mail-type=NONE
#SBATCH --no-requeue
#SBATCH -p ampere

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

source [PATH_TO_YOUR_VENV]/bin/activate
application="[PATH_TO_YOUR_VENV]/bin/python"
options="[PATH_TO_TRAINFILE]/main_cae_train.py > output.log 2> error.log"

workdir="$SLURM_SUBMIT_DIR"
cd $workdir

echo -e "Changed directory to `pwd`.\\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

if [ "$SLURM_JOB_NODELIST" ]; then
    export NODEFILE=`generate_pbs_nodefile`
    cat $NODEFILE | uniq > machine.file.$JOBID
    echo -e "\\nNodes allocated:\\n================"
    echo `cat machine.file.$JOBID | sed -e 's/\\\\..*$//g'`
fi

echo -e "\\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

echo -e "\\nExecuting command:\\n==================\\n$CMD\\n"

eval $CMD

```

## Running Inference

You can write your own Python code or Jupyter notebooks to run inference. Execute them by submitting a GPU job the same way you trained the model or in the interactive mode.

There is a `inference_basic.py` you can run to obtain the difference scans. You can use it as a template for your research-oriented programming.

For more information, refer to the following links:

- [Interactive Mode](https://docs.hpc.cam.ac.uk/hpc/user-guide/interactive.html)
- [Batch Mode](https://docs.hpc.cam.ac.uk/hpc/user-guide/batch.html)
- [Using A100 GPUs](https://docs.hpc.cam.ac.uk/hpc/user-guide/a100.html)
