#!/bin/sh
#SBATCH --partition=GPUQ           # Use a GPU
#SBATCH --account=ie-idi
#SBATCH --time=63:00:00            # Max wait time
#SBATCH --nodes=1
#SBATCH -c4                        # Number of cores
#SBATCH --gres=gpu:1               # Require 1 GPU
#SBATCH --mem-per-gpu=15G          # Require certain amount of GPU memory
#SBATCH --constraint="A100"        # Don't have to use A100
#SBATCH --job-name="CosmoML_VGG"
#SBATCH --output=VGG.out
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Job Name:          $SLURM_JOB_NAME"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo "Job ID:            $SLURM_JOB_ID"
echo "Nodes used:        $SLURM_JOB_NODELIST"
echo "Number of nodes:   $SLURM_JOB_NUM_NODES"
echo "Cores (per node):  $SLURM_CPUS_ON_NODE"
echo "Total cores:       $SLURM_NTASKS"

# module load torchvision/0.8.2-fosscuda-2020b-PyTorch-1.7.1
# module load PyTorch/1.8.1-fosscuda-2020b
# module swap NCCL/2.8.3-CUDA-11.1.1 NCCL/2.8.3-GCCcore-10.2.0-CUDA-11.1.1
# module swap PyTorch/1.7.1-fosscuda-2020b PyTorch/1.8.1-fosscuda-2020b
# module load scikit-learn/0.23.2-fosscuda-2020b

module purge
module load SciPy-bundle/2021.10-foss-2021b
module list

source /cluster/work/modestas/CosmoML/venv/bin/activate

PH=/cluster/work/modestas/CosmoML/src/
export PYTHONPATH=$PH:$PYTHONPATH

time python3 $PH/VGG_example.py -t "/cluster/work/modestas/train100k/train.csv" -i "/cluster/work/modestas/train100k" -T "/cluster/work/modestas/test10k/test.csv" -I "/cluster/work/modestas/test10k" -s "VGG_0001.pt" -p "VGG_0001_pre.csv" -o "VGG_0001.csv"