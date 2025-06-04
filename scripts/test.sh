#!/bin/bash
#SBATCH --job-name test 
#SBATCH --partition dios 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=40:00:00
#SBATCH --output=/mnt/homeGPU/sjhu/buffer/slurm_logs/job-%j.log

HEADER="[CRP-Test] $SLURM_JOB_ID"

# Check if all changes were commited
if ! git diff --quiet || ! git diff --cached --quiet; then
  /mnt/homeGPU/sjhu/scripts/slack_notifier.sh \
  "
  $HEADER cambios no confirmados en git. Proceso rechazado. 
  "
  exit 1
fi

/mnt/homeGPU/sjhu/scripts/slack_notifier.sh \
"
$HEADER comenzado en $SLURM_JOB_NODELIST
CUDA en PyTorch: $(python -c "import torch; print(torch.cuda.is_available())")
Argumentos: $*
Commit: $(git rev-parse HEAD)
Conjunto: test
"

python test.py $*

/mnt/homeGPU/sjhu/scripts/slack_notifier.sh \
"$HEADER completado. Estado: $?. Salida en /mnt/homeGPU/sjhu/buffer/slurm_logs/job-$SLURM_JOB_ID.log"