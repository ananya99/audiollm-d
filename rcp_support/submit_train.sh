#!/bin/bash
# CS-552 — example Run:AI training job launcher.
#
# This is NOT a deliverable script. It is a helper/example for longer
# compute runs where you want the job to execute a training command and
# exit when the command finishes.
#
# For grading, submit the interactive submit.sh next to your notebook.
# Use this file only as a starting point for your own training runs.
#
# Training jobs are lower priority than interactive jobs and can be
# preempted/restarted by the scheduler. Your code must write checkpoints
# to /scratch and resume from them.

set -euo pipefail

# ============== EDIT THESE LINES ==============
GASPAR="gaspar"              # <-- YOUR GASPAR EPFL username.
GROUP="gXX"                  # <-- YOUR TEAM, e.g. g07.
# ==============================================

# Edit this for your project. Keep outputs/checkpoints under /scratch.
TRAIN_COMMAND='cd /scratch/<your-repo> && python train.py --output-dir /scratch/runs/train-v1'

if [[ "${GASPAR}" == "gaspar" || -z "${GASPAR}" ]]; then
    echo "ERROR: edit submit_train.sh and set GASPAR to your EPFL GASPAR username." >&2
    exit 1
fi

if [[ "${GROUP}" == "gXX" || -z "${GROUP}" ]]; then
    echo "ERROR: edit submit_train.sh and set GROUP to your team number (e.g. g07)." >&2
    exit 1
fi

if [[ "${TRAIN_COMMAND}" == *"<your-repo>"* || -z "${TRAIN_COMMAND}" ]]; then
    echo "ERROR: edit TRAIN_COMMAND before submitting a training job." >&2
    exit 1
fi

GPUS=1
NODE="${NODE:-a100-40g}"
SUFFIX="${1:-train}"
JOB_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-$(date +%H%M%S)"
PROJECT="course-cs-552-${GASPAR}"

IMAGE="registry.rcp.epfl.ch/course-cs-552/base-vllm:v1"

SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

echo ">>> Submitting training job ${JOB_NAME}  (1 GPU)"

runai submit \
  --name "${JOB_NAME}" \
  -p "${PROJECT}" \
  --image "${IMAGE}" \
  --gpu "${GPUS}" \
  --large-shm \
  --node-pools "${NODE}" \
  --working-dir /scratch \
  --environment HF_HOME=/scratch/hf_cache \
  --environment HF_HUB_ENABLE_HF_TRANSFER=1 \
  --environment WANDB_DIR=/scratch/wandb \
  --environment TRAIN_COMMAND="${TRAIN_COMMAND}" \
  --existing-pvc "claimname=${SCRATCH_PVC},path=/scratch" \
  --existing-pvc "claimname=${SHARED_RO_PVC},path=/shared-ro" \
  --existing-pvc "claimname=${SHARED_RW_PVC},path=/shared-rw" \
  --command -- /bin/bash -lc "\
    set -euo pipefail && \
    mkdir -p /scratch/hf_cache /scratch/wandb /scratch/runs && \
    ln -sf \"\$(command -v python3)\" /usr/local/bin/python && \
    cd /scratch && \
    eval \"\${TRAIN_COMMAND}\""

cat <<EOF

>>> Training job submitted: ${JOB_NAME}

Watch it start:    runai describe job ${JOB_NAME} -p ${PROJECT}
Stream logs:       runai logs -f ${JOB_NAME} -p ${PROJECT}
List jobs:         runai list jobs -p ${PROJECT}
Stop the job:      runai delete job ${JOB_NAME} -p ${PROJECT}

This is a Run:AI training job, not an interactive Jupyter job. It exits
when TRAIN_COMMAND finishes. Make sure your training code writes
checkpoints to /scratch and can resume after preemption.
EOF
