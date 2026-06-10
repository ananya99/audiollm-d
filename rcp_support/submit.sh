#!/bin/bash
# CS-552 — submit an interactive RCP job with Jupyter Lab.
#
# This is a starter Run:AI launcher and is intended to be the basis for
# the project-wide notebooks/submit.sh deliverable in most cases. Make sure 
# GROUP is set to your team group, and modify anything else only if your
# project genuinely needs it.
#
# GASPAR selects the Run:AI project of the person launching the job. Set
# it to your own EPFL username when testing. For the submitted
# notebooks/submit.sh, it is fine if GASPAR remains "gaspar": TAs will
# replace it with their own username before grading. 

# GROUP selects your team's scratch PVC and MUST be set correctly before
# submission.
#
# ============================================================
#  STEP 1  — set your team group ID below before submission
#            set GASPAR too when you run the script yourself
# ============================================================
#
# Usage:
#   ./submit.sh                # default: 1 GPU interactive Jupyter job
#   ./submit.sh exp1           # same interactive job, custom name suffix
#
# The optional argument only changes the job name.
#
# Once the pod is Running, connect in one of these ways:
#
# 1. Jupyter Lab:
#   runai port-forward <job-name> --port 8888:8888
#   then open http://localhost:8888 (token: cs552).
#   This is a command you run after submission; no submit-time service
#   type is needed.
#
# 2. Shell:
#   runai bash <job-name>
#   Useful for CLI work, package checks, nvidia-smi, or debugging without
#   opening Jupyter.
#
# 3. VS Code:
#   Use the Kubernetes extension to attach VS Code to the running pod
#   named <job-name>-0-0. Open your repo folder inside the pod and use
#   Terminal -> New Terminal for a shell in the same environment.
#
# NOTE: GPU count is fixed at 1 (40GB A100). The course quota is one GPU
# per group allocation; asking for more will leave the job stuck Pending.

set -euo pipefail

# ============== EDIT THESE LINES ==============
GASPAR="gaspar"              # <-- For local runs: your EPFL username. TAs may replace this for grading.
GROUP="gXX"                  # <-- REQUIRED FOR SUBMISSION: your team, e.g. g07.
# ==============================================

# Refuse to run with placeholders. GASPAR is required only by whoever is
# launching the job; GROUP must be correct in the submitted file.
if [[ "${GASPAR}" == "gaspar" || -z "${GASPAR}" ]]; then
    echo "ERROR: set GASPAR to the EPFL username of the person launching this job." >&2
    echo "For the submitted notebooks/submit.sh, GASPAR may stay as 'gaspar' because TAs will replace it." >&2
    exit 1
fi

if [[ "${GROUP}" == "gXX" || -z "${GROUP}" ]]; then
    echo "ERROR: set GROUP to your team number (e.g. g07)." >&2
    echo "GROUP must be correct in the submitted notebooks/submit.sh because it selects your team's scratch PVC." >&2
    exit 1
fi

GPUS=1   # course cap: 1 GPU
NODE="${NODE:-a100-40g}"  # DO NOT CHANGE: course quota is only available on this node pool
SUFFIX="${1:-lab}"
JOB_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-$(date +%H%M%S)"
PROJECT="course-cs-552-${GASPAR}"

# Default image with CUDA, PyTorch, and common libraries.
# Override if you use a custom image. 
IMAGE="registry.rcp.epfl.ch/course-cs-552/base-vllm:v1"

SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

# This script does not mount the personal home PVC, so it does not need
# a hard-coded UID/GID. Use /scratch for course work and deliverables.

echo ">>> Submitting ${JOB_NAME}  (1 GPU)"

runai submit \
  --name "${JOB_NAME}" \
  -p "${PROJECT}" \
  --image "${IMAGE}" \
  --gpu "${GPUS}" \
  --large-shm \
  --interactive \
  --node-pools "${NODE}" \
  --working-dir /scratch \
  --environment HF_HOME=/scratch/hf_cache \
  --environment HF_HUB_ENABLE_HF_TRANSFER=1 \
  --environment WANDB_DIR=/scratch/wandb \
  --existing-pvc "claimname=${SCRATCH_PVC},path=/scratch" \
  --existing-pvc "claimname=${SHARED_RO_PVC},path=/shared-ro" \
  --existing-pvc "claimname=${SHARED_RW_PVC},path=/shared-rw" \
  --command -- /bin/bash -lc "\
    mkdir -p /scratch/hf_cache /scratch/wandb && \
    ln -sf \"\$(command -v python3)\" /usr/local/bin/python && \
    cd /scratch && \
    jupyter lab \
      --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
      --ServerApp.root_dir=/scratch \
      --ServerApp.token=\${JUPYTER_TOKEN:-cs552}"

cat <<EOF

>>> Job submitted: ${JOB_NAME}

Watch it start:    runai describe job ${JOB_NAME} -p ${PROJECT}
Stream logs:       runai logs -f ${JOB_NAME} -p ${PROJECT}
When Running:      runai port-forward ${JOB_NAME} --port 8888:8888 -p ${PROJECT}
Then open:         http://localhost:8888  (token: cs552)
Shell in pod:      runai bash ${JOB_NAME} -p ${PROJECT}
Stop the job:      runai delete job ${JOB_NAME} -p ${PROJECT}

Other connection options:
  Jupyter Lab:
    Wait until the job is Running, use the port-forward command above,
    then open http://localhost:8888.
    This is best for notebooks, plots, and milestone work.

  Shell:
    runai bash ${JOB_NAME} -p ${PROJECT}
    This is useful for nvidia-smi, checking files, installing a temporary
    package, or running scripts without opening Jupyter.

  VS Code:
    In VS Code, install the Microsoft Kubernetes and Remote Development
    extensions. In the Kubernetes sidebar, find pod ${JOB_NAME}-0-0 and
    choose "Attach Visual Studio Code". Then open your repo folder inside
    the pod and use Terminal -> New Terminal for a shell.

WHEN YOU'RE DONE: \`runai delete job ${JOB_NAME} -p ${PROJECT}\`. Idle sessions
take a GPU away from the rest of the course.

Run:AI note:
  If you see a transient [PodGrouperWarning] that says the PodGroup
  "object has been modified", wait and run the describe command again.
  This is a scheduler reconciliation warning, not a submit.sh syntax
  error. The job is only stuck if it remains Pending after a few minutes.

Storage inside the pod:
  /scratch             group scratch PVC (${SCRATCH_PVC}) — your group's primary workspace, RW
  /shared-ro/datasets  course datasets (read-only)
  /shared-ro/models    course models   (read-only)
  /shared-rw           shared with ALL students — be careful what you write

Deliverable notebooks:
  Keep notebooks/<first_name>_<last_name>_<sciper>.ipynb files in your
  git repo and commit them.
  Keep the project-wide notebooks/submit.sh in the same notebooks/
  directory. This script is intended to be the basis for that deliverable
  in most cases. GROUP must be set to your team group in the submitted
  file. GASPAR may be changed by whoever launches the job; TAs can
  replace it before grading.
  Use /scratch for caches, checkpoints, and large generated files, not
  as a submission location.
EOF
