#!/bin/bash
# Build & push a CS-552 custom image. Run from this directory.
#
# Most teams should use the default course image directly and do not need
# this script. It is provided as a starting point only if your project
# genuinely needs a custom image with extra system packages or libraries.
#
# Prereqs:
# - `docker login registry.rcp.epfl.ch` with your EPFL credentials.
# - Create your own PUBLIC Harbor project first. Recommended project name:
#   cs-552-2026-project-<group-name>
#
# Usage:
#   ./build.sh           # build & push :v1
#   ./build.sh v2        # build & push :v2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REGISTRY="registry.rcp.epfl.ch"
PROJECT="cs-552-2026-project-gXX"  # <-- your public Harbor project
IMAGE="cs552-custom"
TAG="${1:-v1}"

# Pin to a specific upstream tag, never `:latest`.
VLLM_TAG="${VLLM_TAG:-v0.11.0}"

FULL="${REGISTRY}/${PROJECT}/${IMAGE}:${TAG}"

if [[ "${PROJECT}" == "cs-552-2026-project-gXX" || -z "${PROJECT}" ]]; then
  echo "ERROR: edit build.sh and set PROJECT to your public Harbor project." >&2
  echo "Recommended format: cs-552-2026-project-<group-name>" >&2
  exit 1
fi

echo ">>> Building ${FULL} on top of vllm/vllm-openai:${VLLM_TAG}"
docker build \
  --pull \
  --platform linux/amd64 \
  -f "${SCRIPT_DIR}/Dockerfile" \
  --build-arg "VLLM_TAG=${VLLM_TAG}" \
  -t "${FULL}" \
  "${SCRIPT_DIR}"

echo ">>> Pushing ${FULL}"
docker push "${FULL}"

echo ">>> Done. Students pull: ${FULL}"
