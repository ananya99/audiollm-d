# CS-552 custom image template
#
# Most teams should use the default course image directly and do not need
# this Dockerfile. It is provided as a starting point only if your project
# genuinely needs a custom image with extra system packages or libraries.
#
# Built on the official vLLM image: CUDA + PyTorch + vLLM, with
# FlashInfer and bitsandbytes available. We add the Python NLP stack
# (transformers, TRL, PEFT, datasets, JupyterLab, eval/RAG libs) on top.
#
# Pin VLLM_TAG to a specific version, never `:latest` or
# `:latest-ubuntu2404` (those move with every vLLM release).
# Tag list: https://hub.docker.com/r/vllm/vllm-openai/tags
ARG VLLM_TAG=v0.11.0
FROM vllm/vllm-openai:${VLLM_TAG}

# The vllm-openai image's default ENTRYPOINT launches the OpenAI server.
# Override so the container behaves like a normal Linux box (Jupyter,
# bash, training scripts, etc).
ENTRYPOINT []

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

# vllm-openai base ships only minimal apt packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs vim less tmux htop curl wget unzip \
        build-essential ca-certificates openssh-client \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Do NOT reinstall torch or vllm — base image has compatible versions.
RUN pip install --upgrade pip && pip install \
    "transformers>=4.46" \
    "tokenizers>=0.20" \
    "accelerate>=1.0" \
    "trl>=0.12" \
    "peft>=0.13" \
    "datasets>=3.0" \
    "evaluate" \
    "huggingface_hub[cli,hf_transfer]" \
    "safetensors"

# RAG / retrieval
RUN pip install \
    "sentence-transformers>=3.0" \
    "faiss-cpu" \
    "rank-bm25" \
    "langchain" "langchain-community"

# Evaluation / utilities
RUN pip install \
    "lm-eval>=0.4.5" \
    "rouge-score" "sacrebleu" "bert-score" \
    "scikit-learn" "scipy" "pandas" "numpy" \
    "matplotlib" "seaborn" "plotly"

# Notebook + experiment tracking
RUN pip install \
    "jupyterlab>=4.2" "ipykernel" "ipywidgets" "jupyterlab-widgets" \
    "wandb" "tensorboard" \
    "tqdm" "rich" "omegaconf" "hydra-core" "pyyaml"

RUN python3 -c "import torch, vllm, transformers, trl, peft, datasets; \
    print('torch', torch.__version__, 'cuda', torch.version.cuda); \
    print('vllm', vllm.__version__); \
    print('transformers', transformers.__version__)"

WORKDIR /workspace
CMD ["/bin/bash"]
