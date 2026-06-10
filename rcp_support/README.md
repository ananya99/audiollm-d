# CS-552 Project — RCP Quick Start

This README focuses on the compute environment and the files needed to
run and grade the code. For the official project scope, grading weights, 
rubrics, report requirements, and deadlines, refer to the
[CS-552 Open Project description](https://docs.google.com/document/d/1NI4UKsasYuFLxOGGzsAweCbW0XOEtc59/edit).

The provided setup launches an interactive Jupyter Lab job on the RCP
cluster with PyTorch, vLLM, and the Hugging Face stack already installed.
No Docker building is required for most teams. The included `Dockerfile`
and `build.sh` are provided as a starting point for teams that genuinely
need a custom image.

This repo includes `submit.sh` as a starter Run:AI launcher for the Open
Project. It submits an interactive Jupyter pod with the course image,
mounts your group scratch and shared course storage, and starts Jupyter
in `/scratch`. In most cases, this script is intended to be the basis for
the project-wide `notebooks/submit.sh` deliverable. For submission, copy it to
`notebooks/submit.sh` (one for the whole team), set `GROUP` to your real team group, and submit it
with your code so TAs can launch the same environment for grading. 
Don't worry about `GASPAR` in the submitted file; TAs can replace it with their 
own username for grading.

This repo also includes `submit_train.sh` as an example helper for longer
Run:AI training jobs. It is useful when you want a command to run and
exit without starting Jupyter. It is **not** the notebook deliverable;
submit `notebooks/submit.sh` for grading.

---

## TL;DR

1. Connect to the **EPFL VPN**, then install the Run:AI CLI and `runai login`.
2. **Set `GROUP="gXX"`** to your real team number, for example `GROUP="g07"`. For your own test
   runs, replace `GASPAR="gaspar"` with your EPFL username. For the file
   you submit, do not worry if `GASPAR` is still `gaspar`: TAs can change
   it to their own username before grading. `GROUP` must be correct.
3. `./submit.sh`
4. Connect to the pod (pick one):
   - **Jupyter:** wait until the job is `Running`, then
     `runai port-forward <job-name> --port 8888:8888`, and open
     `http://localhost:8888` (token: `cs552`)
   - **Shell:** `runai bash <job-name>`
   - **VS Code:** attach via the Kubernetes extension — see below.
5. **When you stop working: `runai delete job <job-name>`.**

---

## Table of Contents

- [One-time setup](#one-time-setup): VPN, Run:AI CLI, login, and project context.
- [Launch a job](#launch-a-job): submit the Jupyter pod and check job status.
- [Troubleshooting startup](#if-the-job-is-pending-or-crashes): pending jobs, crashes, logs, and vLLM/CUDA notes.
- [Connecting to your pod](#connecting-to-your-pod): Jupyter Lab, shell, and VS Code options.
- [Image and packages](#whats-in-the-image): what's already installed and when to build a custom image.
- [Storage](#storage-layout-inside-the-pod): `/scratch`, shared storage, and optional personal home PVC.
- [GPU etiquette](#gpu-etiquette-please-read): deleting idle jobs and avoiding wasted GPUs.
- [Training jobs](#example-training-job): optional `submit_train.sh` helper, not a deliverable.
- [What you turn in](#what-you-turn-in): notebook layout, `notebooks/submit.sh`, and grading expectations.

---

## One-time setup

1. **Connect to the EPFL VPN.** You must be on the EPFL VPN to submit
   jobs to the cluster. EPFL uses Cisco Secure Client; download it from
   [VPN clients available](https://www.epfl.ch/campus/services/en/it-services/network-services/remote-intranet-access/vpn-clients-available/),
   connect to `vpn.epfl.ch`, and sign in with your GASPAR credentials.
   If you hit VPN issues, see
   [Remote Intranet Access](https://www.epfl.ch/campus/services/en/it-services/network-services/remote-intranet-access/).
2. **Install the Run:AI CLI and log in.**
   - Sign in to <https://rcpepfl.run.ai/> with **Sign in with SSO**.
   - Download the CLI from the top-right help icon (`?`) →
     **Researcher command line interface** → your OS.
   - On macOS/Linux, make the binary executable and put it on your PATH:
     ```bash
     chmod +x ./runai
     sudo mv ./runai /usr/local/bin/runai
     ```
   - Download the kubeconfig from
     <https://wiki.rcp.epfl.ch/public/files/kube-config.yaml> and save it
     as `~/.kube/config`.
   - Configure the cluster and log in:
     ```bash
     runai config cluster rcp-caas-prod
     runai login
     ```
   If anything behaves differently on your setup, use the
   [RCP Quick Start](https://wiki.rcp.epfl.ch/home/CaaS/Quick_Start) as
   the canonical reference.
3. **Set your project context** (replace `<gaspar>` with your username):
   ```bash
   runai config project course-cs-552-<gaspar>
   ```
   `submit.sh` also passes this project explicitly when submitting the
   job, so it does not depend on any other default Run:AI project.
4. **Edit `submit.sh`** — set `GROUP="gXX"` to your real team
   number (e.g. `g07`). This is required for submission because it
   selects your team's scratch PVC. Replace `GASPAR="gaspar"` with your
   EPFL username only when you run the script yourself; TAs can replace
   `GASPAR` in the submitted file before grading. Other edits are optional: only change the
   image, mounts, non-secret environment variables, or command if your
   project needs it, and test those changes yourself before submitting.

Avoid setting individual tokens such as Hugging Face or Weights & Biases
credentials in the final `notebooks/submit.sh`. If a notebook needs a personal
token or other individual environment variable, define or read it inside
that notebook so the setup is tied to the notebook being graded.

## Launch a job

Each job runs on **1 GPU (40GB A100)** — the course cap for this
setup. Asking for more leaves the job stuck `Pending`. 

To start an interactive Jupyter Lab job from your submitted launcher, run:

```bash
./submit.sh           # default
./submit.sh exp1      # optional job-name suffix
```

Wait for `Running`:
```bash
runai list jobs
runai describe job <job-name>
```

In a second terminal, forward the port:
```bash
runai port-forward <job-name> --port 8888:8888
# Open http://localhost:8888 — token is "cs552"
```

This port-forward command is run after the job exists. Do not add
`--service-type portforward` to `runai submit`; some Run:AI CLI versions
reject it because `port-forward` is a client-side command.

If you see `address already in use` on `8888`, change the local side to
any free port:
```bash
runai port-forward <job-name> --port 9000:8888
# Open http://localhost:9000
```

When you're done **(read this please)**:
```bash
runai delete job <job-name>
```

### If the job is pending or crashes

These commands cover most cases:
```bash
runai list jobs                 # see your jobs and their current states
runai describe job <job-name>   # pending reason, events, current state
runai logs <job-name>           # stdout/stderr from the container
runai logs -f <job-name>        # follow live
```

If `runai logs` shows a Python traceback, this indicates that the pod
started correctly, so the next step is to fix the code, update the repo,
and resubmit.

If `runai list jobs` shows `Failed`, the pod is no longer running:
`runai bash` and `runai port-forward` cannot attach to it. Read the logs,
delete the failed job, then resubmit:

```bash
runai logs <job-name>
runai delete job <job-name>
./submit.sh
```

### vLLM and CUDA initialization issues

If you use vLLM in a notebook or Python script, initialize vLLM before
calling CUDA APIs through PyTorch. For example, avoid calling
`torch.cuda.is_available()` before constructing `LLM(...)` in the same
Python process. Otherwise, vLLM may fail with a CUDA multiprocessing
error such as `Cannot re-initialize CUDA in forked subprocess`.

If this happens, restart the Python process or notebook kernel and run
the vLLM setup first.

## Connecting to your pod

You have three ways to interact with a running pod. Use whichever fits
the task.

### 1. Jupyter Lab (the default)

Already covered above — wait until the job is `Running`, run
`runai port-forward`, then open `http://localhost:8888`. Best for
notebook-driven exploration, plots, and the milestone deliverables.

### 2. A shell in the pod (`runai bash`)

For quick CLI work — running scripts, checking GPU usage, installing
packages, debugging — you don't need Jupyter at all:

```bash
runai bash <job-name>
```

You're now inside the container with a normal shell. The starter
`submit.sh` sets the container working directory to `/scratch`, so
attached shells should start there. Useful examples:

```bash
nvidia-smi                              # check GPU state
df -h /scratch                          # how much scratch space is left
python my_script.py                     # run something quickly
pip install some-extra-package          # ad-hoc install for this session
```

You can have a Jupyter port-forward running in one terminal *and* a
`runai bash` open in another, on the same pod.

### 3. VS Code attached to the pod

If you prefer VS Code over Jupyter for editing code, you can attach VS
Code directly to your running pod and edit files inside it as if they
were local. Full setup guide from RCP:
<https://wiki.rcp.epfl.ch/home/CaaS/FAQ/how-to-vscode>

Short version:

1. **Install VS Code** from <https://code.visualstudio.com>.
   The official Microsoft build is required — VSCodium does **not**
   work with the Kubernetes attachment flow.
2. **Install two extensions** from the VS Code Marketplace, both from
   Microsoft:
   - [Kubernetes](https://marketplace.visualstudio.com/items?itemName=ms-kubernetes-tools.vscode-kubernetes-tools)
   - [Remote Development](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack)
   Be careful — there are several Kubernetes extensions on the
   Marketplace; only the official Microsoft one is supported.
3. **Set up your kubeconfig** at `~/.kube/config` per the
   [RCP Quick Start](https://wiki.rcp.epfl.ch/home/CaaS/Quick_Start).
   The Kubernetes extension reads it automatically.
4. **Attach to your pod.** Click the Kubernetes icon in the left
   sidebar, expand your namespace under the cluster, find your running
   pod (named `<job-name>-0-0`), right-click → **Attach Visual
   Studio Code**. A new VS Code window opens, connected to the pod.
   The bottom-left status bar shows which pod you're attached to.
5. **Open files**: File → Open Folder, then type a path inside the pod.
   Use `/scratch` for the normal course workspace. If you mounted your
   personal home PVC for private work, you can also open
   `/home/<gaspar>`.
6. **Open a terminal**: Terminal → New Terminal opens a shell in the
   pod, same as `runai bash`.

VS Code is the most ergonomic option for serious code editing during
the project. Jupyter is still the right tool for the deliverable
notebooks.

## What's in the image

The course image (`registry.rcp.epfl.ch/course-cs-552/base-vllm:v1`) is
built on the official vLLM image and ships with a broad set of common
NLP/LLM libraries. You probably will not need all of them; they are
included so most teams can use the same image and rely only on the
pieces relevant to their project.

Version snapshot for `base-vllm:v1`:

- **Core**: Python 3.12.11, CUDA 12.8, PyTorch 2.8.0+cu128, vLLM 0.11.0,
  FlashInfer 0.3.1, bitsandbytes 0.48.1
- **Training**: transformers 4.57.0, TRL 1.3.0, PEFT 0.19.1,
  accelerate 1.10.1
- **Data**: datasets 4.8.5, huggingface_hub 0.35.3, hf_transfer 0.1.9
- **RAG**: sentence-transformers 5.4.1, faiss-cpu 1.13.2, rank-bm25 0.2.2,
  langchain 1.2.15, langchain-community 0.4.1
- **Eval**: lm-eval-harness 0.4.11, rouge-score 0.1.2, sacrebleu 2.6.0,
  bert-score 0.3.13
- **Tracking**: wandb 0.26.1, tensorboard 2.20.0
- **Notebook**: jupyterlab 4.5.6, ipywidgets 8.1.8

## I need a package that isn't in the image

Three options, in order of preference:

1. **`pip install` from a notebook cell** — works for the session, takes
   seconds. Fine for one-off experiments.
2. **`requirements.txt` in your repo** — keep a `pip install -r
   requirements.txt` cell at the top of your notebook. Works for the
   grader too.
3. **Build your own image.** If your project genuinely needs something
   that can't be pip-installed (custom CUDA kernels, weird system libs),
   you can use the included `Dockerfile` and `build.sh` as a base. Make sure, 
   you first create your own Harbor project at <https://registry.rcp.epfl.ch/harbor/projects>
   and push your image there. The Harbor project **must be public** so
   TAs can pull the image during grading. There is no strict naming requirement 
   for Harbor projects or images, but we recommend naming your project
   `cs-552-2026-project-<group-name>` so course images are easier to
   identify. More information about pushing images to the RCP Harbor
   registry is available in the
   [RCP registry guide](https://wiki.rcp.epfl.ch/home/CaaS/FAQ/how-to-registry).
   
   Additional resources for building containers
   and images:
   [part 1](https://wiki.rcp.epfl.ch/home/CaaS/FAQ/how-to-build-a-container-part1)
   and
   [part 2](https://wiki.rcp.epfl.ch/home/CaaS/FAQ/how-to-build-a-container-part2).
   The initial build can take a long time (about 30 minutes), so plan
   for that.

Stick with the course image unless you have a concrete reason to build a
custom one.

## Storage layout inside the pod

| Path | What it is | Access |
|---|---|---|
| `/scratch` | Your group's scratch PVC (`course-cs-552-scratch-gXX`) | shared with your group and TAs, RW |
| `/shared-ro/datasets` | Course datasets | read-only, all students |
| `/shared-ro/models` | Course base models | read-only, all students |
| `/shared-rw` | Course-wide writable scratch | RW for **everyone** — careful |

`RW` means read-write access.

Inside the pod, `/scratch` is not a global scratch folder. It is your
group-specific scratch volume: `course-cs-552-scratch-gXX`, backed by
`/mnt/course-cs-552/rcp-caas-cs-552-gXX/scratch-gXX` on the RCP
jumphost.

RCP also provides a personal `home` PVC in each student namespace
(`home`, pointing to `/home/<gaspar>`), but the starter `submit.sh` does
not mount it. That PVC requires the correct UID/GID inside the container
and is not accessible to TAs, so do not use it for files that need to be
graded.

**Use `/scratch` for everything heavy** — clone your repo there, save
model checkpoints, store the HF cache, log wandb runs. The starter
`submit.sh` sets `HF_HOME=/scratch/hf_cache` and
`WANDB_DIR=/scratch/wandb`, so downloaded models and datasets will land
there automatically and your teammates in the same group will see the
cached files.

`/scratch` is accessible to TAs through the group storage, but it is not
where you submit files. Keep deliverable notebooks and code in your repo,
not as loose files in `/scratch`. It is fine to edit a repo clone that
lives under `/scratch`, but the notebook files must be committed under
`notebooks/` so the graders get them from a clean clone.

Notebooks should be self-contained from a clean clone of the repository:
they should not depend on manual setup or files that exist only on your
laptop, in your personal home directory, or in `/scratch`. If a notebook
needs external files, download or generate them in the notebook and save
large outputs to `/scratch`, not to `/home/<gaspar>`.

> ⚠️ `/shared-rw` is writable by everyone in the course. Don't put
> anything sensitive there, and don't rely on files in it persisting —
> anyone with course access can overwrite or delete them.

> ℹ️ Anything in `/scratch` will be wiped end of July 2026.

### Setting up personal home PVC

For private experiments, you can mount your personal `home` PVC by adding
an extra PVC mount and running the container with your EPFL UID/GID, for
example:

```bash
runai submit \
  --name my-private-test \
  -p course-cs-552-<gaspar> \
  --image <your-image> \
  --gpu 1 \
  --run-as-uid <your-uid> \
  --run-as-gid <your-gid> \
  --existing-pvc "claimname=home,path=/home/<gaspar>" \
  --command -- /bin/bash -lc "cd /home/<gaspar> && bash"
```

This is only for your own debugging or development. Do **not** include a
personal `home` PVC mount in the `submit.sh` that you submit for
grading. TAs cannot access your personal home PVC, and a submitted
notebook must run from the repo plus the course/group storage mounted by
the starter script.

## GPU etiquette (please read)

The course setup supports up to **75 groups** and **up to 300 students**,
with a 40GB A100-type GPU allocation per group. The scheduler caps each
allocation at 1 GPU at a time. Interactive jobs are meant for
development sessions and have a limited duration; training jobs are for
longer compute runs. A few habits keep things working for everyone,
especially around the May 24 and June 7 deadlines:

- **Delete idle Jupyter jobs.** If you walk away from your laptop for
  more than ~30 minutes, `runai delete job <name>`. You can resubmit in
  ~5 seconds when you come back.
- **Use `--interactive` for exploration, debugging, and notebooks.**
  This is what `submit.sh` does by default. Interactive jobs are
  high-priority, limited-duration jobs for development, not long training
  runs.
- **For long final training runs, submit a Run:AI training job** with a
  separate script or command. The included `submit_train.sh` is an
  example helper for this; it is **not** a deliverable and should not be
  submitted next to your notebook. Training jobs are lower priority and
  can be preempted and restarted by the scheduler, so your code must
  write checkpoints to `/scratch` and resume from them. See the
  [RCP Run:AI guide](https://wiki.rcp.epfl.ch/home/CaaS/FAQ/how-to-use-runai)
  for the differences between interactive and training jobs.
- **Expect queues during deadline week.** Plan compute-heavy work
  earlier, not the night before.

### Example Training Job

Use `submit.sh` as base for the notebook deliverable. Use
`submit_train.sh` only as an example for longer training or compute runs
that should execute a command and exit when finished.

Before using it, edit:

- `GASPAR` and `GROUP`
- `TRAIN_COMMAND`, for example:
  ```bash
  TRAIN_COMMAND='cd /scratch/my-repo && python train.py --output-dir /scratch/runs/train-v1'
  ```
- `IMAGE`, only if you intentionally use a custom image

Run it with:

```bash
./submit_train.sh
runai logs -f <job-name> -p course-cs-552-<gaspar>
```

Do not submit `submit_train.sh` as the graded launcher. TAs expect
<u>`notebooks/submit.sh`</u> (**one for the whole team**) to launch the notebook environment.

## What you turn in
 
The starter `submit.sh` in this repo is an example launcher for the RCP
environment and is intended to be the basis for the project-wide
`notebooks/submit.sh` deliverable in most cases. This one script should
launch the environment in which all submitted notebooks run. The full
deliverable checklist and rubrics are in the
[Open Project description](https://docs.google.com/document/d/1NI4UKsasYuFLxOGGzsAweCbW0XOEtc59/edit).

Use this structure for the notebook submissions:

```text
notebooks/
  submit.sh
  <first_name>_<last_name>_<sciper>.ipynb
  <first_name>_<last_name>_<sciper>.ipynb
  <first_name>_<last_name>_<sciper>.ipynb
  <first_name>_<last_name>_<sciper>.ipynb
```

Each notebook should be a **standalone file**. It should run from a clean
clone of the repo in the environment launched by `notebooks/submit.sh`.
If a notebook needs individual environment variables, tokens, or setup
steps, define or read them inside that notebook; do not put them in the
project-wide `notebooks/submit.sh`. Any required individual tokens must
remain active until at least mid-July 2026, or until grading is complete.
If it needs models, datasets, or generated files, download or create them
in the notebook and store large files under `/scratch`. Do not expect TAs
to find or grade anything that exists only in `/scratch`.
 
**Milestone 2 — Preliminary Results (May 24)**
- `notebooks/submit.sh`, with `GROUP` set to your team number. This
  should be the project-wide setup and must work for every submitted
  notebook. `GASPAR` is only the launcher username; TAs can replace it
  before grading, but they should not have to fix `GROUP`.
- `notebooks/<first_name>_<last_name>_<sciper>.ipynb` — one notebook per
  teammate. For this milestone, the notebook content is not graded, but
  you are strongly encouraged to include initial results you expect to
  develop further for the final notebook, such as graphs, tables, or
  examples.
- A 1-page group progress report, per the Open Project description.

**Milestone 3 — Final Submission (June 7)**
- `notebooks/submit.sh`, kept up to date with the correct `GROUP`, and
  still working for every submitted notebook. `GASPAR` may be changed by
  whoever launches the job for grading.
- `notebooks/<first_name>_<last_name>_<sciper>.ipynb` (one per teammate,
  deeper analysis — error analysis, ablations, attention viz, etc.).
- The 4-page report and full project code, per the project handout.


In both cases, your notebooks should preferably load models and datasets from
Hugging Face where relevant, and **must run inside the pod produced by
`notebooks/submit.sh`** — that's how TAs grade them.
For proposal, literature review, progress report, final report, and
rubric details, use the [Open Project description](https://docs.google.com/document/d/1NI4UKsasYuFLxOGGzsAweCbW0XOEtc59) as the source of truth.
 
### Modifying `submit.sh`
 
You can use the starter `submit.sh` with only the required `GROUP` edit
in the submitted file in most cases. Set `GASPAR` when you run it
yourself; TAs can replace `GASPAR` for grading. Modify anything else only
if your project genuinely needs something different — for example
pointing at a custom image you built, mounting additional PVCs, or
changing environment variables. **Both are allowed.**
 
> ⚠️ If your `submit.sh` doesn't work, you get **zero
> code points** (i.e., individual notebook points) for that milestone. 
> When grading the project, the TAs will not debug your `submit.sh` 
> or your image. You are responsible for making sure it works from a clean
> clone of your repo, and that it launches a working pod where your
> notebooks execute without errors. 
 
So:
 
- **Default path** (most teams): leave `submit.sh` alone except for
  the `GROUP="gXX"` line in the submitted file. Set `GASPAR` only for
  your own runs. The course image already covers fine-tuning,
  inference, RAG, and evaluation.
- **Custom path** (advanced teams): if you change the image, mounts,
  or anything else, test that a clean clone of your repo + your
  modified `submit.sh` actually launches a working pod and your
  notebooks execute. Run it yourself from a fresh terminal before
  submitting.
