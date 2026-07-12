# Environment Setup

ClawVLA deliberately separates training, simulator, model-serving, and action-policy processes. Do not merge the CUDA
stacks into one environment: the environments communicate through subprocesses, HTTP, and archived trajectory files.

## Path layout

Run repository-local commands from the checkout root:

```bash
cd /path/to/EmbodiedSkills
export CLAWVLA_ROOT="$PWD"
```

The checked-in lab configs contain absolute paths. The current mirror layout is:

```text
/mnt/linyutong/wangwai_mirror/vla/clawvla   ClawVLA checkout
/mnt/linyutong/wangwai_mirror/RoboTwin      RoboTwin checkout
/mnt/linyutong/wangwai_mirror/pi0.5         training-aligned OpenPI checkout
/mnt/wangwai/weights                         shared model weights
/mnt/wangwai/vla/CALVIN                     CALVIN checkout and dataset
```

If a checkout uses a different layout, update the absolute paths in the selected JSON/YAML profile. Paths under
`/mnt/linyutong` and `/mnt/zrh` are machine-specific, not portable defaults.

## Environment map

| Environment | Python | Responsibility | ClawVLA requirements |
| --- | --- | --- | --- |
| `robotwin-py312` | 3.12 | Main runtime, RoboTwin and LIBERO rollout, data/eval tools | `requirements/robotwin-py312.txt` |
| `vllm` | environment-owned | Local Qwen3-VL OpenAI-compatible service | `requirements/vllm.txt` |
| `openpi-torch-py312` | 3.12 | Frozen pi0.5/OpenPI worker | `requirements/openpi-torch-py312.txt` |
| `.venv-openrlhf-py310-cu128` | 3.10 | OpenRLHF, vLLM, DeepSpeed, Ray training process | `requirements/openrlhf-py310-cu128.txt` |
| `groot-py312` | 3.12 | RoboCasa rollout and frozen GR00T worker | existing RoboCasa/LeRobot/GR00T environment |
| `calvin-py38` | 3.8 | CALVIN rollout subprocess and task oracle | `requirements/calvin-py38.txt` |

LIBERO uses `robotwin-py312`; it does not require a seventh ClawVLA environment.

## Installation

### RoboTwin and LIBERO runtime

Start from an existing RoboTwin environment that already contains the compatible CUDA, PyTorch, SAPIEN, RoboTwin,
LIBERO, and simulator dependencies:

```bash
conda activate robotwin-py312
cd "$CLAWVLA_ROOT"
python -m pip install -r requirements/robotwin-py312.txt
```

`h5py` and OpenCV are direct dependencies of the expert-subtask collection and merge tools. Torch is intentionally not
installed by this file because replacing it can break the simulator CUDA stack.

### Local vLLM service

```bash
conda activate vllm
python -m pip install -r "$CLAWVLA_ROOT/requirements/vllm.txt"
```

### OpenPI/pi0.5 worker

```bash
conda activate openpi-torch-py312
cd "$CLAWVLA_ROOT"
python -m pip install -r requirements/openpi-torch-py312.txt
export PYTHONPATH="$CLAWVLA_ROOT/src:/mnt/linyutong/wangwai_mirror/pi0.5/src${PYTHONPATH:+:$PYTHONPATH}"
```

The direct PyTorch inference path imports `sentencepiece` and the training-aligned OpenPI source. The selected 25k
configuration expects the checkpoint and tokenizer paths declared in `configs/robotwin_pi05_subtasks_25k.json`.

### OpenRLHF trainer

Use Python 3.10 with the CUDA 12.8/Torch 2.8 wheel set declared by the requirements file:

```bash
cd "$CLAWVLA_ROOT"
python3.10 -m venv .venv-openrlhf-py310-cu128
.venv-openrlhf-py310-cu128/bin/python -m pip install -U pip
.venv-openrlhf-py310-cu128/bin/python -m pip install -r requirements/openrlhf-py310-cu128.txt
```

ClawVLA is imported through `PYTHONPATH` by `scripts/run_clawvla_rl.sh`; the Python 3.10 trainer environment does not
install the Python-3.12 package metadata with `-e .`.

### RoboCasa/GR00T

Use the existing `groot-py312` environment prepared by RoboCasa, robosuite, LeRobot, and GR00T:

```bash
conda activate groot-py312
export PYTHONPATH="$CLAWVLA_ROOT/src:/mnt/wangwai/lerobot/src:/mnt/wangwai/RoboCasa:/mnt/wangwai/robosuite"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d
```

See [RoboCasa + GR00T Integration Notes](robocasa_groot.md) for checkpoint and action-schema details.

### CALVIN/X-VLA

Start from the existing CALVIN Python 3.8 environment:

```bash
conda activate calvin-py38
python -m pip install -r "$CLAWVLA_ROOT/requirements/calvin-py38.txt"
export PYTHONPATH="$CLAWVLA_ROOT/src:/mnt/wangwai/vla/CALVIN/calvin_env:/mnt/wangwai/vla/CALVIN/calvin_models"
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d
```

Do not run `pip install -e .` in this environment: `pyproject.toml` targets Python 3.12+. The CALVIN subprocess imports
the source checkout through `PYTHONPATH`. See [CALVIN + X-VLA Integration Notes](calvin_xvla.md).

## Proxy and local services

When HTTP proxies are enabled, always bypass them for local policy/action services:

```bash
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
```

Without this setting, local PolicyProxy, vLLM, OpenPI, or X-VLA requests can be sent to the cluster proxy and return a
misleading HTTP 503.

## Verification

Run the dependency-light suite in `robotwin-py312`:

```bash
cd "$CLAWVLA_ROOT"
NO_PROXY=127.0.0.1,localhost PYTHONPATH=src \
  /mnt/wangwai/miniconda3/envs/robotwin-py312/bin/python -m pytest -q
```

Check the environment-specific bridges without starting a simulator:

```bash
PYTHONPATH="$CLAWVLA_ROOT/src:/mnt/wangwai/vla/CALVIN/calvin_env:/mnt/wangwai/vla/CALVIN/calvin_models" \
  /mnt/wangwai/miniconda3/envs/calvin-py38/bin/python -c \
  "from clawvla.envs.calvin import CalvinAdapter; from clawvla.action_backends.calvin import CalvinHttpActionBackend"

PYTHONPATH="$CLAWVLA_ROOT/src" \
  /mnt/wangwai/miniconda3/envs/openpi-torch-py312/bin/python -c "import sentencepiece"
```
