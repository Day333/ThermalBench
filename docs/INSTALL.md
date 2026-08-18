# Installation

The released ThermalBench checkpoints were produced with Python 3.10, PyTorch 2.0.1 + CUDA 11.8, Transformers 4.29.2, and Accelerate 0.31.0. Therm-FM uses the Hugging Face Trainer stack and is sensitive to changes across these versions.

## Recommended environment

On a Linux machine with an NVIDIA driver compatible with CUDA 11.8:

```bash
conda env create -f environment.yml
conda activate thermalbench
python script/smoke_test.py
```

The smoke test does not download data or checkpoints. It verifies imports, constructs the shared metric pipeline, and runs a synthetic U-Net forward pass on CPU.

## Manual installation

Use this only when Conda cannot consume `environment.yml`:

```bash
conda create -n thermalbench python=3.10 -y
conda activate thermalbench

pip install torch==2.0.1+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118

pip install \
  transformers==4.29.2 \
  accelerate==0.31.0 \
  h5py==3.16.0 \
  numpy==1.26.4 \
  pandas==2.3.3 \
  scikit-learn==1.7.2 \
  pyyaml==6.0.3 \
  matplotlib==3.10.9 \
  psutil \
  wandb==0.27.2
```

## Hardware

- FNO, U-FNO, SAU-FNO, U-Net, and DeepOHeat use one GPU in the released scripts.
- Therm-FM training uses Hugging Face Trainer + Accelerate and the released recipes target four GPUs.
- Evaluation can use a different visible GPU by setting `CUDA_VISIBLE_DEVICES` or passing `--gpus` for Therm-FM.
- The public checkpoints can be evaluated without downloading Poseidon. Poseidon-T/B/L is required only to reproduce Therm-FM training.

Place the upstream Poseidon weights at:

```text
pretrained/
├── Poseidon-T/
├── Poseidon-B/
└── Poseidon-L/
```

The weights are available from [camlab-ethz on Hugging Face](https://huggingface.co/camlab-ethz).

## Version-sensitive components

Do not upgrade these packages when reproducing the released numbers:

| Package | Frozen version | Why |
|---|---:|---|
| PyTorch | 2.0.1+cu118 | checkpoint and optimizer compatibility |
| Transformers | 4.29.2 | Therm-FM/scOT Trainer interface |
| Accelerate | 0.31.0 | Therm-FM distributed launch behavior |
| NumPy | 1.26.4 | tested tensor and metric behavior |

Newer versions can be useful for development, but results from a changed environment should be reported as a separate execution environment rather than compared as a bitwise reproduction.

## Common checks

```bash
# Verify the command surface without loading a dataset
python run.py --help

# Verify the environment and a synthetic forward pass
python script/smoke_test.py

# Verify Accelerate before Therm-FM training
accelerate env
```

If `--task test` reports a missing checkpoint, confirm that the downloaded archive was unpacked into `checkpoints/` and that the folder names were preserved. S5 evaluation resolves S4 weights by design; it does not train an S5 checkpoint.
