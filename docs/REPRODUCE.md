# Reproducing ThermalBench

ThermalBench compares models only after fixing the data, split, labels, physical channels, and metric code. Architecture-specific optimization recipes remain architecture-specific and are recorded in [`exp/exp_basic.py`](../exp/exp_basic.py) and the Therm-FM YAML files.

## Before running

1. Create the [frozen environment](INSTALL.md).
2. Place the [datasets and checkpoints](DATASETS.md) at the repository root.
3. Run `python script/smoke_test.py`.

## Evaluation tracks

| Track | Train support | Test support | Target labels used at test time |
|---|---|---|---:|
| S1 | original source-task protocols | same fixed physical designs | source-defined |
| S2 | Cases 1–10, layout variation | represented cases and support | 0 |
| S3 | S2 + material variation | represented cases and support | 0 |
| S4 | S3 + boundary variation | represented cases and support | 0 |
| S5 zero-shot | frozen S4 model | unseen Cases 16–20 | 0 |
| S5 few-shot | frozen S4 model + adaptation pool | held-out samples from Cases 16–20 | K per case |

S2–S4 are independently generated rather than sample-wise paired perturbations. Their comparison measures a progressive change in distributional difficulty, not a strict causal ablation of one variable.

The commands below currently cover S2–S5. S1's [complete source-suite result record](S1_RESULTS.md) is available, while its canonical data package and one-command evaluator are on the way.

## One model

```bash
# Evaluate a released checkpoint
python run.py --model UFNO --data level2 --task test

# Train with the registered benchmark recipe, then evaluate
python run.py --model UFNO --data level2 --task train

# Structural OOD: automatically use the S4 checkpoint
python run.py --model UFNO --data level5 --task test

# Adapt with 10 labeled samples per unseen case
python run.py --model UFNO --data level5 --task finetune --shots 10
```

Accepted model names are:

```text
FNO  UFNO  SAUFNO  UNet  DeepONet  ThermFM-T  ThermFM-B  ThermFM-L
```

## All baselines

```bash
# S2–S5 evaluation and summary
bash script/test_all.sh

# Registered S2/S3/S4 training recipes
bash script/train_all.sh

# S5 K-shot curves
bash script/finetune_all.sh
```

Model-specific wrappers also accept a scope token or shot list:

```bash
bash script/UFNO/train.sh level2
bash script/UFNO/test.sh level4 level5
bash script/UFNO/finetune.sh 10 50 100 250 500
```

For concurrent Therm-FM jobs, pass distinct Accelerate ports. For example:

```bash
python run.py \
  --model ThermFM-T \
  --data level4 \
  --task train \
  --gpus 0,1,2,3 \
  --port 29815
```

## Selected reference results

These paper-preview values record the strongest method in each track:

| Track | Best method | Best RMSE ↓ | Interpretation |
|---|---|---:|---|
| S1 | Therm-FM L | 0.009–0.076 K | task-specific source results; not pooled |
| S2 | SAU-FNO | 0.657 K | layout/configuration diversity is learnable in support |
| S3 | U-FNO | 0.802 K | adding material variation causes moderate degradation |
| S4 | U-FNO | 1.327 K | multi-physics variation remains tractable in support |
| S5 zero-shot | Therm-FM T | 15.99 K | unseen system structure causes a qualitative failure |
| S5 10-shot | Therm-FM B | 3.19 K | 50 target labels total recover much of the gap |

S1 preserves eight source protocols; see [S1_RESULTS.md](S1_RESULTS.md) for cases, resolutions, RMSE, and MAE. Small last-digit differences in S2–S5 can arise from GPU kernels and execution environments. A valid reproduction should preserve the split, model recipe, normalization mode, and metric implementation before attributing differences to a method.

## Metrics

All baselines call [`utils/metrics.py`](../utils/metrics.py). Do not replace it with an “equivalent” local implementation when reporting ThermalBench numbers.

| Reported metric | Meaning |
|---|---|
| RMSE ↓ | per-sample field RMSE, then averaged |
| MAE ↓ | per-sample field MAE, then averaged |
| R² ↑ | pooled over every test pixel |
| MaxAE ↓ | worst single-pixel absolute error in a field, then averaged |
| Peak-temperature error ↓ | absolute difference between predicted and true maximum temperatures |
| Top-50 MAE ↓ | MAE at the 50 hottest ground-truth pixels |

MaxAE and peak-temperature error are different: the largest pointwise error may occur away from the true or predicted hotspot.

Summarize saved results with:

```bash
python utils/summarize.py
python utils/summarize.py level2 level5
```

Pass `--output path/to/result.json` to `run.py` when integrating with another experiment manager.

## Reproduction boundaries

- Every released baseline uses temperature-field labels; this is a supervised benchmark, not a PINN-training comparison.
- S5 zero-shot evaluation does not update model weights or normalization statistics.
- Therm-FM starts from Poseidon for training; evaluating a released checkpoint does not require Poseidon.
- `--epochs`, `--batch_size`, `--lr`, or `--per_channel_norm 0` create a new experimental recipe and should not be presented as the released reproduction.
- Report dataset generation, conversion, and reference-solver time separately from model training/inference time.

## Output locations

```text
checkpoints/{data}_{model}/...                 trained models
checkpoints/fewshot/{data}_{model}_k{K}.json  adaptation curves
results/...                                   evaluation JSON/plots when requested
```

Few-shot JSON files include the frozen baseline, per-epoch curve, and final metrics. This makes the adaptation result auditable without rerunning every epoch.
