<div align="center">

# IC-ThermBench

[![English](https://img.shields.io/badge/Language-English-2563eb?style=for-the-badge)](README.md)
[![简体中文](https://img.shields.io/badge/%E8%AF%AD%E8%A8%80-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-dc2626?style=for-the-badge)](README.zh-CN.md)

An open benchmark for reproducible, generalizable 2.5D/3D-IC thermal learning.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21992816.svg)](https://doi.org/10.5281/zenodo.21992816)
[![Code: MIT](https://img.shields.io/badge/Code-MIT-2563eb.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-f97316.svg)](LICENSE-DATA)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](environment.yml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C?logo=pytorch&logoColor=white)](docs/INSTALL.md)
[![S2–S5 Data](https://img.shields.io/badge/S2--S5%20data-released-16a34a.svg)](https://drive.google.com/file/d/15Do8Raf070VseV9cn44j1hdVpD3Rz-Un/view?usp=sharing)
[![S1 Pipeline](https://img.shields.io/badge/S1%20pipeline-on%20the%20way-f59e0b.svg)](docs/RESULTS.md)
[![Checkpoints](https://img.shields.io/badge/Checkpoints-released-16a34a.svg)](https://drive.google.com/file/d/1iP6dH3N_s1KzaOugxwe9aK_apO6DkXTD/view?usp=sharing)

[**Quick start**](#quick-start) · [**Datasets**](docs/DATASETS.md) · [**Results**](docs/RESULTS.md) · [**Reproduce**](docs/REPRODUCE.md) · [**Add a model**](docs/ADD_A_MODEL.md) · [**Paper preview**](docs/PAPER_PREVIEW.md)

</div>

![IC-ThermBench: five progressive generalization scopes](assets/ic-thermbench-overview.svg)

Thermal prediction papers often differ in data, simulators, splits, preprocessing, and metrics, making model-to-model comparison surprisingly fragile. Moreover, most existing datasets and implementations are not publicly available, and the community still lacks a unified evaluation standard. IC-ThermBench fixes that evaluation contract. It provides progressive physical support, immutable splits, eight baselines from three model families, and one interface for training, inference, adaptation, and reporting.

Developed and maintained by the IC-ThermBench research team at the **University of Technology Sydney (UTS)**.

## Why IC-ThermBench

Recent methods study geometry, material, cooling, and unseen systems. The remaining problem is comparability. Results are commonly produced with different private or partially released datasets, reference solvers, representations, splits, and metrics. Consequently, a lower error may reflect an easier test distribution rather than a better model, and “generalization” may refer to anything from a new power map on the same chip to a structurally unseen package.

![Representative AI4thermal work, physical factors, evaluation settings, and public artifact status](assets/prior-work-landscape.png)

`G/P/M/B/t` denote geometry or placement, power, material, boundary or cooling, and time. “Closest Scope” is an approximate capability mapping, not a claim that the datasets or splits are equivalent. Citation indices follow the manuscript bibliography.

## What the benchmark reveals

![Selected IC-ThermBench results: gradual in-support degradation followed by a structural-OOD gap and few-shot recovery](assets/generalization-gap.svg)

| Track | Physical support | Best method | Best RMSE ↓ |
|---|---|---|---:|
| **S1** | fixed-design source tasks | **Therm-FM L** | **0.009–0.076 K**¹ |
| **S2** | represented layouts and configurations | **SAU-FNO** | **0.703 K** |
| **S3** | S2 + material conductivity | **U-FNO** | **0.802 K** |
| **S4** | S3 + ambient and cooling conditions | **SAU-FNO** | **1.216 K** |
| **S5 · zero-shot** | five case-disjoint chiplet systems | **Therm-FM T** | **15.99 K** |
| **S5 · 10-shot** | ten labeled samples per unseen case | **Therm-FM B** | **3.19 K** |

¹ S1 is a collection of eight source tasks, so its range is task-specific rather than one pooled score. The [selected benchmark results](docs/RESULTS.md) include representative S1 cases, compact S2–S4 comparisons, and the main S5 zero-shot and few-shot findings. S2–S5 use the controlled IC-ThermBench protocol.

The in-support results degrade gradually as observed physical dimensions are added. The case-disjoint S5 shift is qualitatively different: error grows by roughly an order of magnitude, model rankings change, and a small amount of target supervision recovers much of the gap. See the [paper preview](docs/PAPER_PREVIEW.md) for interpretation and the [reproduction guide](docs/REPRODUCE.md) for the exact protocol.

## Benchmark at a glance

IC-ThermBench uses **Scope** rather than “level”: the sequence describes the capability a deployment needs, not a universal maturity rank.

| Scope | Evaluation setting | Newly variable factors | Samples | Typical use |
|---|---|---|---:|---|
| **S1 · Source suite** | fixed-design prediction | power; time for transient tasks | 32,000 | workload analysis, dynamic thermal management |
| **S2 · Layout** | in-support | placement and orientation across system templates | 15,000 | floorplanning and design-space exploration |
| **S3 · Material** | in-support | S2 + local thermal conductivity | 15,000 | material and process sweeps |
| **S4 · Boundary** | in-support | S3 + ambient temperature and convection | 15,000 | cooling and environment co-design |
| **S5 · Structural OOD** | case-disjoint | unseen chiplet counts, sizes, power-density regimes, and utilization | 5,000 | transfer to a new product or package |

### Dataset provenance and credit

- **S1 is collected, not regenerated.** IC-ThermBench preserves the original Alpha EV6 and industrial task definitions, simulators, resolutions, and evaluation conventions used along the [ARO](https://github.com/Mia-WMY/ARO) → [Therm-FM](https://arxiv.org/abs/2605.22663) research line. We do not alter these source tasks or pool their scores with S2–S5. The unified S1 data package and evaluation code are **on the way**; the [recorded benchmark results](docs/RESULTS.md) are available now.
- **S2 follows the ATPlace2.5D lineage.** Cases 1–10, their chiplet systems, and the HotSpot-based thermal setup originate from Qipan Wang *et al.*'s [ATPlace2.5D public package](https://github.com/PKU-IDEA/ATPlace_pub). IC-ThermBench uses this tested foundation for its layout scope.
- **S3–S5 are consistent extensions.** They retain the S2 case/generation conventions while adding material support, boundary support, and held-out structural systems. This preserves continuity with an established placement benchmark while expanding the thermal-learning evaluation space.

The current executable release contains the generator-backed **S2–S5** data and code path. S1 remains explicitly separated until its unified package is ready.

### Eight baselines, one protocol

| Family | Baselines |
|---|---|
| Convolutional networks | U-Net |
| Neural operators | FNO, U-FNO, SAU-FNO, DeepOHeat |
| PDE foundation models | Therm-FM T / B / L |

Every baseline receives the same labeled samples, split membership, physical channels, and metric implementation. The release does not force one optimizer onto every architecture: model-specific training recipes are preserved in [`MODEL_ZOO`](exp/exp_basic.py).

For DeepOHeat, we report fully supervised training rather than its label-free physics-residual scheme so that it is compared fairly with the other supervised baselines. In our tests, the supervised variant was also substantially more accurate and more stable to train.

## Quick start

### 1. Install

```bash
git clone https://github.com/Day333/IC-ThermBench.git
cd IC-ThermBench
conda env create -f environment.yml
conda activate ic-thermbench
python script/smoke_test.py
```

The frozen environment matches the released checkpoints. Therm-FM is version-sensitive; read the [installation notes](docs/INSTALL.md) before changing PyTorch, Transformers, or Accelerate.

### 2. Place data and checkpoints

Download the [datasets (~4.6 GB)](https://drive.google.com/file/d/15Do8Raf070VseV9cn44j1hdVpD3Rz-Un/view?usp=sharing) and [released checkpoints (~9.6 GB)](https://drive.google.com/file/d/1iP6dH3N_s1KzaOugxwe9aK_apO6DkXTD/view?usp=sharing), then unpack them at the repository root:

```text
IC-ThermBench/
├── datasets/
│   ├── level2_steady/
│   ├── level3_steady/
│   ├── level4_steady/
│   └── level5_steady/
└── checkpoints/
```

The public command tokens remain `level2`–`level5` for checkpoint compatibility; they correspond to paper Scopes S2–S5.

### 3. Evaluate

```bash
# One baseline on one scope
python run.py --model UFNO --data level2 --task test

# Frozen S4 checkpoint on structural-OOD S5
python run.py --model ThermFM-T --data level5 --task test

# All 8 baselines × S2–S5, followed by one summary table
bash script/test_all.sh
```

When `--load` is omitted, S5 zero-shot evaluation automatically resolves the matching S4 checkpoint. No S5 labels or normalization updates are used.

## Train, adapt, compare

```bash
# Train one model on S2/S3/S4
bash script/UFNO/train.sh

# Ten labeled samples per unseen S5 case
python run.py --model UFNO --data level5 --task finetune --shots 10

# Run all released adaptation recipes
bash script/finetune_all.sh

# Rebuild the benchmark summary from saved metrics
python utils/summarize.py level2 level3 level4 level5
```

`run.py` is the single entry point; every command also accepts `--root_path`, `--checkpoints`, `--load`, and `--output`. Full commands, split semantics, metric definitions, and reproducibility boundaries are in [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Use IC-ThermBench for your model

A new predictor only needs to satisfy one tensor contract:

```text
input   (B, X, Y, Z, P)
output  (B, X, Y, Z)       where X = Y = 64 and Z = 1
```

Register the constructor and training recipe once, then the existing scripts provide matched S2–S5 training, zero-shot evaluation, few-shot adaptation, and the shared metrics. The complete three-step example is in [docs/ADD_A_MODEL.md](docs/ADD_A_MODEL.md).

## Reproducibility contract

- **Data:** published tensors, physical-channel schema, and S5 case manifest.
- **Splits:** deterministic, index-based train/validation/test membership.
- **Labels:** supervised reference-temperature fields for every baseline.
- **Metrics:** one implementation for RMSE, MAE, R², MaxAE, peak-temperature error, and Top-50 MAE.
- **Recipes:** architecture-specific optimization is recorded, not retroactively homogenized.
- **Artifacts:** released checkpoints and JSON outputs can be audited without retraining.

The release is designed to be easy to reproduce **and** easy to extend: data conversion, model execution, evaluation, aggregation, and OOD adaptation all use the same repository-level interface.

## Documentation

| Guide | Contents |
|---|---|
| [Installation](docs/INSTALL.md) | exact environment, GPU notes, Therm-FM dependencies |
| [Datasets](docs/DATASETS.md) | files, shapes, channels, splits, manifests, checkpoints |
| [Benchmark results](docs/RESULTS.md) | S1 source tasks, S2–S4 in-support results, S5 zero-shot and adaptation |
| [Reproduce](docs/REPRODUCE.md) | evaluation tracks, commands, metrics, expected key results |
| [Add a model](docs/ADD_A_MODEL.md) | tensor contract, registry entry, scripts, integration checklist |
| [Paper preview](docs/PAPER_PREVIEW.md) | benchmark motivation, design, findings, and paper status |

## Current limitations

- S1 data packaging and unified evaluation scripts are still in progress; its stored values currently preserve source-reported protocols.
- S2–S5 currently cover 64×64 steady-state fields with `Z=1`; progressive transient evaluation beyond S1 remains future work.
- S2–S4 are independently generated, not sample-wise paired, so their differences measure distributional difficulty rather than a strict causal ablation.
- S5 contains five held-out systems. It reveals a substantial structural-OOD gap, but cannot represent every industrial package, process, or cooling technology.
- The benchmark is label-supervised. Physics-only and label-free training answer a different question and are not presented as directly interchangeable baselines.

## Citation

The benchmark artifacts are citable now; the paper citation will replace this software entry after public release.

```bibtex
@software{icthermbench2026,
  title     = {IC-ThermBench: An Open, Progressive Benchmark for Generalizable
               2.5D/3D-IC Thermal Learning},
  author    = {The IC-ThermBench Authors},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21992816},
  url       = {https://doi.org/10.5281/zenodo.21992816},
  note      = {Paper in preparation}
}
```

## License and acknowledgements

- Repository code is released under the [MIT License](LICENSE).
- Original IC-ThermBench S2–S5 data, fixed splits, and released result records are available under [CC BY 4.0](LICENSE-DATA): reuse and extension are welcome with attribution.
- S1 source data and third-party components retain their original licenses. The IC-ThermBench manuscript and paper-derived prose, figures, and tables are not relicensed as dataset content.

See the [license map and attribution guidance](LICENSES.md) for the exact boundaries. Open licensing permits legitimate reuse; it does not permit claiming the benchmark, paper text, or curation work as someone else's contribution.

We thank the authors of [ARO](https://github.com/Mia-WMY/ARO), [Therm-FM](https://arxiv.org/abs/2605.22663), and [ATPlace2.5D](https://github.com/PKU-IDEA/ATPlace_pub), as well as the HotSpot, Poseidon/scOT, FNO, U-FNO, and DeepOHeat communities. Their public artifacts and source tasks make a shared thermal-learning benchmark possible.
