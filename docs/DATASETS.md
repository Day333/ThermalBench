# Datasets and checkpoint layout

ThermalBench separates the paper's capability vocabulary from the stable command-line API:

| Paper scope | CLI token | Role in this release |
|---|---|---|
| S1 | — | collected source suite; results available, unified package on the way |
| S2 | `level2` | in-support layout and configuration variation |
| S3 | `level3` | S2 + material conductivity variation |
| S4 | `level4` | S3 + ambient and convective-boundary variation |
| S5 | `level5` | case-disjoint structural-OOD evaluation and adaptation |

The `level*` tokens are retained for backward compatibility with released archives and checkpoints.

## Download

- [S1–S5 benchmark results and provenance](RESULTS.md) — S1 data package and unified evaluation code on the way
- [S2–S5 datasets (~4.6 GB)](https://drive.google.com/file/d/15Do8Raf070VseV9cn44j1hdVpD3Rz-Un/view?usp=sharing)
- [24 checkpoints: 8 baselines × S2/S3/S4 (~9.6 GB)](https://drive.google.com/file/d/1wisvvO19Fx9Znki651j-QWuJHVz2aPHQ/view?usp=sharing)

Unpack both archives at the repository root. No preprocessing is required for the released `.mat` files.

## Provenance

S1 is an unchanged collection of established source tasks, not a regenerated ThermalBench distribution. It follows the [ARO](https://github.com/Mia-WMY/ARO) and [Therm-FM](https://arxiv.org/abs/2605.22663) research line and retains its source grids, simulators, and task-level evaluation. The unified S1 package is intentionally marked pending rather than being mixed into the current download.

S2 uses Cases 1–10 and the HotSpot thermal lineage from Qipan Wang *et al.*'s [ATPlace2.5D public package](https://github.com/PKU-IDEA/ATPlace_pub). S3–S5 extend the same case conventions with material, boundary, and structurally held-out support.

## Data license

The original ThermalBench S2–S5 tensors, fixed splits/manifests, and released result records are licensed under [CC BY 4.0](../LICENSE-DATA). Reuse, redistribution, and extensions are welcome with attribution and an indication of changes. S1 and all third-party source material retain their upstream licenses; see the complete [license map](../LICENSES.md).

## Dataset files

```text
datasets/
├── level2_steady/
│   ├── input.mat
│   └── output.mat
├── level3_steady/
│   ├── input.mat
│   └── output.mat
├── level4_steady/
│   ├── input.mat
│   └── output.mat
└── level5_steady/
    ├── input.mat
    ├── output.mat
    └── manifest.json
```

| Dataset | Samples | Input channels in C-axis order | Systems |
|---|---:|---|---|
| `level2` / S2 | 15,000 | `chiplet_power`, `grid_x`, `grid_y` | Cases 1–10 |
| `level3` / S3 | 15,000 | S2 + `local_thermal_k` | Cases 1–10 |
| `level4` / S4 | 15,000 | S3 + `ambient_K`, `h_w_m2k`, `r_convec_k_per_w` | Cases 1–10 |
| `level5` / S5 | 5,000 | same schema as S4 | unseen Cases 16–20 |

S2–S4 contain 1,500 samples per represented case. S5 contains 1,000 samples per unseen case and is excluded from standard training.

## Tensor contract

The HDF5-backed MATLAB files use:

```text
input.mat   ["data"]  (B, P, Z, Y, X)  float32
output.mat  ["data"]  (B,    Z, Y, X)  float32, temperature in K
```

The loader transposes them to the repository-wide model interface:

```text
input   (B, X, Y, Z, P)
target  (B, X, Y, Z)
```

For this release, `X = Y = 64` and `Z = 1`. A new model must not hard-code `P`: it is 3, 4, or 7 depending on the scope.

## Deterministic S2–S4 split

The split is index-based and never shuffled:

```text
test  = last 20% of all samples
train = first 90% of the leading 80%
val   = last 10% of the leading 80%
```

The archives store S2–S4 samples in case-interleaved order so every segment remains case-balanced:

| Segment | Samples per case | Total |
|---|---:|---:|
| Train | 1,080 | 10,800 |
| Validation | 120 | 1,200 |
| Test | 300 | 3,000 |

Do not restack the released files by case before using the default loader. Doing so would change the semantic split while leaving the sample count unchanged.

## S5 manifest and adaptation split

S5 is a pure evaluation dataset. Zero-shot testing scores all 5,000 samples using a frozen S4 checkpoint and unchanged preprocessing statistics.

`manifest.json` records the case identity and source file for every sample. Few-shot adaptation groups by case and uses:

```text
first 500 samples per case  → adaptation pool; K-shot selects the first K
last  500 samples per case  → common held-out evaluation set for every K
```

Thus `--shots 10` uses 10 samples from each of five unseen systems—50 labels total—with zero overlap with the 2,500-sample holdout.

## Checkpoints

```text
checkpoints/
├── level2_UFNO/model.pt
├── level3_UFNO/model.pt
├── level4_UFNO/model.pt
├── level2_ThermFM-T/
│   ├── config.json
│   └── pytorch_model.bin
└── fewshot/
    └── level5_UFNO_k10.json
```

Operator-family checkpoints contain the model state and normalization state. Therm-FM checkpoints use the scOT/Hugging Face directory layout. `run.py` resolves both formats automatically.

Use non-default locations without moving files:

```bash
python run.py \
  --model UFNO \
  --data level2 \
  --task test \
  --root_path /path/to/datasets \
  --checkpoints /path/to/checkpoints
```
