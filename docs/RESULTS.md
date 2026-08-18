# ThermalBench results

This page records the current S1–S5 benchmark results in one place. S1 preserves the source-task protocols used by earlier public thermal-learning work; S2–S5 use the shared ThermalBench splits, labels, preprocessing, and metric implementation.

> **Release status:** S2–S5 data, checkpoints, and evaluation code are released. S1 results are recorded, while its unified data package and one-command evaluator are **on the way**.

## Results at a glance

| Track | Evaluation support | Best method | Best RMSE ↓ |
|---|---|---|---:|
| S1 | fixed-design source tasks | Therm-FM L | 0.009–0.076 K¹ |
| S2 | represented layouts/configurations | SAU-FNO | 0.657 K |
| S3 | S2 + material conductivity | U-FNO | 0.802 K |
| S4 | S3 + ambient/cooling conditions | U-FNO | 1.327 K |
| S5 zero-shot | five case-disjoint systems | Therm-FM T | 15.99 K |
| S5 10-shot | ten labels per unseen case | Therm-FM B | 3.19 K |

¹ S1 contains eight task-specific protocols and therefore has no single pooled score.

The central result is the discontinuity between S4 and S5. Accuracy degrades gradually as observed physical dimensions are introduced, but every model deteriorates sharply when the underlying chiplet system is structurally unseen. Ten target labels per OOD case recover much of that gap, although adaptation is reported separately from zero-shot generalization.

## S1: source-suite results and provenance

S1 consolidates established Alpha EV6 and industrial tasks rather than regenerating them. ThermalBench keeps the original physical designs, simulator/fidelity choices, resolutions, and evaluation conventions.

The source lineage is:

- [ARO — Autoregressive Operator Learning for Transferable and Multi-Fidelity 3D-IC Thermal Analysis with Active Learning](https://github.com/Mia-WMY/ARO), ICCAD 2024.
- [SAU-FNO — Self-Attention to Operator Learning-Based 3D-IC Thermal Simulation](https://doi.org/10.1109/DAC63849.2025.11132988), DAC 2025.
- [Therm-FM — Foundation Model Is All You Need for 3D-ICs Thermal Simulation](https://arxiv.org/abs/2605.22663), arXiv 2026 / DAC 2026.

Errors are in kelvin. They remain task-specific and must not be averaged into one controlled S2–S5 score.

| Task | Case | Grid | SAU-FNO RMSE | SAU-FNO MAE | Therm-FM L RMSE | Therm-FM L MAE |
|---|---|---:|---:|---:|---:|---:|
| Steady | HS-SC | 88×88 | 0.090 | 0.049 | **0.021** | **0.012** |
| Steady | HS-QC | 64×64 | 0.203 | 0.162 | **0.076** | **0.023** |
| Steady | HS-OC | 151×151 | 0.296 | 0.172 | **0.069** | **0.049** |
| Transient | HS-SC | 88×88×9 | 0.077 | 0.038 | **0.009** | **0.004** |
| Transient | HS-QC | 64×64×5 | 0.127 | 0.080 | **0.016** | **0.011** |
| Transient | HS-OC | 151×151×9 | 0.409 | 0.214 | **0.060** | **0.030** |
| Industrial steady | IND-8C | 101×101 | 0.104 | 0.079 | **0.011** | **0.008** |
| Industrial steady | IND-32C | 101×101 | 0.096 | 0.085 | **0.010** | **0.008** |

Therm-FM L is the strongest recorded method on all eight source tasks. S1 data are collected, not relabeled, resimulated, resized, or merged into a new distribution. The linked upstream artifacts remain authoritative until the unified S1 package is released.

## S2–S4: progressive in-support evaluation

S2–S4 share Cases 1–10 and progressively broaden the observed physical support. The datasets are independently generated, rather than sample-wise paired perturbations, so the comparison measures distributional difficulty instead of a strict causal ablation.

All errors are in kelvin. Peak ΔT is the absolute peak-temperature error; Top-50 MAE is measured at the 50 hottest ground-truth pixels.

| Method | S2 RMSE | S2 Peak ΔT | S2 Top-50 | S3 RMSE | S3 Peak ΔT | S3 Top-50 | S4 RMSE | S4 Peak ΔT | S4 Top-50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FNO | 1.1665 | 0.7586 | 0.8772 | 1.5234 | 1.1098 | 1.2387 | 2.1793 | 1.4885 | 2.0319 |
| U-Net | 1.4149 | 0.8498 | 0.7691 | 1.8258 | 0.7564 | 0.8375 | 2.5508 | 1.1850 | 1.4636 |
| U-FNO | 0.7047 | 0.4333 | 0.5526 | **0.8016** | **0.3568** | **0.4059** | **1.3265** | **0.6973** | **0.8609** |
| SAU-FNO | **0.6568** | **0.3871** | **0.4580** | 0.8405 | 0.3965 | 0.4609 | 1.4468 | 0.8265 | 0.9496 |
| DeepOHeat | 3.6285 | 2.8812 | 3.6529 | 4.0042 | 3.6426 | 4.8742 | 4.6908 | 4.6527 | 5.8420 |
| Therm-FM T | 1.4679 | 0.7278 | 1.2140 | 2.2188 | 0.9205 | 1.6959 | 2.4703 | 1.0993 | 2.0273 |
| Therm-FM B | 1.1807 | 0.5957 | 0.9594 | 1.6651 | 0.6380 | 1.0824 | 2.0580 | 0.8882 | 1.5807 |
| Therm-FM L | 1.2635 | 0.5358 | 1.0052 | 1.7001 | 0.6913 | 1.1578 | 2.0667 | 0.8801 | 1.6200 |

SAU-FNO is strongest on S2, while U-FNO leads all three reported metrics on S3 and S4. Model scale is not monotonic: the largest Therm-FM variant does not consistently outperform the smaller variants after new physical channels are introduced.

## S5: zero-shot structural OOD

Frozen S4 checkpoints are evaluated on all 5,000 samples from unseen Cases 16–20. No target labels, weight updates, or normalization updates are allowed.

| Method | RMSE ↓ | MAE ↓ | R² ↑ | MaxAE ↓ | Peak ΔT ↓ | Top-50 MAE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| FNO | 23.2161 | 22.6481 | 0.2024 | 33.8672 | 29.2796 | 26.3916 |
| U-Net | 19.0992 | 16.8686 | 0.3865 | 41.0030 | 27.9007 | 19.7449 |
| U-FNO | 25.7595 | 24.5947 | 0.0619 | 38.5007 | 30.5241 | 25.6409 |
| SAU-FNO | 34.8821 | 33.2142 | -0.9908 | 51.5330 | 45.8934 | 40.6425 |
| DeepOHeat | 22.5186 | 21.7859 | 0.2205 | 31.2207 | 23.7507 | 23.4288 |
| Therm-FM T | **15.9878** | **14.9969** | **0.5716** | **29.3286** | **20.0278** | **16.4372** |
| Therm-FM B | 21.4339 | 20.6123 | 0.4041 | 35.8006 | 27.9508 | 22.9437 |
| Therm-FM L | 18.3780 | 17.5994 | 0.5618 | 30.3699 | 22.5160 | 18.1769 |

The best RMSE jumps from 1.327 K on S4 to 15.99 K on S5—approximately 12×. Strong in-support rankings also fail to transfer: U-FNO and SAU-FNO lead S2–S4 but degrade substantially under case-disjoint structure.

### Per-case RMSE

The aggregate result hides distinct failure modes. Column labels identify the principal structural shift; each value is RMSE in kelvin.

| Method | C16: count | C17: power density | C18: size mix | C19: power concentration | C20: utilization |
|---|---:|---:|---:|---:|---:|
| U-FNO | 46.99 | **6.76** | 15.89 | 26.77 | 32.39 |
| SAU-FNO | 85.25 | 11.06 | 14.25 | 22.41 | 41.44 |
| FNO | 50.38 | 10.91 | 9.65 | 18.96 | 26.18 |
| U-Net | 29.33 | 13.46 | 34.93 | **10.05** | 7.72 |
| DeepOHeat | **4.84** | 24.02 | 8.86 | 41.80 | 33.07 |
| Therm-FM T | 15.46 | 15.02 | **8.48** | 33.51 | **7.46** |
| Therm-FM B | 26.49 | 12.10 | 15.03 | 21.95 | 31.60 |
| Therm-FM L | 23.59 | 12.06 | 11.60 | 16.23 | 28.41 |

No model wins every structural shift. Therm-FM T achieves the best aggregate result because its errors are comparatively balanced, whereas specialized strengths such as DeepOHeat on high chiplet count do not transfer to the other cases.

## S5: few-shot target adaptation

Fine-tuning uses a fixed adaptation pool and a fixed 2,500-sample holdout. `K` is the number of labeled samples **per OOD case**, so `K=10` uses 50 labels in total. The `K=0` values below use that holdout and can differ slightly from the full 5,000-sample zero-shot table above.

| Method | K=0 | K=10 | K=50 | K=100 | K=250 | K=500 |
|---|---:|---:|---:|---:|---:|---:|
| U-FNO | 25.83 | 3.59 | 1.87 | 1.53 | 1.31 | 1.20 |
| SAU-FNO | 34.92 | 4.07 | 1.92 | 1.53 | 1.30 | 1.19 |
| FNO | 23.18 | 4.25 | 2.20 | 1.77 | 1.45 | 1.35 |
| U-Net | 19.33 | 7.20 | 3.64 | 2.47 | 1.77 | 1.56 |
| DeepOHeat | 22.53 | 5.68 | 3.67 | 3.17 | 2.96 | 2.70 |
| Therm-FM T | **16.01** | 3.21 | 2.04 | 1.70 | 1.43 | 1.27 |
| Therm-FM B | 21.51 | **3.19** | 2.04 | 1.73 | 1.40 | **1.12** |
| Therm-FM L | 18.43 | 3.29 | **1.96** | **1.63** | 1.40 | 1.19 |

### All metrics at K=10

| Method | RMSE ↓ | MAE ↓ | R² ↑ | MaxAE ↓ | Peak ΔT ↓ | Top-50 MAE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| U-FNO | 3.59 | 2.99 | 0.9787 | 11.11 | 4.58 | 4.88 |
| SAU-FNO | 4.07 | 3.50 | 0.9724 | 10.80 | 4.52 | 5.56 |
| FNO | 4.25 | 3.72 | 0.9736 | 11.01 | 4.22 | 5.05 |
| U-Net | 7.20 | 5.59 | 0.9071 | 21.43 | 11.62 | 8.19 |
| DeepOHeat | 5.68 | 4.97 | 0.9361 | 11.87 | 5.91 | 6.28 |
| Therm-FM T | 3.21 | **2.60** | **0.9882** | **10.05** | **2.97** | **3.35** |
| Therm-FM B | **3.19** | 2.62 | 0.9877 | 10.33 | 3.03 | 3.37 |
| Therm-FM L | 3.29 | 2.69 | 0.9859 | 10.41 | 3.30 | 3.66 |

Ten labels per case reduce RMSE by approximately 63–88% across all eight models. Therm-FM B is marginally best in K=10 RMSE, while Therm-FM T is strongest on the other five reported metrics. This ranking reversal is why ThermalBench reports frozen zero-shot results and target-domain adaptation separately.

## Reproducing these tables

Use the released checkpoints and shared evaluation code described in [REPRODUCE.md](REPRODUCE.md). All models call the same metric implementation in [`utils/metrics.py`](../utils/metrics.py). Small last-digit differences can arise from GPU kernels and execution environments; a valid comparison must preserve the split, normalizer, target-label policy, and metric definitions.

The S2–S5 result records are covered by the repository's [data license](../LICENSE-DATA). S1 source data and values remain subject to their upstream licenses and citation requirements.
