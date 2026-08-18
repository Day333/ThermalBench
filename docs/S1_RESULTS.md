# S1 source-suite results and provenance

> **Release status:** results recorded; unified S1 data packaging and evaluation code are **on the way**.

S1 is the fixed-design anchor of ThermalBench. It consolidates established Alpha EV6 and industrial thermal-learning tasks rather than creating or regenerating them. ThermalBench preserves the source task definitions, simulator/fidelity choices, resolutions, and evaluation conventions.

The main lineage is:

- [ARO — Autoregressive Operator Learning for Transferable and Multi-Fidelity 3D-IC Thermal Analysis with Active Learning](https://github.com/Mia-WMY/ARO), ICCAD 2024. Its public repository contains the steady/transient implementations and a source-data link for three 3D-IC configurations.
- [SAU-FNO — Self-Attention to Operator Learning-Based 3D-IC Thermal Simulation](https://doi.org/10.1109/DAC63849.2025.11132988), DAC 2025.
- [Therm-FM — Foundation Model Is All You Need for 3D-ICs Thermal Simulation](https://arxiv.org/abs/2605.22663), arXiv 2026 / extended DAC 2026 work. It expands the source suite with industrial cases and reports the strongest S1 results.

## Recorded results

The table stores the finest-resolution source-suite comparison used in the ThermalBench manuscript. Errors are in kelvin and are **task-specific**: they should not be averaged into, or directly compared as, one controlled S2–S5 score.

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

Therm-FM L is the strongest recorded method across all eight tasks. Its RMSE ranges from 0.009 to 0.076 K, showing that fixed-design power/time prediction can already reach very low error. This makes S1 an anchor rather than the endpoint: S2–S5 ask what happens as physical support expands and then becomes structurally disjoint.

## What “collected, not modified” means

- ThermalBench does not relabel, resimulate, resize, or merge S1 tasks into a new artificial distribution.
- Solver fidelity and grid resolution remain attached to each source task.
- S1 values are not pooled with the controlled S2–S5 protocol.
- This file records provenance and results now; the upcoming S1 package will add canonical manifests, conversion, and one-command evaluation without changing the physical tasks.

Until that package is released, use the linked ARO artifacts and Therm-FM paper as the authoritative source-task references.
