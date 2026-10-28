# Selected IC-ThermBench results

This page presents a compact subset of the S1–S5 results. It is intended to show the benchmark's main behavior, not reproduce every table from the paper. S1 retains its source-task protocols; S2–S5 use the shared IC-ThermBench splits, preprocessing, labels, and metrics.

> **Release status:** S2–S5 data, evaluation code, and [checkpoints](https://drive.google.com/file/d/1dXtFyhiSo68NFZ-HwW6a3gmjdBfaJ_1-/view?usp=sharing) are released, and the S1 datasets are now released as well. S1 results are recorded, while its one-command evaluator is **on the way**.

## At a glance

| Track | Evaluation support | Best method | Result ↓ | Second-best method | Result ↓ |
|---|---|---|---:|---|---:|
| S1 | fixed-design source tasks | **Therm-FM** | **0.004–0.049 K MAE**¹ | U-FNO / SAU-FNO (task-dependent) | <u>0.011–0.144 / 0.013–0.162 K MAE</u>² |
| S2 | represented layouts/configurations | **Therm-FM** | **0.354 K RMSE** | U-FNO | <u>0.444 K RMSE</u> |
| S3 | S2 + material conductivity | **U-FNO** | **0.670 K RMSE** | SAU-FNO | <u>0.708 K RMSE</u> |
| S4 | S3 + ambient/cooling conditions | **Therm-FM** | **0.811 K RMSE** | U-FNO | <u>0.939 K RMSE</u> |
| S5 zero-shot | five case-disjoint systems | **Therm-FM** | **8.67 K RMSE** | U-Net | <u>18.85 K RMSE</u> |
| S5 10-shot | ten labels per unseen case | **Therm-FM** | **2.09 K RMSE** | U-Net | <u>2.26 K RMSE</u> |

¹ S1 contains eleven task-specific source protocols and therefore has no single pooled score; the paper reports MAE ranges by task group. ² The S1 runner-up varies by task between U-FNO and SAU-FNO; their MAE ranges are listed in that order. Therm-FM T/B/L count as one method family, represented by their best score on each track. Best results are bold; second-best results are underlined.

The best matched-support RMSE worsens gradually from 0.354 K on S2 to 0.811 K on S4. S5 is qualitatively different: changing the underlying chiplet system increases the best RMSE by about 10.7× relative to S4. This is the benchmark's main distinction between learning broader observed physics and extrapolating to unseen structure.

## S1: source-suite snapshot

S1 collects eleven established Alpha EV6 and industrial tasks without relabeling, resizing, or merging them into a new distribution. The lineage follows [ARO](https://github.com/Mia-WMY/ARO), [SAU-FNO](https://doi.org/10.1109/DAC63849.2025.11132988), and [Therm-FM](https://arxiv.org/abs/2605.22663). The compact paper table reports MAE ranges at the finest reported resolution; errors are in kelvin and are not pooled across systems or resolutions.

| Method | HS steady MAE | HS transient MAE | Industrial MAE |
|---|---:|---:|---:|
| FNO | 0.063–0.164 | 0.034–0.163 | 0.017–0.018 |
| U-FNO | 0.046–0.126 | 0.011–0.144 | 0.013 |
| SAU-FNO | 0.041–0.162 | 0.013–0.149 | 0.020–0.026 |
| DeepOHeat | 0.745–2.585 | 0.315–1.820 | 0.039–0.044 |
| **Therm-FM** | **0.012–0.049** | **0.004–0.030** | **0.008** |

These low errors show that fixed-design power/time prediction is already a strong and relatively mature setting. S2–S5 therefore focus on what happens when the physical support expands beyond a fixed task definition. S1 data remain subject to their upstream licenses and citation requirements.

## S2–S4: progressive in-support generalization

Only the best and runner-up method families are shown here; Therm-FM T/B/L count as one family. S2–S4 are independently generated, rather than sample-wise paired perturbations, so the trend measures increasing distributional difficulty rather than a strict one-variable ablation.

| Track | Best method | RMSE | Runner-up | RMSE | Best peak ΔT |
|---|---|---:|---|---:|---:|
| S2 · Layout | Therm-FM | **0.3536** | U-FNO | <u>0.4442</u> | **0.2164** (Therm-FM) |
| S3 · + Material | U-FNO | **0.6700** | SAU-FNO | <u>0.7078</u> | **0.3312** (U-FNO) |
| S4 · + Boundary | Therm-FM | **0.8113** | U-FNO | <u>0.9391</u> | **0.3696** (Therm-FM) |

Adding observed material and boundary variation causes moderate degradation, not collapse. Therm-FM leads S2 and S4, with L supplying the best scores on both, whereas U-FNO leads all six metrics on S3, ahead of SAU-FNO and every Therm-FM variant. U-FNO is also the runner-up method family on S2 and S4. Performance on a simpler scope is therefore not a reliable proxy for performance after new physical dimensions are introduced.

## S5: structural OOD and adaptation

Frozen S4 checkpoints are evaluated on unseen Cases 11–15 without target labels or updated normalization. Few-shot results use `K` labeled samples per OOD case and a separate fixed holdout.

| Setting | Best method | RMSE ↓ | Second-best method | RMSE ↓ | Interpretation |
|---|---|---:|---|---:|---|
| S4 matched-support reference | Therm-FM | **0.8113** | U-FNO | <u>0.9391</u> | observed layout/material/boundary support |
| S5 zero-shot | Therm-FM | **8.6729** | U-Net | <u>18.8542</u> | case-disjoint structural extrapolation |
| S5, K=10 | Therm-FM | **2.085** | U-Net | <u>2.265</u> | 50 target labels in total |
| S5, K=100 | Therm-FM | **1.060** | U-FNO | <u>1.062</u> | most of the gap has been recovered |
| S5, K=500 | U-FNO | **0.762** | Therm-FM | <u>0.806</u> | higher-budget target calibration |

Ten labels per case reduce RMSE by approximately 74–92% across the evaluated models, making few-shot calibration a practical recovery path. It is still reported separately: a model that adapts well after seeing target labels has not solved zero-shot structural generalization.

### Different OOD cases favor different models

| OOD shift | Best method | RMSE ↓ | Second-best method | RMSE ↓ |
|---|---|---:|---|---:|
| C11 · chiplet count | Therm-FM | **4.48** | DeepOHeat | <u>4.79</u> |
| C12 · power density | U-Net | **3.25** | SAU-FNO | <u>4.71</u> |
| C13 · size heterogeneity | Therm-FM | **9.03** | DeepOHeat | <u>10.01</u> |
| C14 · power concentration | Therm-FM | **7.50** | SAU-FNO | <u>9.88</u> |
| C15 · package utilization | Therm-FM | **6.33** | U-Net | <u>17.05</u> |

No model wins every structural shift. Therm-FM has the best aggregate zero-shot result because its strongest variant is comparatively balanced, while other methods show narrow strengths and severe case-specific failures. Reporting the OOD axis is therefore more informative than publishing only one pooled score.

## Takeaways

- Broader **observed** physical support is learnable with moderate accuracy loss.
- Strong in-support performance and model scale do not determine the structural-OOD ranking.
- Few-shot adaptation is effective, but should complement—not replace—an explicit zero-shot report.

Commands, metric definitions, and reproduction boundaries are documented in [REPRODUCE.md](REPRODUCE.md). The selected S2–S5 result records are covered by the repository's [data license](../LICENSE-DATA).
