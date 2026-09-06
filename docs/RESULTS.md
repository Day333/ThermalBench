# Selected IC-ThermBench results

This page presents a compact subset of the S1–S5 results. It is intended to show the benchmark's main behavior, not reproduce every table from the paper. S1 retains its source-task protocols; S2–S5 use the shared IC-ThermBench splits, preprocessing, labels, and metrics.

> **Release status:** S2–S5 data, checkpoints, and evaluation code are released, and the S1 datasets are now released as well. S1 results are recorded, while its one-command evaluator is **on the way**.

## At a glance

| Track | Evaluation support | Best method | Result ↓ | Second-best method | Result ↓ |
|---|---|---|---:|---|---:|
| S1 | fixed-design source tasks | **Therm-FM L** | **0.004–0.049 K MAE**¹ | — | —² |
| S2 | represented layouts/configurations | **Therm-FM L** | **0.443 K RMSE** | Therm-FM B | <u>0.587 K RMSE</u> |
| S3 | S2 + material conductivity | **Therm-FM B** | **0.716 K RMSE** | Therm-FM L | <u>0.796 K RMSE</u> |
| S4 | S3 + ambient/cooling conditions | **Therm-FM B** | **0.933 K RMSE** | Therm-FM L | <u>0.959 K RMSE</u> |
| S5 zero-shot | five case-disjoint systems | **Therm-FM T** | **15.51 K RMSE** | Therm-FM B | <u>17.23 K RMSE</u> |
| S5 10-shot | ten labels per unseen case | **Therm-FM L** | **2.73 K RMSE** | Therm-FM B | <u>2.76 K RMSE</u> |

¹ S1 contains eleven task-specific source protocols and therefore has no single pooled score; the paper reports MAE ranges by task group. ² Its second-best method varies by task and resolution, so no single runner-up is reported. Best results are bold; second-best results are underlined.

The best matched-support RMSE worsens gradually from 0.443 K on S2 to 0.933 K on S4. S5 is qualitatively different: changing the underlying chiplet system increases the best RMSE by about 16.6× relative to S4. This is the benchmark's main distinction between learning broader observed physics and extrapolating to unseen structure.

## S1: source-suite snapshot

S1 collects eleven established Alpha EV6 and industrial tasks without relabeling, resizing, or merging them into a new distribution. The lineage follows [ARO](https://github.com/Mia-WMY/ARO), [SAU-FNO](https://doi.org/10.1109/DAC63849.2025.11132988), and [Therm-FM](https://arxiv.org/abs/2605.22663). The compact paper table reports MAE ranges at the finest reported resolution; errors are in kelvin and are not pooled across systems or resolutions.

| Method | HS steady MAE | HS transient MAE | Industrial MAE |
|---|---:|---:|---:|
| FNO | 0.063–0.164 | 0.034–0.163 | 0.017–0.018 |
| U-FNO | 0.046–0.126 | 0.011–0.144 | 0.013 |
| SAU-FNO | 0.041–0.162 | 0.013–0.149 | 0.020–0.026 |
| DeepOHeat | 0.745–2.585 | 0.315–1.820 | 0.039–0.044 |
| **Therm-FM L (629M)** | **0.012–0.049** | **0.004–0.030** | **0.008** |

These low errors show that fixed-design power/time prediction is already a strong and relatively mature setting. S2–S5 therefore focus on what happens when the physical support expands beyond a fixed task definition. S1 data remain subject to their upstream licenses and citation requirements.

## S2–S4: progressive in-support generalization

Only the best and runner-up results are shown here. S2–S4 are independently generated, rather than sample-wise paired perturbations, so the trend measures increasing distributional difficulty rather than a strict one-variable ablation.

| Track | Best method | RMSE | Runner-up | RMSE | Best peak ΔT |
|---|---|---:|---|---:|---:|
| S2 · Layout | Therm-FM L | **0.4427** | Therm-FM B | <u>0.5874</u> | **0.2610** (Therm-FM L) |
| S3 · + Material | Therm-FM B | **0.7161** | Therm-FM L | <u>0.7957</u> | **0.3111** (Therm-FM B) |
| S4 · + Boundary | Therm-FM B | **0.9334** | Therm-FM L | <u>0.9585</u> | **0.4076** (Therm-FM L) |

Adding observed material and boundary variation causes moderate degradation, not collapse. The ranking nevertheless changes: Therm-FM L leads all six S2 metrics; Therm-FM B leads all six S3 metrics and the S4 global-field errors; Therm-FM L retains the best S4 hotspot-oriented errors. Among operator baselines, SAU-FNO leads S2 and S4 while U-FNO leads S3. Performance on a simpler scope is therefore not a reliable proxy for performance after new physical dimensions are introduced.

### Therm-FM optimization-setting sensitivity

The official S2–S5 tables use configuration B: one GPU, training learning rate `1.5e-4`, embedding/recovery learning rate `1.5e-3`, batch size 40, and validation-best checkpoint selection. Configuration A is the earlier four-GPU run with training learning rate `5e-5` and final-epoch selection. The comparison below is retained because optimization changes both absolute scores and rankings.

| Variant | Config | S2 RMSE | S3 RMSE | S4 RMSE | S5 zero-shot | K=10 | K=500 | GPU·s |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|
| Therm-FM T | A | 1.4679 | 2.2188 | 2.4703 | 15.9878 | 3.21 | 1.27 | 8,824 |
| Therm-FM T | **B** | **1.0269** | **1.1524** | **1.3402** | **15.5102** | **2.86** | **1.00** | **2,232** |
| Therm-FM B | A | 1.1807 | 1.6651 | 2.0580 | 21.4339 | 3.19 | 1.12 | 19,132 |
| Therm-FM B | **B** | **0.5874** | **0.7161** | **0.9334** | **17.2324** | **2.76** | **0.92** | **5,220** |
| Therm-FM L | A | 1.2635 | 1.7001 | 2.0667 | **18.3780** | 3.29 | 1.19 | 43,630 |
| Therm-FM L | **B** | **0.4427** | **0.7957** | **0.9585** | 24.0349 | **2.73** | **0.95** | **9,129** |

Configuration B reduces every Therm-FM variant's in-support RMSE by roughly 30–65% and uses about 4–5× less aggregate GPU time. The change is not a uniform OOD improvement: Therm-FM L's sharper S2–S4 fit under B coincides with worse frozen S5 transfer, reinforcing the need to report the cross-package track separately.

## S5: structural OOD and adaptation

Frozen S4 checkpoints are evaluated on unseen Cases 11–15 without target labels or updated normalization. Few-shot results use `K` labeled samples per OOD case and a separate fixed holdout.

| Setting | Best method | RMSE ↓ | Second-best method | RMSE ↓ | Interpretation |
|---|---|---:|---|---:|---|
| S4 matched-support reference | Therm-FM B | **0.9334** | Therm-FM L | <u>0.9585</u> | observed layout/material/boundary support |
| S5 zero-shot | Therm-FM T | **15.5102** | Therm-FM B | <u>17.2324</u> | case-disjoint structural extrapolation |
| S5, K=10 | Therm-FM L | **2.73** | Therm-FM B | <u>2.76</u> | 50 target labels in total |
| S5, K=100 | Therm-FM B | **1.10** | Therm-FM L | <u>1.16</u> | most of the gap has been recovered |
| S5, K=500 | Therm-FM B | **0.92** | Therm-FM L | <u>0.95</u> | higher-budget target calibration |

Ten labels per case reduce RMSE by approximately 63–89% across the evaluated models, making few-shot calibration a practical recovery path. It is still reported separately: a model that adapts well after seeing target labels has not solved zero-shot structural generalization.

### Different OOD cases favor different models

| OOD shift | Best method | RMSE ↓ | Second-best method | RMSE ↓ |
|---|---|---:|---|---:|
| C11 · chiplet count | DeepOHeat | **4.84** | Therm-FM B | <u>14.74</u> |
| C12 · power density | SAU-FNO | **5.61** | U-FNO | <u>6.76</u> |
| C13 · size heterogeneity | DeepOHeat | **8.86** | FNO | <u>9.65</u> |
| C14 · power concentration | U-Net | **10.05** | SAU-FNO | <u>13.26</u> |
| C15 · package utilization | U-Net | **7.72** | Therm-FM T | <u>8.91</u> |

No model wins every structural shift. Therm-FM T has the best aggregate zero-shot result because it is comparatively balanced, while other methods show narrow strengths and severe case-specific failures. Reporting the OOD axis is therefore more informative than publishing only one pooled score.

## Takeaways

- Broader **observed** physical support is learnable with moderate accuracy loss.
- Strong in-support performance and model scale do not determine the structural-OOD ranking.
- Few-shot adaptation is effective, but should complement—not replace—an explicit zero-shot report.

Commands, metric definitions, and reproduction boundaries are documented in [REPRODUCE.md](REPRODUCE.md). The selected S2–S5 result records are covered by the repository's [data license](../LICENSE-DATA).
