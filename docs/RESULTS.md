# Selected IC-ThermBench results

This page presents a compact subset of the S1–S5 results. It is intended to show the benchmark's main behavior, not reproduce every table from the paper. S1 retains its source-task protocols; S2–S5 use the shared IC-ThermBench splits, preprocessing, labels, and metrics.

> **Release status:** S2–S5 data, checkpoints, and evaluation code are released. S1 results are recorded, while its unified data package and one-command evaluator are **on the way**.

## At a glance

| Track | Evaluation support | Best method | Best RMSE ↓ |
|---|---|---|---:|
| S1 | fixed-design source tasks | Therm-FM L | 0.009–0.076 K¹ |
| S2 | represented layouts/configurations | SAU-FNO | 0.703 K |
| S3 | S2 + material conductivity | U-FNO | 0.802 K |
| S4 | S3 + ambient/cooling conditions | SAU-FNO | 1.216 K |
| S5 zero-shot | five case-disjoint systems | Therm-FM T | 15.99 K |
| S5 10-shot | ten labels per unseen case | Therm-FM B | 3.19 K |

¹ S1 contains task-specific protocols and therefore has no single pooled score.

The matched-support results worsen gradually from S2 to S4. S5 is qualitatively different: changing the underlying chiplet system increases the best RMSE by roughly 12× relative to S4. This is the benchmark's main distinction between learning broader observed physics and extrapolating to unseen structure.

## S1: source-suite snapshot

S1 collects established Alpha EV6 and industrial tasks without relabeling, resizing, or merging them into a new distribution. The lineage follows [ARO](https://github.com/Mia-WMY/ARO), [SAU-FNO](https://doi.org/10.1109/DAC63849.2025.11132988), and [Therm-FM](https://arxiv.org/abs/2605.22663). The entries below are representative Therm-FM L results at the finest reported resolution; errors are in kelvin.

| Task | Case | Grid | RMSE | MAE |
|---|---|---:|---:|---:|
| Steady | HS-SC | 88×88 | 0.021 | 0.012 |
| Steady | HS-QC | 64×64 | 0.076 | 0.023 |
| Transient | HS-SC | 88×88×9 | 0.009 | 0.004 |
| Transient | HS-OC | 151×151×9 | 0.060 | 0.030 |
| Industrial steady | IND-8C | 101×101 | 0.011 | 0.008 |
| Industrial steady | IND-32C | 101×101 | 0.010 | 0.008 |

These low errors show that fixed-design power/time prediction is already a strong and relatively mature setting. S2–S5 therefore focus on what happens when the physical support expands beyond a fixed task definition. S1 data remain subject to their upstream licenses and citation requirements.

## S2–S4: progressive in-support generalization

Only the best and runner-up results are shown here. S2–S4 are independently generated, rather than sample-wise paired perturbations, so the trend measures increasing distributional difficulty rather than a strict one-variable ablation.

| Track | Best method | RMSE | Runner-up | RMSE | Best peak ΔT |
|---|---|---:|---|---:|---:|
| S2 · Layout | SAU-FNO | **0.7028** | U-FNO | 0.7047 | **0.4167** |
| S3 · + Material | U-FNO | **0.8016** | SAU-FNO | 0.8732 | **0.3568** |
| S4 · + Boundary | SAU-FNO | **1.2158** | U-FNO | 1.3265 | **0.6700** |

Adding observed material and boundary variation causes moderate degradation, not collapse. The ranking nevertheless changes: SAU-FNO leads S2 and S4, while U-FNO is strongest on S3. Performance on a simpler scope is therefore not a reliable proxy for performance after new physical dimensions are introduced.

## S5: structural OOD and adaptation

Frozen S4 checkpoints are evaluated on unseen Cases 16–20 without target labels or updated normalization. Few-shot results use `K` labeled samples per OOD case and a separate fixed holdout.

| Setting | Representative best method | RMSE ↓ | Interpretation |
|---|---|---:|---|
| S4 matched-support reference | SAU-FNO | 1.2158 | observed layout/material/boundary support |
| S5 zero-shot | Therm-FM T | 15.9878 | case-disjoint structural extrapolation |
| S5, K=10 | Therm-FM B | 3.19 | 50 target labels in total |
| S5, K=100 | U-FNO | 1.53 | most of the gap has been recovered |
| S5, K=500 | SAU-FNO | 1.00 | higher-budget target calibration |

Ten labels per case reduce RMSE by 63–86% across the evaluated models, making few-shot calibration a practical recovery path. It is still reported separately: a model that adapts well after seeing target labels has not solved zero-shot structural generalization.

### Different OOD cases favor different models

| OOD shift | Best method | Best per-case RMSE |
|---|---|---:|
| C16 · chiplet count | DeepOHeat | 4.84 |
| C17 · power density | U-FNO | 6.76 |
| C18 · size mixture | Therm-FM T | 8.48 |
| C19 · power concentration | U-Net | 10.05 |
| C20 · utilization | Therm-FM T | 7.46 |

No model wins every structural shift. Therm-FM T has the best aggregate zero-shot result because it is comparatively balanced, while other methods show narrow strengths and severe case-specific failures. Reporting the OOD axis is therefore more informative than publishing only one pooled score.

## Takeaways

- Broader **observed** physical support is learnable with moderate accuracy loss.
- Strong in-support performance and model scale do not determine the structural-OOD ranking.
- Few-shot adaptation is effective, but should complement—not replace—an explicit zero-shot report.

Commands, metric definitions, and reproduction boundaries are documented in [REPRODUCE.md](REPRODUCE.md). The selected S2–S5 result records are covered by the repository's [data license](../LICENSE-DATA).
