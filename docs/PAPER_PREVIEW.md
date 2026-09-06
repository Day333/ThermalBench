# IC-ThermBench paper preview

> **Status:** the paper is available on arXiv: [arXiv:2608.23977](https://arxiv.org/abs/2608.23977). This page summarizes the benchmark's central design and selected results; the paper itself is the reference for the full protocol, tables, and discussion.

## Motivation

Learning-based thermal solvers are increasingly evaluated under power, geometry, material, and cooling variation. Yet comparisons remain difficult because method papers often use different private or partially released datasets, simulators, splits, preprocessing, and metrics. A result can therefore improve because the task changed—not because the predictor is better.

IC-ThermBench addresses this evaluation gap rather than proposing another thermal model. Its novelty is the combination of:

- an open benchmark and common evaluation contract;
- fixed splits, channel semantics, labels, and metrics;
- convolutional, neural-operator, and PDE-foundation-model baselines;
- progressive physical support matched to deployment needs; and
- case-disjoint structural out-of-distribution evaluation.

The release also bundles the practical path around the benchmark: data formatting, model execution, training, inference, adaptation, aggregation, and result export share one interface.

![Five IC-ThermBench generalization scopes](../assets/ic-thermbench-overview.svg)

## Why “Scope”

The five Scopes are not a leaderboard rank that every application must maximize. They specify the physical support a deployment needs:

| Scope | Capability | Representative scenario |
|---|---|---|
| S1 | new power/time inputs on a fixed physical design | runtime workload analysis and dynamic thermal management |
| S2 | new layouts from represented system templates | floorplanning, placement, design-space exploration |
| S3 | S2 + represented material variation | effective-material and process-conductivity sweeps |
| S4 | S3 + represented boundary variation | cooling and deployment-environment co-design |
| S5 | structurally unseen chiplet systems | transfer to a new package or product family |

This progression lets a practitioner choose the smallest sufficient contract instead of treating all forms of “generalization” as interchangeable.

## Dataset design

S1 consolidates eleven established fixed-design thermal-learning tasks spanning steady-state and transient Alpha EV6 configurations and two industrial packages. It contains 32,000 samples and preserves the source task definitions and fidelity settings from the [ARO](https://github.com/Mia-WMY/ARO) and [Therm-FM](https://arxiv.org/abs/2605.22663) lineage; its [recorded results](RESULTS.md) remain task-specific.

S2–S4 use the same ten system families and generation lineage. S2 originates from Qipan Wang *et al.*'s [ATPlace2.5D public cases and thermal setup](https://github.com/PKU-IDEA/ATPlace_pub); S3–S4 extend that foundation consistently. They independently sample layouts while cumulatively exposing material and boundary variables. Each scope contains 15,000 samples. These tracks test generalization within represented structural support.

S5 preserves the S4 physical schema but replaces the represented systems with five case-disjoint systems. Its 5,000 samples stress unseen chiplet count, size heterogeneity, power density, power concentration, and utilization. This distinguishes learning many observed physical variables from transferring to an unseen system structure.

## Selected findings

![IC-ThermBench selected generalization results](../assets/generalization-gap.svg)

All reported Therm-FM results use the released configuration B: one GPU, training learning rate `1.5e-4`, embedding/recovery learning rate `1.5e-3`, batch size 40, and validation-best checkpoint selection. The earlier four-GPU configuration is retained only as an optimization-sensitivity comparison in the paper and [result record](RESULTS.md).

### Source-suite anchor — fixed-design prediction is already highly accurate

Across S1's eleven steady, transient, and industrial source tasks, Therm-FM L records the strongest grouped results, with MAE ranging from 0.012–0.049 K on HS steady tasks, 0.004–0.030 K on HS transient tasks, and 0.008 K on the industrial group. S1 therefore anchors what is attainable when the physical design remains fixed; it is not pooled with the controlled S2–S5 comparison.

### Finding 1 — added in-support physics is difficult but learnable

The best RMSE rises from 0.443 K on S2 to 0.716 K on S3 and 0.933 K on S4. Explicit material and boundary channels increase the learning burden, but the degradation remains gradual when the relevant case families and parameter support appear in training.

### Finding 2 — model ranking depends on physical support

Therm-FM L leads all six S2 metrics, while Therm-FM B leads all six S3 metrics and the S4 global-field errors. Therm-FM L retains the best S4 hotspot-oriented errors. Among operator baselines, SAU-FNO is strongest on S2 and S4 while U-FNO is strongest on S3. Model size therefore does not translate monotonically into accuracy once material and boundary inputs are introduced.

### Finding 3 — structural OOD is a different failure mode

With the same seven-channel schema, the best RMSE jumps from 0.933 K on S4 to 15.51 K on S5, an approximately 16.6× increase. Therm-FM T becomes the strongest frozen S5 model even though B and L lead within-family evaluation. Diverse observed layouts, materials, and cooling conditions do not establish transfer to an unseen chiplet system.

### Finding 4 — limited target labels are a practical recovery path

Ten labels per OOD case—50 labels total—reduce the best S5 RMSE to 2.73 K and the best MAE to 2.25 K. The first labels recover most of the zero-shot gap, followed by diminishing returns. Zero-shot robustness, low-label adaptation, and high-budget target accuracy should therefore be reported as distinct capabilities.


## What is public now

- the [complete S1–S5 benchmark result record](RESULTS.md), with the S1 datasets released and the unified S1 evaluator on the way;
- S2–S5 data tensors and the S5 case manifest;
- 112 baseline checkpoints spanning the 11 S1 source tasks and S2/S3/S4;
- eight baseline configurations across three model families;
- zero-shot and few-shot execution paths;
- one shared metric implementation and summary command; and
- a citable Zenodo software release.

The arXiv manuscript contains the full related-work comparison, complete baseline tables, per-case S5 analysis, optimization-sensitivity study, and formal citation.
