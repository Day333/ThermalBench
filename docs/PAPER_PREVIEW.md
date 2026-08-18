# ThermalBench paper preview

> **Status:** manuscript in preparation. This page exposes the benchmark's central design and selected results without presenting an unpublished draft as the final paper. The formal citation will be added to the repository when the paper is public.

## Motivation

Learning-based thermal solvers are increasingly evaluated under power, geometry, material, and cooling variation. Yet comparisons remain difficult because method papers often use different private or partially released datasets, simulators, splits, preprocessing, and metrics. A result can therefore improve because the task changed—not because the predictor is better.

ThermalBench addresses this evaluation gap rather than proposing another thermal model. Its novelty is the combination of:

- an open benchmark and common evaluation contract;
- fixed splits, channel semantics, labels, and metrics;
- convolutional, neural-operator, and PDE-foundation-model baselines;
- progressive physical support matched to deployment needs; and
- case-disjoint structural out-of-distribution evaluation.

The release also bundles the practical path around the benchmark: data formatting, model execution, training, inference, adaptation, aggregation, and result export share one interface.

![Five ThermalBench generalization scopes](../assets/thermalbench-overview.svg)

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

S1 consolidates established fixed-design thermal-learning tasks: steady-state and transient HS-SC, HS-QC, and HS-OC Alpha EV6 configurations, plus IND-8C and IND-32C industrial cases. It contains 32,000 samples across eight tasks and five physical designs. These tasks are collected without modification from the [ARO](https://github.com/Mia-WMY/ARO) and [Therm-FM](https://arxiv.org/abs/2605.22663) lineage; their [complete recorded results](RESULTS.md) remain task-specific.

S2–S4 use the same ten system families and generation lineage. S2 originates from Qipan Wang *et al.*'s [ATPlace2.5D public cases and thermal setup](https://github.com/PKU-IDEA/ATPlace_pub); S3–S4 extend that foundation consistently. They independently sample layouts while cumulatively exposing material and boundary variables. Each scope contains 15,000 samples. These tracks test generalization within represented structural support.

S5 preserves the S4 physical schema but replaces the represented systems with five case-disjoint systems. Its 5,000 samples stress unseen chiplet count, size heterogeneity, power density, power concentration, and utilization. This distinguishes learning many observed physical variables from transferring to an unseen system structure.

## Selected findings

![ThermalBench selected generalization results](../assets/generalization-gap.svg)

### Source-suite anchor — fixed-design prediction is already highly accurate

Across S1's eight steady, transient, and industrial source tasks, Therm-FM L records the strongest result, with task-specific RMSE between 0.009 and 0.076 K. S1 therefore anchors what is attainable when the physical design remains fixed; it is not pooled with the controlled S2–S5 comparison.

### Finding 1 — added in-support physics is difficult but learnable

The best RMSE rises from 0.657 K on S2 to 0.802 K on S3 and 1.327 K on S4. Explicit material and boundary channels increase the learning burden, but the degradation remains gradual when the relevant case families and parameter support appear in training.

### Finding 2 — model ranking depends on physical support

SAU-FNO is strongest on S2, while U-FNO is strongest on S3 and S4 in the preliminary comparison. Therm-FM size also does not translate monotonically into in-support accuracy. A result on a fixed or simpler benchmark is therefore not a reliable proxy for performance after new physical dimensions are introduced.

### Finding 3 — structural OOD is a different failure mode

With the same seven-channel schema, the best RMSE jumps from 1.327 K on S4 to 15.99 K on S5. Several of the strongest in-support models degrade more sharply than models that ranked lower on S4. Diverse observed layouts, materials, and cooling conditions do not establish transfer to an unseen chiplet system.

### Finding 4 — limited target labels are a practical recovery path

Ten labels per OOD case—50 labels total—reduce the best S5 result to 3.19 K. The first labels recover most of the zero-shot gap, followed by diminishing returns. Zero-shot robustness, low-label adaptation, and high-budget target accuracy should therefore be reported as distinct capabilities.

## Central takeaway

> **In-support multi-physics generalization does not imply structural generalization.**

Thermal-learning papers should report whether test cases are within a represented design family, unseen parameter combinations, unseen geometry, or case-disjoint structural OOD. A single undifferentiated “generalization” score obscures the largest capability gap observed in ThermalBench.

## What is public now

- the [complete S1–S5 benchmark result record](RESULTS.md), with S1 data/evaluation packaging on the way;
- S2–S5 data tensors and the S5 case manifest;
- 24 S2/S3/S4 baseline checkpoints;
- eight baseline configurations across three model families;
- zero-shot and few-shot execution paths;
- one shared metric implementation and summary command; and
- a citable Zenodo software release.

The complete paper, full related-work comparison, complete baseline tables, per-case S5 analysis, and formal author/venue citation will be linked after public release.
