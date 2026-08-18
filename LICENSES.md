# Licensing and attribution

ThermalBench uses separate licenses because software, datasets, upstream source tasks, and a research manuscript are different kinds of work.

| Material | License | What this permits |
|---|---|---|
| Original repository code | [MIT](LICENSE) | use, modify, distribute, and integrate while retaining the license notice |
| Original S2–S5 data, fixed splits/manifests, and released result records | [CC BY 4.0](LICENSE-DATA) | use, redistribute, benchmark, and extend with attribution and change disclosure |
| S1 source data and third-party components | upstream terms | consult and cite the source project; ThermalBench does not relicense them |
| ThermalBench manuscript and paper-derived prose, figures, and tables | © 2026 ThermalBench Authors; all rights reserved unless marked otherwise | normal scholarly quotation and citation; not included in the data license |

The vendored `model/scOT/` component retains its upstream license. The paper-derived prior-work image at `assets/prior-work-landscape.png` is displayed for project documentation and is not part of the CC BY dataset grant.

## How to use the benchmark correctly

You are welcome to:

- evaluate a new model on the published splits;
- compare with the released baselines;
- extend the data, simulator coverage, physical variables, or OOD cases;
- redistribute adapted CC BY data while identifying the changes.

For an academic publication, cite the ThermalBench version/DOI and clearly distinguish the released benchmark from your new method, data, or analysis. CC BY 4.0 legally permits sharing and adaptation with attribution; it does not permit false claims of authorship or provenance. Academic plagiarism and research-misconduct rules apply independently of the copyright license.

## Suggested attribution

> ThermalBench: An Open, Progressive Benchmark for Generalizable 2.5D/3D-IC Thermal Learning, The ThermalBench Authors, version used, https://doi.org/10.5281/zenodo.21992816, licensed data under CC BY 4.0. Changes, if any, are described by the reuser.

The software DOI currently archives the repository. When a standalone dataset DOI is released, use that dataset DOI in addition to the benchmark citation.
