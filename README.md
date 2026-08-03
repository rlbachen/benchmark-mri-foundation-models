[![bioRxiv](https://img.shields.io/badge/bioRxiv-2026.05.15.725427-b31b1b.svg)](https://www.biorxiv.org/content/10.64898/2026.05.15.725427v3)

## additional-evaluation-brainfm

A reproducible, end-to-end benchmarking framework for evaluating structural MRI foundation models across multiple large-scale neuroimaging datasets and clinically relevant tasks.

---

## Overview

Foundation models for brain MRI promise general-purpose representations that transfer across tasks — but how well do they actually hold up against classical pipelines like FreeSurfer? This project benchmarks five self-supervised/pretrained models against FreeSurfer-derived features across three public datasets, evaluating sex classification, age regression, and Parkinson's disease classification.

All preprocessing, feature extraction, and downstream evaluation are fully scripted and run on a SLURM HPC cluster using containerized pipelines.

---

## Models Evaluated

| Model | Architecture | Pretraining | Native Preprocessing | Preprocessing in This Study |
|---|---|---|---|---|
| **AnatCL** (Global Descriptor) | ResNet-18 | Weakly Contrastive | CAT12 VBM | CAT12 (VBM) |
| **AnatCL** (Local Descriptor) | ResNet-18 | Weakly Contrastive | CAT12 VBM | CAT12 (VBM) |
| **BrainIAC** | ViT | Self-supervised, SimCLR contrastive learning | BrainIAC T1 / T2 pipeline | TurboPrep *(main)* + BrainIAC pipeline *(sensitivity)* |
| **SwinBrain** | Swin UNETR | Self-supervised | SwinBrain Pipeline | TurboPrep *(main)* + BrainIAC pipeline *(sensitivity)* |
| **3D-Neuro-SimCLR** | ResNet-18 | Self-supervised, SimCLR | TurboPrep | TurboPrep |
| **CNN3D** *(baseline)* | Custom 4-layer Convolutional Neural Network | Random initialization (Untrained) | — | TurboPrep *(main)* |
| **FreeSurfer** *(reference)* | — | — | FreeSurfer | Pre-computed FreeSurfer features |

FreeSurfer features use both the **Schaefer 400-parcel** and **aparc** (Desikan-Killiany) atlases, combining cortical thickness and surface area.

> **Note on preprocessing:** BrainIAC and SwinBrain each come with their own native preprocessing pipelines. In the main benchmarking experiments, all models are evaluated on TurboPrep outputs for a fair comparison. A separate **preprocessing sensitivity analysis** additionally evaluates SwinBrain and BrainIAC across all three preprocessing variants (TurboPrep, BrainIAC pipeline same template, BrainIAC pipeline different template) to quantify how much preprocessing choice alone affects downstream performance and feature representations — independently of the model itself.

---

## Datasets
 
| Dataset | N | Sex (M/F) | Mean Age | Clinical Label | Notes |
|---|---|---|---|---|---|
| **HBN** | 1,000 | 519M / 481F | 11.33 ± 3.77 yrs | — | Adolescent, multi-site |
| **NKI** | 958 | 357M / 601F | 41.99 ± 20.22 yrs | — | Lifespan, healthy |
| **PPMI** | 894 | 555M / 339F | 62.48 ± 9.99 yrs | Parkinson's (711 PD) | Clinical cohort |

---

## Tasks

**Classification**
- Sex classification (HBN, NKI, PPMI)
- Parkinson's disease classification (PPMI)

**Regression**
- Age regression — males and females evaluated separately (all datasets)

---

## Preprocessing Pipelines

Two parallel preprocessing pipelines are used, depending on the model:

### CAT12 VBM (for AnatCL)

CAT12 runs voxel-based morphometry, producing modulated, warped grey matter density maps (`mwp1*.nii`) registered to MNI space. Jobs are parallelized per-subject via SLURM arrays using Boutiques + Apptainer.

Output shape after interpolation: **(121 × 128 × 121)**

### TurboPrep (for BrainIAC, SwinBrain, SimCLR, CNN3D)

TurboPrep performs skull stripping, bias correction, and MNI registration. Each subject runs in an Apptainer container.

```bash
apptainer run turboprep.sif T1w.nii.gz output_dir/ mni_template.nii -m t1 -r r
```

Output used: `normalized.nii.gz`. Each model applies its own resizing at load time:

| Model | Input Shape |
|---|---|
| BrainIAC | 96 × 96 × 96 |
| SwinBrain | 128 × 128 × 64 (3-channel repeat) |
| 3D-Neuro-SimCLR | 150 × 192 × 192 (center-cropped) |
| CNN3D | 91 × 109 × 91 |

---

## Containers

### CAT12 — `rlbachen/cat12-segment-smooth`

[![Docker Pulls](https://img.shields.io/docker/pulls/rlbachen/cat12-segment-smooth)](https://hub.docker.com/r/rlbachen/cat12-segment-smooth)
![Docker Image Size](https://img.shields.io/docker/image-size/rlbachen/cat12-segment-smooth/latest)
![Docker Tag](https://img.shields.io/docker/v/rlbachen/cat12-segment-smooth)

The CAT12 preprocessing pipeline uses a custom Docker/Apptainer image built on top of [`jhuguetn/cat12`](https://hub.docker.com/r/jhuguetn/cat12), with one key addition: a **built-in smoothing step (`[6 6 6]` FWHM kernel)** applied after segmentation. This keeps the smoothing stage reproducible and eliminates the need for a separate post-processing call.

| | |
|---|---|
| **Image** | `rlbachen/cat12-segment-smooth:latest` |
| **Base image** | `jhuguetn/cat12` |
| **Added** | Gaussian smoothing kernel `[6 6 6]` mm FWHM |
| **Size** | 14.4 GB |
| **Source** | [github.com/rlbachen/cat12-segment-smooth](https://github.com/rlbachen/cat12-segment-smooth) |
| **Docker Hub** | [hub.docker.com/r/rlbachen/cat12-segment-smooth](https://hub.docker.com/r/rlbachen/cat12-segment-smooth) |

The build source for this image is available in this repository:
 
**`cat12.dockerfile`** — extends the base CAT12 image by injecting the custom entrypoint:

**`entrypoint.sh`** — runs CAT12 segmentation, locates the output `mwp1` file, then applies `[6 6 6]` mm FWHM smoothing in a single call:

**Pull and convert to Apptainer `.sif`:**

```bash
apptainer pull cat12_prepro.sif docker://rlbachen/cat12-segment-smooth:latest
```

The `.sif` file is then referenced in the Boutiques descriptor and passed to all SLURM jobs via `--imagepath`.


---

## Evaluation Framework

### Feature Extraction

Features are extracted in inference mode (no fine-tuning). Each model produces a fixed-size embedding per subject:

- **AnatCL**: Average of 5 cross-validation fold encoders
- **BrainIAC**: CLS token from ViT output
- **SwinBrain**: Global average pooling over `encoder10` hook output
- **3D-Neuro-SimCLR**: ResNet-18 backbone output (projector removed)
- **CNN3D**: FC layer output (512-d), random weight initialized untrained baseline (seed = 0)
  
### Downstream Classifier / Regressor

A **Random Forest** (200 trees, `max_depth=6`, `min_samples_split=5`) is used as the downstream probe for all tasks. This is intentionally simple — differences in performance reflect the quality of the learned representations, not the power of the downstream model.

**Evaluation protocol:**
- 5 random seeds × 5-fold stratified cross-validation
- 10% held-out test set per seed (unseen during all CV)
- **Learning curves**: Training fraction α ∈ {1%, 5%, 10%, 15%, 25%, 40%, 60%, 80%, 100%}
- Metrics: Balanced Accuracy (classification), MAE (regression)

### Statistical Testing

Permutation tests (Bonferroni-corrected, 9,999 resamples) compare each model against FreeSurfer on fold-level accuracy/MAE. Only models beating FreeSurfer at full training data are tested.

---

### Dependencies (key packages)

```
torch
monai
nibabel
scikit-learn
scipy
pandas
matplotlib
anatcl
boutiques
```

---

## Correlation Analysis

Beyond task performance, the framework computes cross-model feature correlation matrices to examine how differently each model represents the brain. Features are z-scored, hierarchically clustered per model, and visualized as a full pairwise heatmap. FreeSurfer features are color-annotated by feature type (thickness vs. surface area).

---

## Preprocessing Sensitivity Analysis

A secondary experiment isolates the effect of preprocessing on representation quality, independent of model architecture. For SwinBrain, BrainIAC, and CNN3D, features are extracted under each of three preprocessing variants on the same subjects:

| Preprocessing | Label | Description |
|---|---|---|
| TurboPrep | `_TP` | Skull stripping + MNI registration via TurboPrep |
| BrainIAC T1 | `_T1` | Native BrainIAC T1-weighted pipeline (with T1 template) |
| BrainIAC T2 | `_T2` | Native BrainIAC T1-weighted pipeline (with T2 template from BrainIAC's original repository) |

---

## Reproducibility Notes

- All SLURM jobs use fixed `--nodes=1` to avoid distributed non-determinism
- `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` all set to 1
- `torch.set_num_threads(1)` for CPU inference
- Preprocessing containers are pinned by `.sif` file (not pulled at runtime)

---

## License

See `LICENSE` for details. Dataset access is governed by the respective data use agreements (HBN, NKI, PPMI).
