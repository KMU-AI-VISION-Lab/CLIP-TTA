# Realistic Test-Time Adaptation of Vision-Language Models (StatA) [CVPR 2025 - Highlight]

<a href="https://arxiv.org/abs/2501.03729" style="vertical-align:middle; display:inline;">
    <img src="https://img.shields.io/badge/cs.CV-arXiv%3A2501.03729-B31B1B.svg" class="plain" style="height:25px;" />
</a>

       
The official implementation of [*Realistic Test-Time Adaptation of Vision-Language Models*](https://arxiv.org/abs/2501.03729).

Authors:
[Maxime Zanella*](https://scholar.google.com/citations?user=FIoE9YIAAAAJ&hl=fr&oi=ao),
[Clément Fuchs*](https://scholar.google.com/citations?user=ZXWUJ4QAAAAJ&hl=fr&oi=ao),
[Christophe De Vleeschouwer](https://scholar.google.ca/citations?user=xb3Zc3cAAAAJ&hl=en),
[Ismail Ben Ayed](https://scholar.google.com/citations?user=29vyUccAAAAJ&hl=fr&oi=ao).

*Denotes equal contribution

## Quick Overview

We introduce **StatA**, a robust and versatile unsupervised transductive method designed to handle diverse deployment scenarios, including those involving a variable number of effective classes during testing. Our approach features a novel anchor-based regularization term specifically crafted for Vision-Language Models (VLMs). This term serves as a statistical anchor, preserving the initial knowledge of the text encoder, especially in low-data settings.

The experiments presented in this paper are organized into two main categories:  
1. **Batch Adaptation: Test-time adaptation methods are applied independently to each batch with a varying number of effective classes.**

   <div align="center" style="margin-top:20px; margin-bottom:20px;">
      <img src="images/realistic_batch.png" alt="Realistic Scenarios" width="500">
      <p style="font-size:75%;"><em>Realistic batches may not contain all the classes of interest.</em></p>
   </div>

   <div align="center" style="margin-top:20px; margin-bottom:20px;">
      <img src="images/summary_batch.png" alt="Batch Adaptation" width="500">
      <p style="font-size:75%;"><em>StatA brings consistent improvement when facing Low (between 2 and 10), Medium (between 5 and 25) number of effective classes (Keff) in each batch, or All classes. In comparison, other transductive methods engender significant performance drops in at least one scenario.</em></p>
   </div>





2. **Online Adaptation: Test-time adaptation methods are applied to a continuous stream of batches with varying correlation in the appearance of each class.**  

   <div align="center" style="margin-top:20px; margin-bottom:20px;">
      <img src="images/realistic_online.png" alt="Realistic Online Scenarios" width="500">
      <p style="font-size:75%;"><em>Realistic data streams contain correlated frames.</em></p>
   </div>

   <div align="center" style="margin-top:20px; margin-bottom:20px;">
      <img src="images/summary_online.png" alt="Online Adaptation" width="500">
      <p style="font-size:75%;"><em>StatA shows strong performance when applied on streams of data, with Low or High correlation between batches, and when all the classes are appearing sequentially (Separate).</em></p>
   </div>


## Table of Contents

1. [Installation](#installation) 
2. [Batch adaptation](#batch-adaptation)
3. [Online adaptation](#online-adaptation)
4. [Citation](#citation)
5. [Contact](#contact) 


---

## Installation
This repository requires to install an environment and datasets:
### Environment
Create a Python environment with your favorite environment manager. For example, with `conda`: 
```bash
conda create -y --name my_env python=3.10.0
conda activate my_env
pip3 install -r requirements.txt
```
And install Pytorch according to your configuration:
```bash
pip3 install torch==2.0.1 torchaudio==2.0.2 torchvision==0.15.2
```
### Datasets
Please follow [DATASETS.md](DATASETS.md) to install the datasets.
You will get a structure with the following dataset names:
```
$DATA/
|–– imagenet/
|–– caltech-101/
|–– oxford_pets/
|–– stanford_cars/
|–– oxford_flowers/
|–– food-101/
|–– fgvc_aircraft/
|–– sun397/
|–– dtd/
|–– eurosat/
|–– ucf101/
```

---

## Batch Adaptation
We present the basic usage to get started with our method. Each batch is generated using a data sampler (see `sampler.py`) called **BatchSampler**. The **BatchSampler** dynamically samples a specified number of effective classes (i.e., the number of classes effecitvely present in each batch) and corresponding indices from the dataset.


Here is an example for the imagenet dataset, with the CLIP-ViT-B/16 architecture, a batch size of 64, a variable number of effective classes between 1 and 4. This experiment is run 1000 times.
```bash
python3 main.py --root_path /path/to/datasets/folder --dataset imagenet --method StatA --backbone vit_b16 --batch_size 64 --num_class_eff_min 1 --num_class_eff_max 4 --n_tasks 1000
```

To run the whole experiment of Table 1 in the paper, use the following command:
```bash
bash ./scripts/StatA_batch.sh /path/to/datasets/folder vit_b16
```

The table below summarizes the **average performance** (averaged over 11 datasets) you should obtain by running the above script.

For a small batch size of **64**, we focus on three realistic configurations:
- **Very Low**: 1–4 effective classes
- **Low**: 2–10 effective classes
- **Medium**: 5–25 effective classes

For a larger batch size of **1000**, we examine:
- **Medium**: 5–25 effective classes
- **High**: 25–50 effective classes
- **Very High**: 50–100 effective classes

Additionally, we provide the results on the full dataset, containing **All Classes**.


|     | Very Low <br> (B=64)    | Low <br> (B=64)     | Medium <br> (B=64)  | Medium <br> (B=1000)  | High <br> (B=1000)  | Very High <br> (B=1000)  | All Classes <br> (All dataset) |
|----------------|----------------|-----------------|--------------------|--------------------|------------------|---------------------|----------------|
| CLIP       | 65.2           | 65.2            | 65.2              | 65.2               | 65.2             | 65.2               | 65.2           |
| MTA        | 66.6  <br> `↑1.4` | 66.6  <br> `↑1.4` | 66.6  <br> `↑1.4` | 66.6  <br> `↑1.4` | 66.6  <br> `↑1.4` | 66.6  <br> `↑1.4` | 66.6  <br> `↑1.4` |
| Dirichlet | <ins>68.5</ins> ✅ <br> `↑3.3` | **70.3** ✅ <br> `↑5.1` | **67.5** ✅ <br> `↑2.2` | <ins>64.4</ins> ❌ <br> `↓0.8` | 45.3 ❌ <br> `↓20.0` | 33.6 ❌ <br> `↓31.6` | 29.5 ❌ <br> `↓35.7` |
| ZLaP       | 27.5 ❌ <br> `↓37.7` | 35.2 ❌ <br> `↓30.0` | 44.7 ❌ <br> `↓20.6` | 41.5 ❌ <br> `↓23.7` | 52.2 ❌ <br> `↓13.0` | 58.4 ❌ <br> `↓6.8` | 66.4 ✅ <br> `↑1.1` |
| TransCLIP  | 38.9 ❌ <br> `↓26.3` | 40.4 ❌ <br> `↓24.8` | 42.7 ❌ <br> `↓22.5` | 56.5 ❌ <br> `↓8.7` | <ins>62.0</ins> ❌ <br> `↓3.3` | <ins>64.4</ins> ❌ <br> `↓0.8` | **70.3** ✅ <br> `↑5.1` |
| **StatA (ours)** | **70.4** ✅ <br> `↑5.1` | <ins>69.3</ins> ✅ <br> `↑4.1` | <ins>67.4</ins> ✅ <br> `↑2.2` | **69.7** ✅ <br> `↑4.4` | **69.8** ✅ <br> `↑4.5` | **69.0** ✅ <br> `↑3.7` | <ins>69.9</ins> ✅ <br> `↑4.7` |



- **B** indicates the batch size 
- ✅ (Green): Indicates a performance gain compared to the zero-shot baseline (CLIP).
- ❌ (Red): Indicates a performance deterioration compared to the zero-shot baseline (CLIP).

StatA demonstrates robustness across all scenarios, whereas other transductive methods exhibit strong performance only within specific, narrow application ranges.
For more detailed results, please refer to **Table 1** in the paper.

---

## Online Adaptation
We present the basic usage to get started with our method. Each batch is generated using a data sampler (see `sampler.py`) called **OnlineSampler**. The **OnlineSampler** dynamically samples indices from the dataset according to a Dirichlet law parametrized by gamma (see Appendix of the paper for more details).

Here is an example for the imagenet dataset, with the CLIP-ViT-B/16 architecture, a batch size of 128, a stream correlation factor gamma of 0.1. This experiment is run 100 times.
```bash
python3 main.py --root_path /path/to/datasets/folder --dataset imagenet --method StatA --backbone vit_b16 --batch_size 64 --online --gamma 0.1 --n_tasks 100
```

To run the whole experiment of Table 2 in the paper, use the following command:
```bash
bash ./scripts/StatA_online.sh /path/to/datasets/folder vit_b16
```

The table below presents the **average performance** (averaged over 11 datasets) you should obtain by running the above script.

We focus on four realistic configurations:
1. **Low correlation in the stream** ($\gamma = 0.1$).
2. **Medium correlation in the stream** ($\gamma = 0.01$).
3. **High correlation in the stream** ($\gamma = 0.001$).
4. **Classes appearing sequentially** (Separate).




|           | Low | Medium  | High | Separate  |
|------------------|----------------------|--------------------------|-------------------------|--------------------------|
| CLIP            | 65.2                | 65.2                    | 65.2                   | 65.2                    |
| MTA             | 66.6 `↑1.4`         | 66.6 `↑1.4`             | 66.6 `↑1.4`            | 66.6 `↑1.4`         |
| TENT             | 65.8 `↑0.6`     | 65.5 `↑0.2`           | 65.3 `↑0.1`          | 64.5 `↓0.7`           |
| TDA             | **67.7 `↑2.5`**     | <ins>67.1</ins> `↑1.9`           | <ins>66.8</ins> `↑1.6`          | <ins>66.6</ins> `↑1.4`           |
| DMN             | <ins>67.2</ins> `↑2.0`         | 66.5 `↑1.2`             | 66.3 `↑1.0`            | 65.8 `↑0.6`             |
| **StatA (ours)**  | 67.0 `↑1.7`       | **68.9 `↑3.7`**         | **69.5 `↑4.2`**        | **69.1 `↑3.8`**         |


StatA demonstrates robustness across all scenarios, providing a strong baseline for future reasearch in the field.
For more detailed results, please refer to **Table 2** in the paper.

---

## Geometry Analysis

This fork also includes a lightweight analysis extension for comparing class-conditional CLIP embedding geometry across the ImageNet family. This analysis code is not part of the original StatA paper and does not modify the default adaptation pipeline in `main.py`.

### Goal

For a semantic class `y`, the analysis compares the CLIP image-feature cloud from dataset `A` with the cloud from dataset `B`:

```text
Z_A^y = {f(x) | x in class y from dataset A}
Z_B^y = {f(x) | x in class y from dataset B}
```

The main question is whether the same class keeps a similar geometry across domains such as ImageNet_v1, ImageNet_v2, and ImageNet_Sketch.

### Expected dataset layout

The loaders in this fork support the canonical ImageNet-family naming below, while still keeping backward-compatible fallbacks for the older directory names.

```text
/data2/TTA_dataset/
|-- ImageNet_v1/
|   |-- train/ or images/train/
|   `-- val/   or images/val/
|-- ImageNet_v2/
|   `-- imagenetv2-matched-frequency-format-val/
|-- ImageNet_Sketch/
|   `-- images/ or class folders directly
|-- ImageNet_A/
|   `-- images/ or class folders directly
`-- ImageNet_R/
    `-- images/ or class folders directly
```

Supported ImageNet-family datasets:
- `imagenet`
- `imagenet_v2`
- `imagenet_sketch`
- `imagenet_a`
- `imagenet_r`

The internal CLI keys remain lowercase for compatibility, but saved feature files, JSON outputs, and displayed dataset names use the canonical naming:
- `ImageNet_v1`
- `ImageNet_v2`
- `ImageNet_Sketch`
- `ImageNet_A`
- `ImageNet_R`

The ImageNet-family loaders align labels to ImageNet-1k indices whenever folder names permit it. For subset datasets such as ImageNet_R, the dumped feature files also expose `available_imagenet_indices` and `subset_classnames`.

### Step 1: dump CLIP features once

Each feature dump stores a `.pt` dictionary with:
- `features`
- `labels`
- `classnames`
- `dataset_name`
- `backbone`
- `image_paths` when available

Example:

```bash
python tools/dump_features.py --dataset imagenet --root_path /data2/TTA_dataset --backbone vit_b16 --cache_dir ./caches/geometry_vit_b16/ImageNet_v1 --output ./caches/geometry_vit_b16/ImageNet_v1/ImageNet_v1_vit_b16_features.pt --device cuda:0
python tools/dump_features.py --dataset imagenet_v2 --root_path /data2/TTA_dataset --backbone vit_b16 --cache_dir ./caches/geometry_vit_b16/ImageNet_v2 --output ./caches/geometry_vit_b16/ImageNet_v2/ImageNet_v2_vit_b16_features.pt --device cuda:0
python tools/dump_features.py --dataset imagenet_sketch --root_path /data2/TTA_dataset --backbone vit_b16 --cache_dir ./caches/geometry_vit_b16/ImageNet_Sketch --output ./caches/geometry_vit_b16/ImageNet_Sketch/ImageNet_Sketch_vit_b16_features.pt --device cuda:0
python tools/dump_features.py --dataset imagenet_a --root_path /data2/TTA_dataset --backbone vit_b16 --cache_dir ./caches/geometry_vit_b16/ImageNet_A --output ./caches/geometry_vit_b16/ImageNet_A/ImageNet_A_vit_b16_features.pt --device cuda:0
python tools/dump_features.py --dataset imagenet_r --root_path /data2/TTA_dataset --backbone vit_b16 --cache_dir ./caches/geometry_vit_b16/ImageNet_R --output ./caches/geometry_vit_b16/ImageNet_R/ImageNet_R_vit_b16_features.pt --device cuda:0
```

### Step 2: run geometry evaluation

Example cross-dataset evaluation:

```bash
python tools/geometry_eval.py \
  --source_feature_file ./caches/geometry_vit_b16/ImageNet_v1/ImageNet_v1_vit_b16_features.pt \
  --target_feature_file ./caches/geometry_vit_b16/ImageNet_v2/ImageNet_v2_vit_b16_features.pt \
  --output ./outputs/geometry_vit_b16/ImageNet_v1_vs_ImageNet_v2_nc200_spc10_knn3_5_seed1.json \
  --num_classes 200 \
  --samples_per_class 10 \
  --seed 1 \
  --compute_knn_distribution \
  --compute_distance_histogram \
  --compute_graph_stats \
  --knn_k 3 5 \
  --device cuda:0
```

Example same-dataset split-half upper bound:

```bash
python tools/geometry_eval.py \
  --same_dataset_upper_bound \
  --source_feature_file ./caches/geometry_vit_b16/ImageNet_v1/ImageNet_v1_vit_b16_features.pt \
  --output ./outputs/geometry_vit_b16/ImageNet_v1_split_half_upper_bound_nc200_spc10_rep5_knn3_5_seed1.json \
  --num_classes 200 \
  --samples_per_class 10 \
  --min_samples_per_class_for_split 20 \
  --upper_bound_num_repeats 5 \
  --seed 1 \
  --compute_knn_distribution \
  --compute_distance_histogram \
  --compute_graph_stats \
  --knn_k 3 5 \
  --device cuda:0
```

The split-half mode is a reliability reference. It estimates how high a metric can be when comparing two disjoint subsets from the same dataset, so you can tell whether a low cross-dataset score reflects real domain shift or just finite-sample noise.

### Step 3: optional UMAP visualization without rerunning the full evaluator

If feature `.pt` files already exist, you do not need to rerun `tools/geometry_eval.py` just to create plots. Use `tools/geometry_viz.py` to reuse the feature dumps and, optionally, an existing evaluation JSON to auto-select interesting classes.

Example: visualize classes with the largest centroid shifts.

```bash
python tools/geometry_viz.py \
  --source_feature_file ./caches/geometry_vit_b16/ImageNet_v1/ImageNet_v1_vit_b16_features.pt \
  --target_feature_file ./caches/geometry_vit_b16/ImageNet_Sketch/ImageNet_Sketch_vit_b16_features.pt \
  --eval_json ./outputs/geometry_vit_b16/ImageNet_v1_vs_ImageNet_Sketch_nc200_spc10_knn3_5_seed1.json \
  --output_prefix ./outputs/geometry_vit_b16/ImageNet_v1_vs_ImageNet_Sketch_shift_focus \
  --viz_select_mode centroid_shift \
  --viz_topk 12 \
  --samples_per_class 10 \
  --viz_max_points_per_class 50 \
  --viz_dim 3 \
  --interactive
```

Available class-selection modes:
- `manual`
- `retrieval_error`
- `centroid_shift`
- `topology`

Outputs:
- `*_umap_2d.png` or `*_umap_3d.png`
- `*_umap_2d.html` or `*_umap_3d.html` when `--interactive` is used
- `*_viz_metrics.json` with `trustworthiness` and `continuity`

### Metric guide

The geometry metrics can be read as three complementary views of a class cloud in feature space.

Location and utility:
- `centroid_cosine`, `centroid_euclidean`
  - These compare the class mean in two datasets.
  - Intuition: if you average many "dog" features, do the two average-dog vectors still point in the same direction and stay close together?
- `retrieval_acc_*`
  - For each source class representation, retrieve the most similar target class.
  - High accuracy means class identity is still easy to match across datasets.

Shape and orientation:
- `cov_frobenius`
  - Measures whether the class cloud keeps a similar spread and anisotropy.
  - Intuition: do the two clouds have a similar volume and deformation?
- `pca_subspace`
  - Measures whether the dominant directions of variation still align.
  - This is often the clearest sign that the feature space is relying on similar or different cues across domains.

Topology and neighborhood structure:
- `knn_wasserstein`, `knn_mean_diff`, `knn_std_diff`
  - Compare local neighbor-distance distributions without requiring sample correspondence.
  - Intuition: are points inside the class similarly dense and similarly arranged locally?
- `distance_histogram_js`, `distance_histogram_wasserstein`
  - Compare the whole within-class distance distribution.
  - Intuition: does the class preserve its overall internal distance map?
- `neighbor_graph_stats`
  - Summarizes average local spacing from the k-nearest-neighbor graph.

Deprecated metric kept for backward compatibility:
- `pairwise_spearman`, `pairwise_pearson`
  - These are not reliable for cross-dataset comparison without sample correspondence because entry-wise distances do not refer to the same object pairs.

In the JSON output, `same` compares the same semantic class across datasets, while `different` is a control obtained by comparing a source class to other target classes. A useful metric should show a clear gap between `same` and `different`.

### Full driver script

```bash
CUDA_VISIBLE_DEVICES=0 GEOMETRY_DEVICE=cuda:0 bash scripts/run_geometry_imagenet_family.sh lab_server vit_b16
```

Example: parallel feature dumping and evaluation on the `naver` server with 4 GPUs. Each dataset-level job is independent, so the script can run ImageNet-family feature dumps and geometry evaluations in parallel waves across GPUs.

```bash
PARALLEL_EXECUTION=1 \
GPU_IDS=0,1,2,3 \
RUN_IMAGENET_A=0 \
RUN_IMAGENET_R=1 \
RUN_INTER_CLASS_GEOMETRY=1 \
SIGN_EPSILON=0.05 \
NUM_CLASSES=100 \
IMAGENET_SAMPLES_PER_CLASS=30 \
IMAGENET_V2_SAMPLES_PER_CLASS=10 \
IMAGENET_SKETCH_SAMPLES_PER_CLASS=10 \
IMAGENET_R_SAMPLES_PER_CLASS=10 \
IMAGENET_UPPER_BOUND_SAMPLES_PER_CLASS=25 \
IMAGENET_V2_UPPER_BOUND_SAMPLES_PER_CLASS=5 \
IMAGENET_SKETCH_UPPER_BOUND_SAMPLES_PER_CLASS=10 \
UPPER_BOUND_REPEATS=5 \
FEATURE_BATCH_SIZE=64 \
NUM_WORKERS=8 \
bash scripts/run_geometry_imagenet_family.sh naver vit_b16
```

The script:
- skips feature dumping if the expected feature file already exists
- can run independent dataset jobs in parallel when `PARALLEL_EXECUTION=1`
- runs cross-dataset evaluation for ImageNet_v1 vs ImageNet_v2 and ImageNet_v1 vs ImageNet_Sketch
- optionally runs ImageNet_v1 vs ImageNet_A and ImageNet_v1 vs ImageNet_R
- runs same-dataset split-half upper bounds for ImageNet_v1, ImageNet_v2, and ImageNet_Sketch

`run_geometry_imagenet_family.sh` dataset-location shortcuts:
- `lab_server` -> `/data2/TTA_dataset`
- `naver` -> `/data/tta/ImageNet_Family`
- any other first argument is treated as an explicit dataset root path

### Optional adaptation comparison

```bash
python tools/dump_adapted_features.py --dataset imagenet_sketch --root_path /data2/TTA_dataset --backbone vit_b16 --method StatA --batch_size 64
```

This script saves frozen CLIP image embeddings together with zero-shot and adapted prediction outputs for a sampled batch. It does not currently extract adapted image embeddings from the online adaptation methods.

---

## Citation

If you find this repository useful, please consider citing our paper:
```
@article{zanella2025realistic,
title={Realistic Test-Time Adaptation of Vision-Language Models},
author={Zanella, Maxime and Fuchs, Cl{\'e}ment and De Vleeschouwer, Christophe and Ben Ayed, Ismail}
journal={arXiv preprint arXiv:2501.03729},
  year={2025}
}
```

You can also cite the TransCLIP paper on which this work is based on:
```
@article{zanella2024boosting,
  title={Boosting vision-language models with transduction},
  author={Zanella, Maxime and G{\'e}rin, Beno{\^\i}t and Ben Ayed, Ismail},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={62223--62256},
  year={2024}
}
```

## Contact

For any inquiries, please contact us at [maxime.zanella@uclouvain.be](mailto:maxime.zanella@uclouvain.be) and [clement.fuchs@uclouvain.be](mailto:clement.fuchs@uclouvain.be) or feel free to [create an issue](https://github.com/MaxZanella/StatA/issues).


## License
[AGPL-3.0](https://github.com/MaxZanella/StatA/blob/main/LICENSE)

## Acknowledgment
This repository is mainly based on [CLIP](https://github.com/openai/CLIP) and [TransCLIP](https://github.com/MaxZanella/transduction-for-vlms). 
