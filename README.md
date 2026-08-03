## UniV: Unified Consistency-Driven ERUS Video Multi-Task Learning
by Yuncheng Jiang, Yiwen Hu, Zixun Zhang, Jun Wei, Chun-Mei Feng, Xuemei Tang, Xiang Wan, Yong Liu, Shuguang Cui, Zhen Li

---

## :sparkles: Introduction
![framework](./figs/framework.png) 
This repository now implements **UniV**, an upgraded end-to-end framework for ERUS video analysis with joint **frame-level lesion segmentation** and **video-level T-stage classification**. UniV replaces the earlier ASTR-specific heuristic pipeline with:

- **FAMH**: learnable frequency-adaptive mode harmonization instead of offline scan-mode conversion
- **UATSSM + LATA**: PyTorch-compatible ultrasound-aware temporal modeling and lesion-aware video aggregation
- **CTSI**: cross-task synergy between segmentation and classification with video-level consistency

The default code path uses `Data/ERUS/video_labels.txt` for 5-way T-stage supervision and geometry-based pseudo labels for scan mode.

## :mag: Prerequisites
---
### Clone repository

```shell
# clone project
git clone https://github.com/yuncheng97/ASTR.git
cd ASTR/

# create conda environment and install dependencies
conda env create -f environment.yaml
conda activate ASTR
```

### Download dataset
This database is available for only non-commercial use in research or educational purpose. As long as you use the database for these purposes, you can edit or process images and annotations in this database. Please sign the [license agreement](figs/ERUS-License.pdf) and send it to yuncheng.jiang97@gmail.com to obtain the download link.

After download the dataset, put the dataset in the "/data" folder


```shell
mkdir data/
```

### Download pretrained backbone

download the pretrained backbone weights and put them in the "/pretrained" folder. Then you can train the model on the ERUS-10K or your own dataset from scratch.
- [Res2Net50](https://drive.google.com/file/d/1RzSdIGhM6kR7yJQWHWy8ed7WNhGrt-m3/view?usp=sharing)
- [PVT_v2_b2](https://drive.google.com/file/d/1I8uPAEzKuI311V_HJpQ7Ppf-LDgi7K_O/view?usp=sharing)

```shell
mkdir pretrained/
```

### Download pretrained model (optional)

you can also download our [pretrained model checkpoint](https://drive.google.com/file/d/1hM7vZuKroNqbO0gaZiQVAP4xcSLXTjHW/view?usp=sharing) on ERUS-10K for evaluation.


### Legacy mode conversion
`scan_mode_convert.py` is kept as a legacy tool, but UniV no longer depends on offline scan-mode augmentation during training.
---


## :rocket: Training and evaluation
Set your own training configuration before training.

**Recommended shell entrypoint**
```shell
bash scripts/train.sh
```

Useful environment overrides:
```shell
DATA_ROOT=../Data/ERUS \
OUTPUT_ROOT=./results \
BACKBONE=res2net50 \
TRAIN_SIZE=352 \
CLIP_SIZE=3 \
BATCH_SIZE=2 \
EPOCHS=20 \
GPU_ID=0 \
NOTE=univ_run \
bash scripts/train.sh
```

**Training on single node**
```shell
python train.py \
    --gpu_id 0 \
    --batchsize 2 \
    --lr 0.0001 \
    --data_root ../Data/ERUS \
    --train_size 352 \
    --clip_size 3 \
    --backbone res2net50 \
    --scheduler cos \
    --optimizer adamw \
    --epoch 20 \
    --use_mode_pseudo_labels \
    --eval_split val \
    --note univ_run
```

**Minimal smoke train**
```shell
python train.py \
    --gpu_id 0 \
    --data_root ../Data/ERUS \
    --backbone res2net50 \
    --train_size 64 \
    --clip_size 3 \
    --batchsize 1 \
    --epoch 1 \
    --num_workers 0 \
    --limit_train_batches 2 \
    --limit_val_batches 1 \
    --use_mode_pseudo_labels \
    --note smoke
```


**Evaluation**
```shell
python eval.py \
    --gpu_id 0 \
    --data_root ../Data/ERUS \
    --train_size 352 \
    --clip_size 3 \
    --resume ./results/univ_run/log_xxx \
    --eval_split test \
    --task both \
    --use_mode_pseudo_labels
```

## :pray: Acknowledgement
This code of repository is built on [FLA-Net](https://github.com/jhl-Det/FLA-Net) and [segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch). Thanks for their valuble contributions.

## :book: Citation
- If you find this work is helpful, please cite our paper
```
@article{jiang2024towards,
  title={Towards a Benchmark for Colorectal Cancer Segmentation in Endorectal Ultrasound Videos: Dataset and Model Development},
  author={Jiang, Yuncheng and Hu, Yiwen and Zhang, Zixun and Wei, Jun and Feng, Chun-Mei and Tang, Xuemei and Wan, Xiang and Liu, Yong and Cui, Shuguang and Li, Zhen},
  journal={arXiv preprint arXiv:2408.10067},
  year={2024}
}
```

