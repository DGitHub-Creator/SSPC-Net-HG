# SSPC-Net-HG

This repository is a **reproduction and dataset-adapted version** of **SSPC-Net: Semi-supervised Semantic 3D Point Cloud Segmentation Network** (AAAI 2021, by Mingmei Cheng, Le Hui, Jin Xie and Jian Yang), modified for the custom ODPT-HG dataset. It is NOT the official code released by the paper authors. Details of the original method are in the [paper](https://arxiv.org/abs/2104.07861).

> 说明：本仓库是基于原论文方法、为适配自定义 ODPT-HG 数据集而修改后的复现与适配版本，并非论文作者发布的官方代码。

## Project Overview

- Semi-supervised semantic segmentation network for 3D point clouds
- Trained and evaluated on the S3DIS dataset
- Superpoint generation can refer to [SPGraph](https://github.com/loicland/superpoint_graph/tree/release)

## Requirements

- Ubuntu 18.04
- Python packages:

  ```
  ./Anaconda3-5.1.0-Linux-x86_64.sh
  ```

- PyTorch:

  ```
  conda install pytorch==1.4.0
  ```

## Usage

- **Dataset**: details of pseudo label generation and data processing will be updated later.

- **Train**:

  ```
  sh semantic_seg/train_s3dis.sh DATASET_S3DIS_PATH
  ```

## Citation

```
@article{cheng2021sspc,
title={SSPC-Net: Semi-supervised Semantic 3D Point Cloud Segmentation Network},
author={Cheng, Mingmei and Hui, Le and Xie, Jin and Yang, Jian},
booktitle={AAAI},
year={2021}
}
```

## Acknowledgement

Our code refers to [SPGraph](https://github.com/loicland/superpoint_graph/tree/release).
