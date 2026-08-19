# SSPC-Net-HG

Official PyTorch implementation of **SSPC-Net: Semi-supervised Semantic 3D Point Cloud Segmentation Network** (AAAI 2021) with hierarchical grouping, by Mingmei Cheng, Le Hui, Jin Xie and Jian Yang. Details are in the [paper](https://arxiv.org/abs/2104.07861).

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
