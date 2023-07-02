# Lifelong AUC Maximization
This is the official implementation of the UAI 2023 paper [AUC Maximization in Imbalanced Lifelong Learning
](https://openreview.net/pdf?id=X7-y2_vjvk)

## Abstract

We investigate the problem of optimizing the Area Under the Curve (AUC) in the continual learning setting with an imbalanced data stream. Current research on AUC optimization focus on learning on a single task. For applications in the online advertisement or satellite imagery, it is often the case that the tasks are arriving sequentially and the current AUC optimization methods may suffer from the problem of ``catastrophic forgetting". In this paper, we propose a method to optimize the AUC continuously such that the model can retain the performance on all the tasks after training, called Lifelong AUC (L-AUC). L-AUC is built upon memory-based lifelong learning. The imbalanced data stream poses severe challenges for commonly-used gradient surgery approaches (e.g., GEM, A-GEM, etc.). We address this issue by maintaining two models simultaneously: one focuses on learning the current knowledge while the other concentrates on reviewing previously-learned knowledge. The two models gradually align during training. 

## Requirements
PyTorch >= v1.6.0. The code is based on Improved Schemes for Episodic Memory-based Lifelong Learning and
Gradient Episodic Memory for Continual Learning 

## Prepare data
````
sh prepare_data.sh
````


to replicate the results of the paper on a particular dataset (downloading and propocessing the CIFAR100, CUB and AWA ISIC2019 and EuroSat datasets):
   
## Training & Evaluation
````
sh run_auc.sh
````
Note: to run on the different datasets, please modify the corresponding name (CIFAR_100i, CUB200, AWA2, ISIC_SPLIT, EuroSat_SPLIT) of dataset in the run_auc.sh

## Citation
If you find this repo helpful, please cite the following paper:

````
@inproceedings{
zhu2023auc,
title={{AUC} Maximization in Imbalanced Lifelong Learning},
author={Xiangyu Zhu and Jie Hao and Yunhui Guo and Mingrui Liu},
booktitle={The 39th Conference on Uncertainty in Artificial Intelligence},
year={2023},
url={https://openreview.net/forum?id=X7-y2_vjvk}
}
