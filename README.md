# S5: Self-Supervised Semantic Segmentation with Suggestions

## Introduction

S5 (Self-Supervised Semantic Segmentation with Suggestions) is a model-agnostic framework for self-supervised training of interactive semantic segmentation task, which utilizes general semantic segmentation datasets for generating and integrating suggestions by performing certain augmentations to the known targets in semantic segmentation datasets, without the need of using human annotations.

Based on the S5 framework, we elaborated several empirical augmentations such as the block-based method and flood-based method, which are convenient for real-world applications.

We tested our S5 framework along with the augmentations using DeepLabV3+HRNet on binarized Cityscapes dataset, comparing them to traditional click-based or extreme-point-based (box-based) methods.

We demonstrated that our method boosts the performance of semantic segmentation models remarkably, consistently and conveniently, in a generalizable fashion.

## Runthrough Instruction

0. Clone this github repo and install dependencies (mainly `torch`, `timm` and packages used in `utils.py`).
1. Prepare [Cityscapes](https://www.cityscapes-dataset.com/) dataset under folder `data/` according to `data/README.md`.
2. Modify & Run `example.sh` to train model (or user  `ddp_launch.sh` with training parameters, to achieve distributed data parallel training). The trained model will be stored in sub-folders created automatically under `models/` folder, named by the experiment name assigned by parameter `--exp`.
3. The training & validation results and visualizations are stored in `models/<experiment_name>/stats/` and are updated on the fly during training.
4. For output visualization, run `visualize_cityscapes.py` whenever the `best.pth` model is saved in `models/<experiment_name>/`.
5. For more advanced usage, please refer to the main program `train_cityscapes.py` for tunable arguments.
6. To use custom suggestion generation methods or modify the default parameters of implemented suggestion generation methods, please refer to `transforms_vision.py`.