# Skip-WGANomaly

Skip-WGANomaly: Unsupervised Image Anomaly Detection Based in WGAN

This repository contains the framework for implementation and training of Skip-WGANomaly jointly with the implementation of Skip-GANomaly for comparison and testing purposes. 

## 1. Contents
- [Skip-WGANomaly](#skip-wganomaly)
  - [1. Contents](#1-contents)
  - [2. Requirements](#2-requirements)
  - [3. Execution](#4-training)
    - [3.1. Training in CIFAR-10](#31-training-on-cifar10)
    - [3.2. Tuning test on CIFAR-10](#32-tuning-test-on-cifar10)
    - [3.3. Train on Custom Dataset](#33-train-on-custom-dataset)
    - [3.4. Visualization and evaluation](#34-visualization-and-evaluation)
  - [4. Results](#4-results)
    


## 2. Requirements
The file `environment.txt`includes the installed packages of the environment used for the project. The model was implemented using PyTorch (v1.11, Python 3.8.10 and CUDA 11.6). Experiments of the paper were performed using an NVIDIA GeForce RTX 2080 GPU.
  
## 3. Execution

### 3.1. Training on CIFAR10
Example: training all cifar classes as anomaly in w_skipganomaly

``` shell
# CIFAR
bash experiments/run_cifar_wskip.sh
```
Example: run a parametrized training for airplane as anomaly in Skip-WGANomaly
```
python train.py --dataset cifar10 --abnormal_class 'airplane' --niter 10 --display --w_con 50 --w_lat 1 --w_adv 1 --model "w_skipganomaly"
```
All arguments are detailed in `options.py`. For displaying all possible arguments, run: 
```
python train.py -h
```

### 3.2. Tuning test on CIFAR10

Example: run a RayTune test on CIFAR-10 with Skip-WGANomaly. 

```
python train_tune.py --dataset cifar10 --abnormal_class 'bird' --tune True  --display --model "w_skipganomaly"
```
Parameters to be tested for tuning have to be edited and personalized inside `train_tune.py`.
### 3.3. Train on Custom Dataset
To train the model on a custom dataset, the dataset should be copied into `./data` directory, and should have the following directory & file structure:

```
Custom Dataset
├── test
│   ├── 0.normal
│   │   └── normal_tst_img_0.png
│   │   └── normal_tst_img_1.png
│   │   ...
│   │   └── normal_tst_img_n.png
│   ├── 1.abnormal
│   │   └── abnormal_tst_img_0.png
│   │   └── abnormal_tst_img_1.png
│   │   ...
│   │   └── abnormal_tst_img_m.png
├── train
│   ├── 0.normal
│   │   └── normal_tst_img_0.png
│   │   └── normal_tst_img_1.png
│   │   ...
│   │   └── normal_tst_img_t.png

```

Then model training is the same as the training explained above.

```
python train.py                     \
    --dataset <name-of-the-data>    \
    --isize <image-size>            \
    --niter <number-of-epochs>      \
    --display                       # optional if you want to visualize
```

### 3.4. Visualization and evaluation
In the file `histogram.csv`, the  labels (normal, abnormal) and anomaly scores of the last experiment are saved and can be retrieved for plotting.

When `--display` enabled, use Visdom for metrics and images visualization. Run in a different window:
```
python -m visdom.server
```
By default, port 8097 of `localhost` will hold the service. 

## 4. Results
The results of Skip-WGANomaly for CIFAR-10 obtained by us, with max 15 epochs:

| Model                                    | auto  | airplane | horse | bird  | deer  | frog  | cat   | truck | ship  | dog   |
|------------------------------------------|-------|----------|-------|-------|-------|-------|-------|-------|-------|-------|
| **Skip-WGANomaly**                       | 0.901 |   0.998  | 0.686 | 0.660 | 0.920 | 0.983 | 0.709 | 0.856 | 0.971 | 0.700 |
| **Skip-WGANomaly (no skip connections)** | 0.836 | 0.921    | 0.682 | 0.466 | 0.644 | 0.839 | 0.642 | 0.821 | 0.824 | 0.680 |
