# ND-GRL (Decycling of networks with optimized deep reinforcement learning)

This is a TensorFlow implementation of ND-GRL, as described in our paper:

Fengyu Zhang, Tianqi Li and Wei Chen. [Decycling of networks with optimized deep reinforcement learning]

## Contents

- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Reproduction instructions](#reproduction-instructions)
- [Basebline methods implementation](#basebline-methods-implementation)

# Overview

Decycling of networks is a long-standing problem in network science with a wide range of applications from vaccination and epidemic control to information spreading and viral marketing. However, it remains an outstanding challenge in pushing for the limit of low computational cost while maintaining accuracy in predicting the seed nodes. In this paper, we develop a two-stage framework called optimized deep reinforcement learning to deal with this challenge. The first stage based on generic algorithm enables exploiting the solution space effectively while maintaining high efficiency of training process. The second stage combines the improved Deep Q-learning algorithm with Tarjan algorithm so that the seed nodes are selected through the Markov decision process. Our algorithm significantly outperforms existing methods on four synthetic networks and eight real networks in terms of prediction accuracy. We also show that our algorithm achieves several orders of magnitude faster than existing methods in large scale real-world networks.

# Repo Contents

- [code](./code/FINDER_CN): code introduced in the paper refers to some basic classes of FINDER ([Finding key players in complex networks through deep reinforcement learning](https://www.nature.com/articles/s42256-020-0177-2.epdf?sharing_token=0CAxnrCP1THxBEtK2mS5c9RgN0jAjWel9jnR3ZoTv0O3ej6g4eVo3V4pnngJO-QMH375GbplyUstNSGUaq-zMyAnpSrZIOiiDvB0V_CqsCipIfCq-enY3sK3Uv_D_4b4aRn6lYXd8HEinWjLNM42tQZ0iVjeMBl6ZRA7D7WUBjM%3D)), such as node classes and utility classes
     - [Related links](https://doi.org/10.24433/CO.3005605.v1)
- [data](./code/data): relevant experimental data, including artificial synthetic networks and real networks.
- [results](./code/results): results obtained by ND-GRL, which should be the same as are reported in the paper.
- [drawing](./code/drawing): script for drawing the result graph.

# System Requirements

## Software dependencies and operating systems

### Software dependencies

Users should install the following packages first, which will install in about 5 minutes on a machine with the recommended specs. The versions of software are, specifically:
```
cython==0.29.13 
networkx==2.3 
numpy==1.17.3 
pandas==0.25.2 
scipy==1.3.1 
tensorflow-gpu==1.14.0 
tqdm==4.36.1
```

### Operating systems
The package development version is tested on *Linux and Windows 10* operating systems. The developmental version of the package has been tested on the following systems:

Linux: Ubuntu 18.04  
Windows: 10

The pip package should be compatible with Windows, and Linux operating systems.

Before setting up the ND-GRL users should have `gcc` version 7.4.0 or higher.

## Hardware Requirements
The `ND-GRL` model requires a standard computer with enough RAM and GPU to support the operations defined by a user. For minimal performance, this will be a computer with about 4 GB of RAM and 16GB of GPU. For optimal performance, we recommend a computer with the following specs:

RAM: 16+ GB  
CPU: 4+ cores, 3.3+ GHz/core
GPU: 16+ GB

The runtimes below are generated using a computer with the recommended specs (16 GB RAM, 4 cores@3.3 GHz) and internet of speed 25 Mbps.

Note: Some of our experiments are conducted on the advanced computing platform of Beihang University !

# Installation Guide

## Instructions
1. First install all the above required packages, which are contained in the requirements.txt file
```
pip install -r requirements.txt
```
2. Make all the file
```
python setup.py build_ext -i
```

## Typical install time
It took about 5 mins to install all the required packages, and about 1 mins to make all the files.

# Reproduction instructions

## Instructions to run
1. Train the model, 
```
CUDA_VISIBLE_DEVICES=gpu_id python train.py
```
Modify the hyper-parameters in `FINDER.pyx` to tune the model, and make files after the the modification.

2. Test synthetic data,
```
CUDA_VISIBLE_DEVICES=-1 python testSynthetic.py (do not use GPU for test)
```
Using the well-trained model (stored in `./models`), you can obtain the results reported in the paper.

3. Test real data,
```
CUDA_VISIBLE_DEVICES=-1 python testReal.py (do not use GPU for test)
```
Using the well-trained model (stored in `./models`), you can obtain the results reported in the paper.


## Expected output
The experimental results are saved in the `results` folder.

# Basebline methods implementation
We compared with HDA, HBA, HCA, HPRA, greedy algorithm, Tarjan and FINDER.

We ran HDA, HBA, HCA, and HPRA with Networkx 2.0.



