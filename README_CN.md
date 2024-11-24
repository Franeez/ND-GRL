# ND-GRL (Decycling of networks with optimized deep reinforcement learning)

这是使用了TensorFlow1.14实现的ND-GRL的相关工程代码

相关实现参考了FINDER ([Finding key players in complex networks through deep reinforcement learning](https://www.nature.com/articles/s42256-020-0177-2.epdf?sharing_token=0CAxnrCP1THxBEtK2mS5c9RgN0jAjWel9jnR3ZoTv0O3ej6g4eVo3V4pnngJO-QMH375GbplyUstNSGUaq-zMyAnpSrZIOiiDvB0V_CqsCipIfCq-enY3sK3Uv_D_4b4aRn6lYXd8HEinWjLNM42tQZ0iVjeMBl6ZRA7D7WUBjM%3D))
，对应的[工程链接](https://doi.org/10.24433/CO.3005605.v1)
，比如一些基础工具类：节点类、工具类等

Fengyu Zhang, Tianqi Li and Wei Chen. [Decycling of networks with optimized deep reinforcement learning]

## Contents

- [Repo Contents](#repo-contents)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Reproduction instructions](#reproduction-instructions)
- [Basebline methods implementation](#basebline-methods-implementation)


# Repo Contents

- [code](./code/FINDER_CN): 代码
- [data](./code/data): 数据，包含人工合成网络和真实网络
- [results](./code/results): 文章中所提及的实验结果
- [drawing](./code/drawing): 结果图绘制


# System Requirements

## Software dependencies and operating systems

### Software dependencies

主要的python包:
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
实验环境可以在Linux(推荐)或Windows上进行，但请注意不同环境对应的pip包的要求，我们使用了如下环境:

Linux: Ubuntu 18.04  
Windows: 10

注：需要 `gcc` version >= 7.4.0

## Hardware Requirements
参考README.md英文版

注：我们的一些实验是在北京航空航天大学的高算平台上进行的

# Installation Guide

## Instructions
参考README.md英文版
主要包含两步：
1. 使用requirements.txt安装所需的第三方python库
```
pip install -r requirements.txt
```
2. 使用setup.py编译（使用c++扩展）
```
python setup.py build_ext -i
```
# Reproduction instructions

## Instructions to run
参考README.md英文版
1. 模型训练
```
CUDA_VISIBLE_DEVICES=gpu_id python train.py
```
在 `FINDER.pyx` 文件中修改模型的配置参数, 并重新进行`setup.py`编译.

2. 人工合成网络数据集测试
```
CUDA_VISIBLE_DEVICES=-1 python testSynthetic.py (do not use GPU for test)
```

3. 真实网络数据集测试
```
CUDA_VISIBLE_DEVICES=-1 python testReal.py (do not use GPU for test)
```


## Expected output
实验结果保存在 `results` 文件夹

# Basebline methods implementation
我们使用的对比方法： HDA, HBA, HCA, HPRA, greedy algorithm, Tarjan 和 FINDER

其中 HDA, HBA, HCA, 和 HPRA 使用了Networkx 2.0封装的函数方法实现


