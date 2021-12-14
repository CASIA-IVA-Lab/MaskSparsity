# MaskSparsity
​

This is the code repository of the paper **Pruning-aware Sparse Regularization for Network Pruning**
​

## Abstract
Structural neural network pruning aims to remove the redundant channels in the deep convolutional neural networks (CNNs) by pruning the filters of less importance to the final output accuracy. To reduce the degradation of performance after pruning, many methods utilize the loss with sparse regularization to produce structured sparsity. In this paper, we analyze these sparsity-training-based methods and find that the regularization of unpruned channels is unnecessary. Moreover, it restricts the network's capacity, which leads to under-fitting. To solve this problem, we propose a novel pruning method, named MaskSparsity, with pruning-aware sparse regularization. MaskSparsity imposes the fine-grained sparse regularization on the specific filters selected by a pruning mask, rather than all the filters of the model. Before the fine-grained sparse regularization of MaskSparity, we can use many method to get the pruning mask, such as running the global sparse regularization. MaskSparsity achieves 63.03\%-FLOPs reduction on ResNet-110 by removing 60.34\% of the parameters, with no top-1 accuracy loss on CIFAR-10. Moreover, on ILSVRC-2012, MaskSparsity reduces more than 51.07\% FLOPs on ResNet-50, with only a loss of 0.76\% in the top-1 accuracy. 


![image.png](https://cdn.nlark.com/yuque/0/2021/png/329429/1639493307825-b5680ea6-5617-42a3-bc06-986af87fbd8e.png#clientId=u82a26e47-fb94-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=215&id=u00dd1335&margin=%5Bobject%20Object%5D&name=image.png&originHeight=430&originWidth=1478&originalType=binary&ratio=1&rotation=0&showTitle=false&size=47927&status=done&style=none&taskId=u5c2bcd40-239f-4757-8731-634b45704ed&title=&width=739)


## User Guide
​

This code is based on  [pycls[](/https://github.com/facebookresearch/pycls))](https://github.com/facebookresearch/pycls](/https://github.com/facebookresearch/pycls))).
​

The following training, finetuning, pruning stages are achieved via modifying the correspoding parameters of mmdet/api/train.py.


### Installation
### Prepare the pytorch evoriments.
​

```shell
conda create -n pycls python=3.7 -y
conda activate pycls

# geting CUDA-10 or 9
# for CUDA 9.0
conda install pytorch==1.3.1 torchvision==0.4.2 cudatoolkit=9.0 -c pytorch

# for CUDA 10.0
conda install pytorch==1.3.1 torchvision==0.4.2 cudatoolkit=10.0 -c pytorch

pip install -r requirements.txt
python setup.py develop --user
```
​

### Prepare the ImageNet dataset
```shell
ln -s path  pycls/datasets/data/imagenet
```
​

### Traning and Pruning
​

### 训练baseline
```
./train_baseline.sh
```
​

### 稀疏训练
​

```
./sparsity.sh    ../path_to_weight   sparsity_facotr   
./sparsity.sh  "paper/imagenet_uniform/baseline/weight/model_epoch_0100.pyth"  5e-4  
```
### 剪枝获得mask
​

```
./prune.sh  ../path_to_weight  pruning_threshold
./prune.sh  "/data/jiangnanfei/github/AAAI/pycls/paper/imagenet_uniform/prune_ratio_0.5/model_epoch_0100.pyth"  0.01
```
​

### maskL1稀疏训练
​

```
./mask_sparsity.sh    ../path_to_weight   sparsity_facotr   ../path_to_maskweight
./mask_sparsity.sh  "paper/imagenet_uniform/baseline/weight/model_epoch_0100.pyth"  5e-4  "paper/imagenet_uniform/prune_ratio_0.5/0.5_uniform.mask"
```
​

### 剪枝
​

```
./prune.sh  ../path_to_weight  pruning_threshold
./prune.sh  "/data/jiangnanfei/github/AAAI/pycls/paper/imagenet_uniform/prune_ratio_0.5/model_epoch_0100.pyth"  0.01
```
​

### finetune
​

```
./finetune.sh  ./path_to_weight  lr
./finetune.sh  "paper/imagenet_uniform/prune_ratio_0.5/prune/th-p0.01_model.pt"  5e-4
```
​

### scratch
​

```
./scratch.sh  ./path_to_weight  lr
./scratch.sh  "paper/imagenet_uniform/prune_ratio_0.5/prune/th-p0.01_model.pt"  5e-4
```
### 数据集与权重文件
实验权重文件下载 [https://pan.baidu.com/s/1OP-tF94DVN0xKvNFCZ1HhQ](https://pan.baidu.com/s/1OP-tF94DVN0xKvNFCZ1HhQ)   
​

## Citation
```latex
@inproceedings{jiang2021,
  title = {Pruning-aware Sparse Regularization for Network Pruning},
  author = {Nanfei Jiang and Xu Zhao and Chaoyang Zhao and Ming Tang and Jinqiao Wang},
  booktitle={Arxiv},
  year={2021}
}
```
