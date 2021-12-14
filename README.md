# pycls-MaskL1





### pycls 剪枝
对 pycls (链接：[https://github.com/facebookresearch/pycls](/https://github.com/facebookresearch/pycls))进行剪枝。
training , finetuning , pruning 等阶段都需要修改mmdet/api/train.py中相关参数


### 准备Imagenet数据集
```
ln -s path  pycls/datasets/data/imagenet
```


### 准备pycls与剪枝环境
```
conda create -n pycls python=3.7 -y
conda activate pycls

cuda环境
cuda-10 or 9

CUDA 9.0
conda install pytorch==1.3.1 torchvision==0.4.2 cudatoolkit=9.0 -c pytorch

CUDA 10.0
conda install pytorch==1.3.1 torchvision==0.4.2 cudatoolkit=10.0 -c pytorch

pip install -r requirements.txt
python setup.py develop --user
```




### 训练baseline
```
./train_baseline.sh
```

### 稀疏训练

```
./sparsity.sh    ../path_to_weight   sparsity_facotr   
./sparsity.sh  "paper/imagenet_uniform/baseline/weight/model_epoch_0100.pyth"  5e-4  
```
### 剪枝获得mask

```
./prune.sh  ../path_to_weight  pruning_threshold
./prune.sh  "/data/jiangnanfei/github/AAAI/pycls/paper/imagenet_uniform/prune_ratio_0.5/model_epoch_0100.pyth"  0.01
```

### maskL1稀疏训练

```
./mask_sparsity.sh    ../path_to_weight   sparsity_facotr   ../path_to_maskweight
./mask_sparsity.sh  "paper/imagenet_uniform/baseline/weight/model_epoch_0100.pyth"  5e-4  "paper/imagenet_uniform/prune_ratio_0.5/0.5_uniform.mask"
```

### 剪枝

```
./prune.sh  ../path_to_weight  pruning_threshold
./prune.sh  "/data/jiangnanfei/github/AAAI/pycls/paper/imagenet_uniform/prune_ratio_0.5/model_epoch_0100.pyth"  0.01
```

### finetune

```
./finetune.sh  ./path_to_weight  lr
./finetune.sh  "paper/imagenet_uniform/prune_ratio_0.5/prune/th-p0.01_model.pt"  5e-4
```

### scratch

```
./scratch.sh  ./path_to_weight  lr
./scratch.sh  "paper/imagenet_uniform/prune_ratio_0.5/prune/th-p0.01_model.pt"  5e-4
```
### 数据集与权重文件
实验权重文件下载 https://pan.baidu.com/s/1OP-tF94DVN0xKvNFCZ1HhQ   

