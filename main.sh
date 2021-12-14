#################################################################
#Prepare data set
ln -s path  pycls/datasets/data/imagenet



#####################################################################
#set environment
conda create -n pycls python=3.7 -y
conda activate pycls

#cuda环境
cuda-10 or 9
# CUDA 9.0
conda install pytorch==1.3.1 torchvision==0.4.2 cudatoolkit=9.0 -c pytorch
# CUDA 10.0
conda install pytorch==1.3.1 torchvision==0.4.2 cudatoolkit=10.0 -c pytorch

pip install -r requirements.txt
python setup.py develop --user
###############################################################

export PYTHONPATH=.
#train baseline
./train_baseline.sh

#mask sparsity
nohup mask_sparsity.sh  > mask_sparsity.out &

#prune
./prune.sh

#finetune
nohup finetune.sh > finetune.out &


#test
#$1 is the finetuned.pth
./test.sh  $1



