export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=4  python tools/prune_net.py \
                              --cfg configs_shiyan_prune/imagenet/resnet50/R-50-1x64d_step_1gpu_sparse.yaml   \
                              TRAIN.WEIGHTS  $1 \
                              PRUNE.PRUNE_RESIDUAL True \
                              PRUNE.GET_MASK  True \
                              PRUNE.FACTOR $2 \
                              REBUILD True
                              #PRUNE.USE_MASK "/data/jiangnanfei/github/pycls/checkpoint/imagenet/resnet50/prune_model/pca/new_0.47mask.pth"\

# "/data/jiangnanfei/github/AAAI/pycls/paper/imagenet_uniform/prune_ratio_0.5/model_epoch_0100.pyth"
# 0.01



