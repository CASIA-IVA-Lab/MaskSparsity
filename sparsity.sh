#mask_sparsity
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   python tools/train_net.py  \
                              --cfg    "configs_shiyan_prune/imagenet/resnet50/R-50-1x64d_step_8gpu_sparse.yaml"  \
                              TRAIN.WEIGHTS   $1 \
                              SPARSE.SPARSITY  $2 \
                              SPARSE.REGULAR_METHOD L1 \
                              REBUILD True \

# "paper/imagenet_uniform/baseline/weight/model_epoch_0100.pyth" \
# 5e-4 \
                                                                                          
