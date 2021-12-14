#scratch
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=4,5,6,7  python tools/train_net.py  \
                              --cfg    configs_shiyan_prune/imagenet/resnet50/R-50-1x64d_step_4gpu_sparse.yaml   \
                              TRAIN.WEIGHTS  paper/imagenet_uniform/prune_ratio_0.5/prune/th-p0.01_model.pt \
                              REBUILD True \
                              SCRATCH  True

 
