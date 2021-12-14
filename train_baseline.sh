export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   python tools/train_net.py  \
                              --cfg    "configs_shiyan_prune/imagenet/resnet50/R-50-1x64d_step_8gpu.yaml"  \
                                                                                          
