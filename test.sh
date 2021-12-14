#test
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python tools/test_net.py  \
                              --cfg "configs_shiyan_prune/imagenet/resnet50/R-50-1x64d_step_8gpu_sparse.yaml" \
                              TEST.WEIGHTS $1 \
                              OUT_DIR  "test_result_out/imagenet/resnet50/" \
                              NUM_GPUS  8 \
                              TEST.BATCH_SIZE 128 \
                              REBUILD True
 
 
#"/data/jiangnanfei/github/pycls/checkpoint/imagenet/resnet50/sparsity_train/L1/0.5scale_channel.mask/lamuda_0.0007/checkpoints/model_epoch_0100.pyth"
