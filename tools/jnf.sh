#train
CUDA_VISIBLE_DEVICES=4,5,6,7  python tools/train_net.py \
                              --cfg configs/archive/imagenet/resnet/R-50-1x64d_step_4gpu.yaml  \
                              OUT_DIR checkpoint/imagenet/resnet50
#test
CUDA_VISIBLE_DEVICES=4,5,6,7  python tools/test_net.py  \
                              --cfg configs/archive/imagenet/resnet/R-50-1x64d_step_4gpu.yaml  \
                              TEST.WEIGHTS    "/data/jiangnanfei/github/pycls/checkpoint/imagenet/resnet50/checkpoints/model_epoch_0100.pyth"   \
                              OUT_DIR  test_result_out/imagenet/resnet50
#Timing
CUDA_VISIBLE_DEVICES=4,5,6,7  python tools/time_net.py  \
                              --cfg configs/archive/imagenet/resnet/R-50-1x64d_step_4gpu.yaml \
                              NUM_GPUS 4 \
                              TRAIN.BATCH_SIZE 64 \
                              TEST.BATCH_SIZE 64 \
                              PREC_TIME.WARMUP_ITER 5 \
                              PREC_TIME.NUM_ITER 50
#finetune
CUDA_VISIBLE_DEVICES=4,5,6,7  python tools/train_net.py \
                              --cfg configs/archive/imagenet/resnet/R-50-1x64d_step_4gpu.yaml \
                              TRAIN.WEIGHTS  ???
                              OUT_DIR checkpoint/imagenet/resnet50

