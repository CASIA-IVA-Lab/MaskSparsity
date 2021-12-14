#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tools for training and testing a model."""

import os

import numpy as np
import pycls.core.benchmark as benchmark
import pycls.core.builders as builders
import pycls.core.checkpoint as checkpoint
import pycls.core.config as config
import pycls.core.distributed as dist
import pycls.core.logging as logging
import pycls.core.meters as meters
import pycls.core.net as net
import pycls.core.optimizer as optim
import pycls.datasets.loader as loader
import torch
from pycls.core.config import cfg


from netslim.sparse  import update_bn_by_names_,update_bn, update_bn_by_names,update_bn_by_names_dict
from netslim.prune  import get_pruning_layers
from netslim.display_BN import display_BN
from netslim.rebuild import rebuild
from netslim.flops_counter import get_model_complexity_info


logger = logging.get_logger(__name__)
#########sparse_func
update_BN =None
mask_dict = None

def setup_env():
    """Sets up environment for training or testing."""
    if dist.is_master_proc():
        # Ensure that the output dir exists
        os.makedirs(cfg.OUT_DIR, exist_ok=True)
        # Save the config
        config.dump_cfg()
    # Setup logging
    logging.setup_logging()
    # Log the config as both human readable and as a json
    logger.info("Config:\n{}".format(cfg))
    logger.info(logging.dump_log_data(cfg, "cfg"))
    # Fix the RNG seeds (see RNG comment in core/config.py for discussion)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Configure the CUDNN backend
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK


def setup_model():
    """Sets up a model for training or testing and log the results."""
    # Build the model
    model = builders.build_model()
    ############################################################################################################onnx
    #import pdb;pdb.set_trace()
    #aa = torch.load( "/data/jiangnanfei/github/pycls/checkpoint/imagenet/effient/B0_merge_noSE/checkpoints/model_epoch_0100.pyth"   )
    #aa = aa[ 'model_state'   ]
    #model.load_state_dict(aa,strict = False)
    #torch.onnx.export(model ,  torch.randn(1, 3, 224, 224, requires_grad=True), 'netB0_merge_nose_nolinear_100ep.onnx')

    #############################################################################obtain the leagaly norm_layer_names
    #prec_layers, succ_layers = get_pruning_layers(model.cpu(), [3,cfg.TEST.IM_SIZE,cfg.TEST.IM_SIZE])
    #norm_layer_names = list(set(succ_layers) & set(prec_layers))
    #model.cuda()
    #assert set(cfg.NORM_LAYER_NAMES) ==  set(norm_layer_names), 'error'
    
    global update_BN
    if cfg.SPARSE.MASK_DICT is not None:
        print("\nupdate_bn_by_layer_mask\n")
        update_BN = update_bn_by_names_dict
        global mask_dict
        mask_dict = torch.load(cfg.SPARSE.MASK_DICT)
        for i in mask_dict.keys():
            mask_dict[i] = (~mask_dict[i]).cuda()
    else:
        print("\nupdate_bn_by_layer_names\n")
        update_BN = update_bn_by_names
    #############################################################################
    logger.info("Model:\n{}".format(model))
    
    # Log model complexity
    gs_method_flops , gs_method_params  = get_model_complexity_info(   model , (3, cfg.TRAIN.IM_SIZE, cfg.TRAIN.IM_SIZE) , print_per_layer_stat=False)
    logger.info(logging.dump_log_data(net.complexity(model), "complexity"))
    # Transfer the model to the current GPU device
    if cfg.REBUILD:
        if cfg.TEST.WEIGHTS != '':
            weight_ = torch.load(cfg.TEST.WEIGHTS)
        if cfg.TRAIN.WEIGHTS != '':
            weight_ = torch.load(cfg.TRAIN.WEIGHTS)
        weight_ = weight_['model_state']
        if cfg.SCRATCH:
            model =rebuild(model,weight_,load_pruned_weights=False) 
        else: 
            model =rebuild(model,weight_)
        gs_method_flops_pruned , gs_method_params_pruned  = get_model_complexity_info(   model, (3, cfg.TRAIN.IM_SIZE,  cfg.TRAIN.IM_SIZE) , print_per_layer_stat=False)
        print('\n\nFLOPs ratio {:.2f} = {:.4f} [G] / {:.4f} [G]; Parameter ratio {:.2f} = {:.2f} [k] / {:.2f} [k].\n\n'
            .format(gs_method_flops_pruned  / gs_method_flops * 100, gs_method_flops_pruned / 10. ** 9, gs_method_flops / 10. ** 9,
                    gs_method_params_pruned / gs_method_params * 100, gs_method_params_pruned / 10. ** 3, gs_method_params / 10. ** 3))
    err_str = "Cannot use more GPU devices than available"
    model = model.cuda() 
    assert cfg.NUM_GPUS <= torch.cuda.device_count(), err_str
    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device
        )
        # Set complexity function to be module's complexity function
        model.complexity = model.module.complexity
    return model


def train_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch):
    """Performs one epoch of training."""
    # Shuffle the data
    loader.shuffle(train_loader, cur_epoch)
    # Update the learning rate
    lr = optim.get_epoch_lr(cur_epoch)
    optim.set_lr(optimizer, lr)
    # Enable training mode
    model.train()
    train_meter.iter_tic()
    global update_BN
    if cfg.SPARSE.SPARSITY is not None :
        display_BN(model,cfg.SPARSE.NORM_LAYER_NAMES,ngpu=cfg.NUM_GPUS > 1,grad =True)
    for cur_iter, (inputs, labels) in enumerate(train_loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Perform the forward pass
        preds = model(inputs)
        # Compute the loss
        loss = loss_fun(preds, labels)
        # Perform the backward pass
        optimizer.zero_grad()
        loss.backward()
        ############################################Perform the sparse
        if cfg.SPARSE.SPARSITY is not None:
            update_BN( model, cfg.SPARSE.NORM_LAYER_NAMES , lr = lr ,s = cfg.SPARSE.SPARSITY ,ngpu= cfg.NUM_GPUS > 1 , mask_dict = mask_dict , regular_method=cfg.SPARSE.REGULAR_METHOD )
        ###########################################
        # Update the parameters
        optimizer.step()
        # Compute the errors
        top1_err, top5_err = meters.topk_errors(preds, labels, [1, 5])
        # Combine the stats across the GPUs (no reduction if 1 GPU used)
        loss, top1_err, top5_err = dist.scaled_all_reduce([loss, top1_err, top5_err])
        # Copy the stats from GPU to CPU (sync point)
        loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()
        train_meter.iter_toc()
        # Update and log stats
        mb_size = inputs.size(0) * cfg.NUM_GPUS
        train_meter.update_stats(top1_err, top5_err, loss, lr, mb_size)
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

        if cfg.SPARSE.SPARSITY is not None and cur_iter % int(0.1* train_meter.epoch_iters)==0:
            display_BN(model,cfg.SPARSE.NORM_LAYER_NAMES,ngpu=cfg.NUM_GPUS > 1,grad =True)
    # Log epoch stats
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def test_epoch(test_loader, model, test_meter, cur_epoch):
    """Evaluates the model on the test set."""
    # Enable eval mode
    model.eval()
    test_meter.iter_tic()
    display_BN(model,cfg.SPARSE.NORM_LAYER_NAMES,ngpu=cfg.NUM_GPUS > 1)
    for cur_iter, (inputs, labels) in enumerate(test_loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Compute the predictions
        preds = model(inputs)
        # Compute the errors
        top1_err, top5_err = meters.topk_errors(preds, labels, [1, 5])
        # Combine the errors across the GPUs  (no reduction if 1 GPU used)
        top1_err, top5_err = dist.scaled_all_reduce([top1_err, top5_err])
        # Copy the errors from GPU to CPU (sync point)
        top1_err, top5_err = top1_err.item(), top5_err.item()
        test_meter.iter_toc()
        # Update and log stats
        test_meter.update_stats(top1_err, top5_err, inputs.size(0) * cfg.NUM_GPUS)
        test_meter.log_iter_stats(cur_epoch, cur_iter)
        test_meter.iter_tic()
    # Log epoch stats
    test_meter.log_epoch_stats(cur_epoch)
    test_meter.reset()



@torch.no_grad()
def test_epoch_jnf(test_loader, model, test_meter, cur_epoch):
    """Evaluates the model on the test set."""
    # Enable eval mode
    model.eval()
    test_meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(test_loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Compute the predictions
        preds = model(inputs)
        inputs_numpy = inputs.cpu().numpy()
        preds_numpy  = preds.cpu().numpy()
        import torch
        torch.save({'inputs_numpy':inputs_numpy , 'preds_numpy':preds_numpy},'input_and_pred.pth')
    # Log epoch stats
    test_meter.log_epoch_stats(cur_epoch)
    test_meter.reset()





def train_model():
    """Trains the model."""
    # Setup training/testing environment
    setup_env()
    # Construct the model, loss_fun, and optimizer
    model = setup_model()
    loss_fun = builders.build_loss_fun().cuda()
    optimizer = optim.construct_optimizer(model)
    # Load checkpoint or initial weights
    start_epoch = 0
    if cfg.TRAIN.AUTO_RESUME and checkpoint.has_checkpoint():
        last_checkpoint = checkpoint.get_last_checkpoint()
        checkpoint_epoch = checkpoint.load_checkpoint(last_checkpoint, model, optimizer)
        logger.info("Loaded checkpoint from: {}".format(last_checkpoint))
        start_epoch = checkpoint_epoch + 1
    elif cfg.TRAIN.WEIGHTS:
        if not cfg.SCRATCH:
            checkpoint.load_checkpoint(cfg.TRAIN.WEIGHTS, model,strict = False)
        logger.info("Loaded initial weights from: {}".format(cfg.TRAIN.WEIGHTS))
    # Create data loaders and meters
    train_loader = loader.construct_train_loader()
    test_loader = loader.construct_test_loader()
    train_meter = meters.TrainMeter(len(train_loader))
    test_meter = meters.TestMeter(len(test_loader))
    # Compute model and loader timings
    if start_epoch == 0 and cfg.PREC_TIME.NUM_ITER > 0:
        benchmark.compute_time_full(model, loss_fun, train_loader, test_loader)
    # Perform the training loop
    logger.info("Start epoch: {}".format(start_epoch + 1))
    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        # Train for one epoch
        train_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch)
        # Compute precise BN stats
        if cfg.BN.USE_PRECISE_STATS:
            net.compute_precise_bn_stats(model, train_loader)
        # Save a checkpoint
        if (cur_epoch + 1) % cfg.TRAIN.CHECKPOINT_PERIOD == 0:
            checkpoint_file = checkpoint.save_checkpoint(model, optimizer, cur_epoch)
            logger.info("Wrote checkpoint to: {}".format(checkpoint_file))
        # Evaluate the model
        next_epoch = cur_epoch + 1
        if next_epoch % cfg.TRAIN.EVAL_PERIOD == 0 or next_epoch == cfg.OPTIM.MAX_EPOCH:
            test_epoch(test_loader, model, test_meter, cur_epoch)


def test_model():
    """Evaluates a trained model."""
    # Setup training/testing environment
    setup_env()
    # Construct the model
    model = setup_model()
    # Load model weights
    checkpoint.load_checkpoint(cfg.TEST.WEIGHTS, model)
    model.eval()
    #torch.onnx.export(model.cpu(), torch.randn(32, 3,cfg.TRAIN.IM_SIZE,cfg.TRAIN.IM_SIZE, requires_grad=False), cfg.TEST.WEIGHTS+'.onnx' ,verbose= False)
    #display_BN(model,cfg.SPARSE.NORM_LAYER_NAMES,ngpu=cfg.NUM_GPUS > 1,grad =False)
    logger.info("Loaded model weights from: {}".format(cfg.TEST.WEIGHTS))
    # Create data loaders and meters
    test_loader = loader.construct_test_loader()
    test_meter = meters.TestMeter(len(test_loader))

    # Evaluate the model
    test_epoch(test_loader, model, test_meter, 0)
    #test_epoch_jnf(test_loader, model, test_meter, 0)


def inference_model():
    """Evaluates a trained model."""
    # Setup training/testing environment
    setup_env()
    # Construct the model
    model = setup_model()
    # Load model weights
    checkpoint.load_checkpoint(cfg.TEST.WEIGHTS, model)
    model.eval()
    #torch.onnx.export(model.cpu(), torch.randn(32, 3,cfg.TRAIN.IM_SIZE,cfg.TRAIN.IM_SIZE, requires_grad=False), cfg.TEST.WEIGHTS+'.onnx' ,verbose= False)
    display_BN(model,cfg.SPARSE.NORM_LAYER_NAMES,ngpu=cfg.NUM_GPUS > 1,grad =False)
    logger.info("Loaded model weights from: {}".format(cfg.TEST.WEIGHTS))
    # Create data loaders and meters
    test_loader = loader.construct_test_loader()
    test_meter = meters.TestMeter(len(test_loader))

    # Evaluate the model
    test_epoch_jnf(test_loader, model, test_meter, 0)




def time_model():
    """Times model and data loader."""
    # Setup training/testing environment
    setup_env()
    # Construct the model and loss_fun
    model = setup_model()
    loss_fun = builders.build_loss_fun().cuda()
    # Create data loaders
    train_loader = loader.construct_train_loader()
    test_loader = loader.construct_test_loader()
    # Compute model and loader timings
    benchmark.compute_time_full(model, loss_fun, train_loader, test_loader)
