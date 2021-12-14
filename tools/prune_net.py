#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Train a classification model."""

import pycls.core.config as config
import pycls.core.distributed as dist
import pycls.core.trainer as trainer
from pycls.core.config import cfg
import os

import numpy as np
import pycls.core.benchmark as benchmark
import pycls.core.builders as builders
import pycls.core.checkpoint as checkpoint
import pycls.core.distributed as dist
import pycls.core.logging as logging
import pycls.core.meters as meters
import pycls.core.net as net
import pycls.core.optimizer as optim
import pycls.datasets.loader as loader
import torch
from netslim.display_BN import display_BN
from netslim.prune  import prune, load_pruned_model
from netslim.rebuild import rebuild

def main():
    config.load_cfg_fom_args("Train a classification model.")
    config.assert_and_infer_cfg()
    cfg.freeze()
    # Ensure that the output dir exists
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    # Save the config
    config.dump_cfg()
    # Build the model
    model = builders.build_model() 
    # Load model weights

    if cfg.REBUILD:
        if cfg.TEST.WEIGHTS != '':
            weight_ = torch.load(cfg.TEST.WEIGHTS)
        if cfg.TRAIN.WEIGHTS != '':
            weight_ = torch.load(cfg.TRAIN.WEIGHTS)
        weight_ = weight_['model_state']
        model =rebuild(model,weight_)
    else:
        checkpoint.load_checkpoint(cfg.TRAIN.WEIGHTS, model)
    weight_file = torch.load(cfg.TRAIN.WEIGHTS)
    #######prune
    import pdb;pdb.set_trace()
    if cfg.PRUNE.USE_MASK is not None:
        pruning_mask = torch.load(cfg.PRUNE.USE_MASK)
        pruned_mask , pruned_model_weights = prune(model=model ,
                                                   input_shape =  (3,cfg.TRAIN.IM_SIZE,cfg.TRAIN.IM_SIZE) ,
                                                   prune_ratio = cfg.PRUNE.FACTOR,
                                                   get_mask = cfg.PRUNE.GET_MASK ,
                                                   pruning_mask = pruning_mask,
                                                   min_channels = 0 ,
                                                   Threshold_method  = cfg.PRUNE.THRESHOLD_METHOD ,
                                                   super_norm_layers = cfg.PRUNE.SPUER_NORM_LAYERS ,
                                                   super_succ_layers = cfg.PRUNE.SPUER_SUCC_LAYERS ,
                                                   super_prec_layers = cfg.PRUNE.SPUER_PREC_LAYERS   )
    elif cfg.PRUNE.PRUNE_RESIDUAL: 
        if cfg.PRUNE.MIN_CHANNELS  ==0:
            pruned_mask , pruned_model_weights = prune(model=model , 
                                                   input_shape =  (3,cfg.TRAIN.IM_SIZE,cfg.TRAIN.IM_SIZE) ,
                                                   prune_ratio = cfg.PRUNE.FACTOR, 
                                                   get_mask = cfg.PRUNE.GET_MASK , 
                                                   min_channels = 0 ,
                                                   Threshold_method  = cfg.PRUNE.THRESHOLD_METHOD ,
                                                   super_norm_layers = cfg.PRUNE.SPUER_NORM_LAYERS ,
                                                   super_succ_layers = cfg.PRUNE.SPUER_SUCC_LAYERS ,
                                                   super_prec_layers = cfg.PRUNE.SPUER_PREC_LAYERS   )
        else:
            pruned_mask , pruned_model         = prune(model=model , 
                                                   input_shape = (3,cfg.TRAIN.IM_SIZE,cfg.TRAIN.IM_SIZE) ,  
                                                   prune_ratio = cfg.PRUNE.FACTOR, 
                                                   get_mask = cfg.PRUNE.GET_MASK , 
                                                   min_channels =1 ,
                                                   Threshold_method  = cfg.PRUNE.THRESHOLD_METHOD ,
                                                   super_norm_layers = cfg.PRUNE.SPUER_NORM_LAYERS ,
                                                   super_succ_layers = cfg.PRUNE.SPUER_SUCC_LAYERS ,
                                                   super_prec_layers = cfg.PRUNE.SPUER_PREC_LAYERS  )
            pruned_model_weights = pruned_model.state_dict()
    else:
        if cfg.PRUNE.MIN_CHANNELS  ==0:
            pruned_mask , pruned_model_weights = prune(model=model ,
                                                   input_shape =  (3,cfg.TRAIN.IM_SIZE,cfg.TRAIN.IM_SIZE) ,
                                                   prune_ratio = cfg.PRUNE.FACTOR,
                                                   get_mask = cf.PRUNE.GET_MASK ,
                                                   min_channels = 0 ,
                                                   Threshold_method  = cfg.PRUNE.THRESHOLD_METHOD )
        else:
            pruned_mask , pruned_model         = prune(model=model ,
                                                   input_shape = (3,cfg.TRAIN.IM_SIZE,cfg.TRAIN.IM_SIZE) ,
                                                   prune_ratio = cfg.PRUNE.FACTOR,
                                                   get_mask = cfg.PRUNE.GET_MASK ,
                                                   min_channels =1 ,
                                                   Threshold_method  = cfg.PRUNE.THRESHOLD_METHOD )
            pruned_model_weights = pruned_model.state_dict()

   
    if  cfg.PRUNE.GET_MASK :
        torch.save(   pruned_mask ,   os.path.join( cfg.OUT_DIR , cfg.PRUNE.THRESHOLD_METHOD +"-p"+  str(cfg.PRUNE.FACTOR) + "_mask.pth")  )
    weight_file['model_state'] = pruned_model_weights
    torch.save(weight_file , os.path.join( cfg.OUT_DIR , cfg.PRUNE.THRESHOLD_METHOD +"-p"+  str(cfg.PRUNE.FACTOR) + "_model.pt")  )
    

if __name__ == "__main__":
    main()
