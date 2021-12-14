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



from netslim.sparse  import update_bn_by_names_,update_bn, update_bn_by_names,update_bn_by_names_dict
from netslim.prune  import get_pruning_layers
from netslim.display_BN import display_BN
from netslim.rebuild import rebuild
from netslim.flops_counter import get_model_complexity_info



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
    model.eval()
    gs_method_flops , gs_method_params  = get_model_complexity_info(   model , (3, 32, 32) , print_per_layer_stat=False)
    
    # Load model weights
    import pdb;pdb.set_trace()
    checkpoint.load_checkpoint(cfg.TRAIN.WEIGHTS, model)
    weight_file = torch.load(cfg.TRAIN.WEIGHTS)

    #################################################getmask
    channelnumbers = torch.load( "/data/jiangnanfei/github/pycls/medium_save_channelnumbers.pth")
    
    
    for jj in [3]:#[0,1,2,3,4,5,6,7,8,9]:

        pruning_mask = {}
        for i in cfg.SPARSE.NORM_LAYER_NAMES :
            number  = len(weight_file['model_state'][i+".weight"] )
            prune_num =   int(channelnumbers[i][jj].item())
            print( i, channelnumbers[i] , number    )
            pruning_mask[i] =    torch.tensor( prune_num*[True] + (number - prune_num)*[False]  ).bool().cpu()
            #pruning_mask[i] =    torch.tensor( prune_num*[False] + (number - prune_num)*[True]  ).bool().cpu()#org
        import copy
        net = copy.deepcopy(model) 
        pruned_mask , pruned_model = prune(model=net ,
                                                   input_shape =  (3,cfg.TRAIN.IM_SIZE,cfg.TRAIN.IM_SIZE) ,
                                                   prune_ratio = cfg.PRUNE.FACTOR,
                                                   get_mask = cfg.PRUNE.GET_MASK ,
                                                   pruning_mask = pruning_mask,
                                                   min_channels = 1 ,
                                                   Threshold_method  = cfg.PRUNE.THRESHOLD_METHOD ,
                                                   super_norm_layers = cfg.PRUNE.SPUER_NORM_LAYERS ,
                                                   super_succ_layers = cfg.PRUNE.SPUER_SUCC_LAYERS ,
                                                   super_prec_layers = cfg.PRUNE.SPUER_PREC_LAYERS   )

        gs_method_flops_pruned , gs_method_params_pruned  = get_model_complexity_info(   pruned_model , (3, 32, 32) , print_per_layer_stat=False)

       
        print('FLOPs ratio {:.2f} = {:.4f} [G] / {:.4f} [G]; Parameter ratio {:.2f} = {:.2f} [k] / {:.2f} [k].'
          .format(gs_method_flops_pruned  / gs_method_flops * 100, gs_method_flops_pruned / 10. ** 9, gs_method_flops / 10. ** 9,
                  gs_method_params_pruned / gs_method_params * 100, gs_method_params_pruned / 10. ** 3, gs_method_params / 10. ** 3)) 

        import pdb;pdb.set_trace()
        flops_ = gs_method_flops_pruned  / gs_method_flops * 100
        print(flops_)
        if flops_ < 46:
            del net
            continue
        else :
            print("\n\n\n\nprune:",flops_)
            break




    import pdb;pdb.set_trace() 
    if  cfg.PRUNE.GET_MASK :
        torch.save(   pruned_mask ,   os.path.join( cfg.OUT_DIR , 'pca' , cfg.PRUNE.THRESHOLD_METHOD +"-p"+  str(cfg.PRUNE.FACTOR) + "_mask.pth")  )
    weight_file['model_state'] = pruned_model.state_dict()
    torch.save(weight_file , os.path.join( cfg.OUT_DIR ,'pca' , cfg.PRUNE.THRESHOLD_METHOD +"-p"+  str(cfg.PRUNE.FACTOR) + "_model.pt")  )
    

if __name__ == "__main__":
    main()
