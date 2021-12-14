# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
#from dataset import *
from torch.autograd import Variable

#from tensorboardX import SummaryWriter



from .prune  import prune, load_pruned_model,  liu2017_normalized_by_layer
from .sparse  import update_bn,update_bn_by_names
from .graph_parser import get_norm_layer_names
import numpy as np
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False
 

def rebuild(net,new_state_dict,load_pruned_weights=True):
    weights_m = new_state_dict
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in weights_m.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net =  load_pruned_model(net, new_state_dict,load_pruned_weights=load_pruned_weights) #load_pruned_weights = scratch flag
    for k,v in net.state_dict().items():
        aa = k.split('.')
        layer_name_ = ''
        for ai in aa[:-1]:
            if is_number(ai):
                layer_name_ +=( '[' + ai + ']')
            else :
                layer_name_ +=('.'+ai)
        groups = 1
        if(torch.tensor(torch.tensor(v.shape).shape).item()) != 4:
            continue
        exec('groups = '+'net'+layer_name_+'.groups')
        if groups !=1:
            exec('net'+layer_name_+'.groups'+     '='+ 'net'+layer_name_+'.out_channels')
            exec('net'+layer_name_+'.in_channels'+'='+ 'net'+layer_name_+'.out_channels')
    return net


