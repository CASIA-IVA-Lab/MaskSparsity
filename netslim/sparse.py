import torch
import torch.nn as nn
from .prune import get_pruning_layers

def update_bn(model, norm_layer_names=None ,s1=1e-4 ,ngpu=False ,mask_dict =None):
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) or \
            isinstance(m, nn.modules.instancenorm._InstanceNorm) or \
            isinstance(m, nn.GroupNorm):
            m.weight.grad.data.add_(s1 * torch.sign(m.weight.data))

def update_bn_by_names(model, norm_layer_names, lr=None , s=1e-4, ngpu=False, mask_dict = None,regular_method='L1'):
    for norm_layer_name in norm_layer_names:
        *container_names, module_name = norm_layer_name.split('.')
        if ngpu :
            container_names.insert(0,'module')
        container = model
        for container_name in container_names:
            container = container._modules[container_name]
        m = container._modules[module_name]
        if regular_method=='L1':
            add_grad = s * torch.sign(m.weight.data) # L1
        elif regular_method=='L2':
            add_grad = s * m.weight.data # L2
        elif regular_method=='shiyan':
            #import pdb;pdb.set_trace()
            mask = m.weight.data.abs()  <=  9e-2
            add_grad = s * m.weight.data
            add_grad = add_grad.mul(mask)

        m.weight.grad.data.add_( add_grad  )


def update_bn_by_names_(model, norm_layer_names, s=1e-4, ngpu=False, mask_dict = None):
    for norm_layer_name in norm_layer_names:
        *container_names, module_name = norm_layer_name.split('.')
        if ngpu :
            container_names.insert(0,'module')
        container = model
        for container_name in container_names:
            container = container._modules[container_name]
        m = container._modules[module_name]
############################################################
        #import  pdb;pdb.set_trace()
        mask = m.weight.data.abs()  <=  9e-2  ########
        add_grad = s * m.weight.data
        add_grad = add_grad.mul(mask)
        m.weight.grad.data.add_( add_grad )  ##
#############################################################
        #m.weight.grad.data.add_(s * torch.sign(m.weight.data)  )  ##L1
        #m.weight.grad.data.add_(s * m.weight.data )  ##L2
      

###########L0.5 useless
#def update_bn_by_names_L05 (model, norm_layer_names, s=1e-4,ngpu=False , mask_dict =None ):
#    for norm_layer_name in norm_layer_names:
#        *container_names, module_name = norm_layer_name.split('.')
#        if ngpu :
#            container_names.insert(0,'module')
#        container = model
#        for container_name in container_names:
#            container = container._modules[container_name]
#        m = container._modules[module_name]
#        #import pdb;pdb.set_trace()
#        add_grad  = s * torch.sign(m.weight.data)
#        add_grad  = add_grad *     torch.abs( (m.weight.data)).pow(-0.5)
#        m.weight.grad.data.add_( add_grad  )


def update_bn_by_names_dict(model, norm_layer_names, lr,s=1e-4,ngpu=False ,mask_dict =None,regular_method='L1'):
    assert mask_dict is not None ,"mask_dict needed"
    for norm_layer_name in norm_layer_names:
        *container_names, module_name = norm_layer_name.split('.')
        if ngpu :
            container_names.insert(0,'module')
        container = model
        for container_name in container_names:
            container = container._modules[container_name]
        m = container._modules[module_name]
        
        if norm_layer_name not in mask_dict.keys():
            continue
        mask = mask_dict[norm_layer_name]
        #mask = ~mask # mask is the saved index
        if regular_method=='L1':
            add_grad = s * torch.sign(m.weight.data) # L1
        elif regular_method=='L2':
            add_grad = s * m.weight.data # L2
        elif regular_method=='smoothL1':
            add_grad = s * torch.sign(m.weight.data) # L1
            L2mask = m.weight.data.abs() < lr *  s
            L2mask = L2mask.cuda()
            add_grad =  add_grad.clone().mul(~L2mask.cuda()) + L2mask.clone().mul(s * m.weight.data)    #s * m.weight.data  L2  #s * torch.sign(m.weight.data) L1
        elif regular_method=='shiyan':
            pass
        add_grad = add_grad.mul(mask)
        m.weight.grad.data.add_( add_grad  )





