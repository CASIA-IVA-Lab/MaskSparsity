import copy
from functools import partial
import torch
import torch.nn as nn
from .graph_parser import get_pruning_layers
import numpy

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

MIN_SCALING_FACTOR = 1e-18
OT_DISCARD_PERCENT = 0.0001
OUT_CHANNEL_DIM = 0
IN_CHANNEL_DIM = 1
WEIGHT_POSTFIX = ".weight"
BIAS_POSTFIX = ".bias"
MIN_CHANNELS = 1




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

def plt_weight_OT(model_weight_dict):
    #import pdb;pdb.set_trace()
    #print(model_weight_dict)
    for k, (weight , prune_th) in model_weight_dict.items():
        #import pdb;pdb.set_trace()
        prune_th = css_thresholding(weight) 
        weight  = numpy.array(weight)
        weight[ weight  < MIN_SCALING_FACTOR] = MIN_SCALING_FACTOR
        weight  = numpy.sort(weight)
        
    pass

def is_transposed(model,data_name):
    *container_names, module_name = data_name.split('.')
    container = model
    for container_name in container_names:
        container = container._modules[container_name]
    
    if isinstance(container, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d)):
        return container.transposed
    else:
        return False




 
 


def css_thresholding(x, percent=OT_DISCARD_PERCENT, method = 'OT'):
    x_np = numpy.array(x)
    x_np[x_np < MIN_SCALING_FACTOR] = MIN_SCALING_FACTOR
    x_sorted = numpy.sort(x_np)
    print("\n max:",x_sorted[-1])
    print("\n",torch.from_numpy(x_sorted))
    if method == 'th':
        th = percent
        print("\nth-"+str(percent)+"  th:",th,"\n")
    elif method == 'OT':
        if x_sorted[-1] <1e-3 :# no bigger peak, all channels have been sparsed , prune all
            th = 1
        #elif x_sorted[-1] < x_sorted[0]*20  and x_sorted[-1] > 1e-3 :   #no smaller peak, all the channels are important,none will be pruned
        #    th = 0
        else:
            x2 = x_sorted**2
            Z = x2.sum()
            energy_loss = 0
            for i in range(x2.size):
                energy_loss += x2[i]
                if energy_loss / Z > percent:
                    break
            th = (x_sorted[i-1] + x_sorted[i]) / 2 if i > 0 else 0
        print("\nOT-"+str(percent)+"  th:",th,"\n") 
    return th


def group_weight_names(weight_names):
    grouped_names = {}
    for weight_name in weight_names:
        group_name = '.'.join(weight_name.split('.')[:-1])
        if group_name not in grouped_names:
            grouped_names[group_name] = [weight_name, ]
        else:
            grouped_names[group_name].append(weight_name)
    return grouped_names

######OT method
def liu2017(model,weights, percent, prec_layers, succ_layers, per_layer_normalization=False,get_mask=False,pruning_mask=None,method = 'OT',super_norm_layers = None , super_succ_layers = None ,super_prec_layers = None):
    """default pruning method as described in:
            Zhuang Liu et.al., "Learning Efficient Convolutional Networks through Network Slimming", in ICCV 2017"

    Arguments:
        weights (OrderedDict): unpruned model weights
        percent (float): OT algorithm Super-parameter
        prec_layers (dict): mapping from BN names to preceding convs/linears
        succ_layers (dict): mapping from BN names to succeeding convs/linears
        per_layer_normalization (bool): if do normalization by layer

    Returns:
        pruned_weights (OrderedDict): pruned model weights
    """

    # find all scale weights in BN layers
    #get the prune_mask for further sparsity
    prune_mask_dict = None
    if get_mask: 
        from collections import OrderedDict
        prune_mask_dict = OrderedDict()
    
    #get weight and
    from collections import OrderedDict 
    plt_dict = OrderedDict()
    
    scale_weights = []
    deepwise_mask = {}#####用于储存deepwise卷积剪枝
    no_name_list = [] #用于存下第一回没有遍历到的deepwise
    norm_layer_names = list(set(succ_layers) & set(prec_layers))

    # print("norm_layer_names:\n",norm_layer_names)
    print("\nsucc_layers(",type(succ_layers),"):\n",succ_layers)
    print("\nprec_layers(",type(prec_layers),"):\n",prec_layers)
    # unpruned_norm_layer_names = list(set(succ_layers) ^ set(prec_layers))
    
    grouped_weight_names = group_weight_names(weights.keys())
    ###遍历每一layers
    for norm_layer_name in norm_layer_names:
        norm_weight_name = norm_layer_name + WEIGHT_POSTFIX
        print(norm_layer_name)
        scale_weight = weights[norm_weight_name].abs()
################优先级最高
        ####group卷积重新制作剪枝掩码
        # 假设deepwise卷积只有一个输入一个输出
        flag_no  = False
        flag_yes = False
        for prune_layer_name in prec_layers[norm_layer_name]: #遍历前继
            for weight_name in grouped_weight_names[prune_layer_name]:  # 遍历前继层成员
                if weight_name.endswith(WEIGHT_POSTFIX):  # endswith以“.weight”结尾
                    if (weights[weight_name].size(IN_CHANNEL_DIM) == 1):  # 判断该层是不是一个deepwise卷积
                        if (weight_name in deepwise_mask):
                            prune_indices, prune_mask = deepwise_mask[weight_name]
                            #修改in_channels,groups,out_channels
                            inum = int(prune_mask.sum().item())
                            aa = weight_name.split('.')
                            layer_name_ = ''
                            for ai in aa[:-1]:
                                if is_number(ai):
                                    layer_name_ +=( '[' + ai + ']')
                                else :
                                    layer_name_ +=('.'+ai)                         
   
                            exec('model'+layer_name_+'.groups='+'inum')
                            exec('model'+layer_name_+'.in_channels='+'inum')
                            flag_yes = True
                        else:
                            no_name_list.append(norm_layer_name)
                            flag_no = True
                    break
        if flag_no:
            #print("deepwise等待上一层的剪枝掩码:", norm_layer_name, weight_name )
            continue
        if not flag_yes :
#############优先级次高
            if per_layer_normalization:
                scale_weight = scale_weight / scale_weight.sum()
            scale_weight_list = [_.abs().item() for _ in list(scale_weight)]
            prune_th =   css_thresholding(scale_weight_list, percent=percent ,method = method)


            plt_dict[norm_layer_name] = [scale_weight_list , prune_th]  #
            
          
            #if(norm_layer_name == 'model_s32.model_s32.1.4'):
             #   prune_th = 0.010
            #print(norm_weight_name,"剪枝阈值:",prune_th)
             
            if pruning_mask is not None:
                prune_mask = pruning_mask[norm_layer_name]
            else: 
                prune_mask = scale_weight > prune_th#获取剪枝掩码，大于剪枝阈值保留
            
          
            if prune_mask.sum().item() == scale_weight.size(0): #要剪枝的通道数（输出）为零，则跳过
            
             
                #print("要剪枝的为0:", norm_layer_name)
                continue

            # in case not to prune the whole layer
            if prune_mask.sum() < MIN_CHANNELS:#   局部剪枝保护（这里是最小通道3，我们也可以改成按比例保留）,重新构建剪枝掩码
                
                scale_weight_list = [_.abs().item() for _ in list(scale_weight)]
                scale_weight_list.sort(reverse=True)
                prune_mask = scale_weight >= scale_weight_list[MIN_CHANNELS-1]
                if prune_mask.sum()  != MIN_CHANNELS :
                    prune_mask = scale_weight > prune_th
                    prune_mask[0]=True
                   

            prune_indices = torch.nonzero(prune_mask).flatten()#nonzero得到数组中非零值的坐标


        # print(norm_layer_name)

        if get_mask:
            prune_mask_dict[norm_layer_name] = prune_mask
 
        #print(prune_indices , prune_mask)
        #tensor([ 0,  2,  3,  4,  5,  6,  7,  8, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21,
        #22, 23, 25, 26, 27, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 42, 43,
        #44, 46, 47, 48, 49, 51, 52, 55, 57, 58, 60, 61, 62, 63]) tensor([ True, False,  True,  True,  True,  True,  True,  True,  True, False,
        # True,  True,  True, False,  True,  True,  True,  True, False,  True,
        # True,  True,  True,  True, False,  True,  True,  True, False,  True,
        # True,  True,  True, False,  True,  True,  True,  True,  True,  True,
        # True, False,  True,  True,  True, False,  True,  True,  True,  True,
        #False,  True,  True, False, False,  True, False,  True,  True, False,
        # True,  True,  True,  True])

        if  get_mask:
            pass 

        # 1. prune source normalization layer，先剪掉剪掉源层（BN）
        for weight_name in grouped_weight_names[norm_layer_name]:
            weights[weight_name] = weights[weight_name].masked_select(prune_mask)

        # 2. prune target succeeding conv/linear/... layers     剪掉源层（BN）后继层的（输入通道）
        for prune_layer_name in succ_layers[norm_layer_name]:   #遍历后继
            for weight_name in grouped_weight_names[prune_layer_name]: #遍历后继的各层(如bn层的 weights ，bias ，meaning ，var)
                if weight_name.endswith(WEIGHT_POSTFIX):#endswith以“.weight”结尾
                    # print(weight_name,prune_indices)
                    if (weights[weight_name].size(IN_CHANNEL_DIM) == 1):#deepwise卷积，剪枝掩码存入字典
                        # print("                        ",norm_layer_name,weight_name)
                        deepwise_mask[weight_name]=[prune_indices , prune_mask]
                        continue
                    weights[weight_name] = weights[weight_name].index_select(IN_CHANNEL_DIM, prune_indices)#剪输入通道

        # 3. prune target preceding conv/linear/... layers    #剪掉源层（BN）前继层的（输入通道）也就是该modules的卷积
        for prune_layer_name in prec_layers[norm_layer_name]: #同上
            for weight_name in grouped_weight_names[prune_layer_name]:#同上
                if weight_name.endswith(WEIGHT_POSTFIX):#endswith以“.weight”结尾
                    weights[weight_name] = weights[weight_name].index_select(OUT_CHANNEL_DIM, prune_indices)#剪输出通道
                elif weight_name.endswith(BIAS_POSTFIX):#endswith以“.bias”结尾
                    weights[weight_name] = weights[weight_name].index_select(0, prune_indices)#剪输出通道



    print(no_name_list)
    #处理之前没有遍历到的deepwise卷积
    for norm_layer_name in no_name_list:
        ####group卷积重新制作剪枝掩码
        # 假设deepwise卷积只有一个输入一个输出
        flag_no  = False
        for prune_layer_name in prec_layers[norm_layer_name]:  # 遍历前继
            for weight_name in grouped_weight_names[prune_layer_name]:  # 遍历前继层成员
                if weight_name.endswith(WEIGHT_POSTFIX):  # endswith以“.weight”结尾
                    if(weights[weight_name].size(IN_CHANNEL_DIM) == 1):#判断该层是不是一个deepwise卷积
                        if (weight_name in deepwise_mask):
                            prune_indices, prune_mask = deepwise_mask[weight_name]
                            #修改in_channels,groups,out_channels
                            inum = int(prune_mask.sum().item())
                            aa = weight_name.split('.')
                            layer_name_ = ''
                            for ai in aa[:-1]:
                                if is_number(ai):
                                    layer_name_ +=( '[' + ai + ']')
                                else :
                                    layer_name_ +=('.'+ai)
                            exec('model'+layer_name_+'.groups='+'inum')
                            exec('model'+layer_name_+'.in_channels='+'inum')
                            # exec("print('model.'+layer_name_+'.groups')")
                        else:
                            flag_no = True
        if(flag_no):
            print(norm_layer_name,":daaddddasdasdasdasdasdasdasda!!!!!")
            continue


        print(norm_layer_name)
        #print(prune_indices , prune_mask)
        # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
        #         18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])
        # tensor([True, True, True, True, True, True, True, True, True, True, True, True,
        #         True, True, True, True, True, True, True, True, True, True, True, True,
        #         True, True, True, True, True, True, True, True])




        # 1. prune source normalization layer，先剪掉剪掉源层（BN）
        for weight_name in grouped_weight_names[norm_layer_name]:
            weights[weight_name] = weights[weight_name].masked_select(prune_mask)

        # 2. prune target succeeding conv/linear/... layers     剪掉源层（BN）后继层的（输入通道）
        for prune_layer_name in succ_layers[norm_layer_name]:   #遍历后继
            for weight_name in grouped_weight_names[prune_layer_name]: #遍历后继的各层(如bn层的 weights ，bias ，meaning ，var)
                if weight_name.endswith(WEIGHT_POSTFIX):#endswith以“.weight”结尾
                    # print(weight_name,prune_indices)
                    if (weights[weight_name].size(IN_CHANNEL_DIM) == 1):#deepwise卷积，剪枝掩码存入字典
                        deepwise_mask[weight_name]=[prune_indices , prune_mask]
                        continue
                    weights[weight_name] = weights[weight_name].index_select(IN_CHANNEL_DIM, prune_indices)#剪输入通道

        # 3. prune target preceding conv/linear/... layers    #剪掉源层（BN）前继层的（输入通道）也就是该modules的卷积
        for prune_layer_name in prec_layers[norm_layer_name]: #同上
            for weight_name in grouped_weight_names[prune_layer_name]:#同上
                if weight_name.endswith(WEIGHT_POSTFIX):#endswith以“.weight”结尾
                    weights[weight_name] = weights[weight_name].index_select(OUT_CHANNEL_DIM, prune_indices)#剪输出通道
                elif weight_name.endswith(BIAS_POSTFIX):#endswith以“.bias”结尾
                    weights[weight_name] = weights[weight_name].index_select(0, prune_indices)#剪输出通道

    
    #plt_weight_OT( plt_dict  )
######################################################################################
    if   super_norm_layers is None or super_succ_layers is None or super_prec_layers is None:
        return prune_mask_dict , weights 
    for norm_layer_name_list, succ_layers_list, prec_layers_list in zip( super_norm_layers, super_succ_layers, super_prec_layers):
        mask_list = []
        for norm_layer_name in norm_layer_name_list:
            norm_weight_name = norm_layer_name + WEIGHT_POSTFIX
            scale_weight = weights[norm_weight_name].abs()
            scale_weight_list = [_.abs().item() for _ in list(scale_weight)]
            prune_th =   css_thresholding(scale_weight_list, percent=percent ,method = method)
            if pruning_mask is not None:
                prune_mask = pruning_mask[norm_layer_name]
            else: 
                prune_mask = scale_weight > prune_th#获取剪枝掩码，大于剪枝阈值保留
            if prune_mask.sum() < MIN_CHANNELS:#   局部剪枝保护（这里是最小通道3，我们也可以改成按比例保留）,重新构建剪枝掩码
                scale_weight_list = [_.abs().item() for _ in list(scale_weight)]
                scale_weight_list.sort(reverse=True)
                prune_mask = scale_weight >= scale_weight_list[MIN_CHANNELS-1]
                if prune_mask.sum()  != MIN_CHANNELS :
                    prune_mask = scale_weight > prune_th
                    prune_mask[0]=True   
            mask_list.append((~prune_mask).int())


        prune_mask = torch.zeros( len(mask_list[0]))
        prune_mask_ = prune_mask.clone()
        min_prunenum = min([ _.sum().item() for _ in mask_list])
        
        for mask in mask_list:
            prune_mask_ =  prune_mask_ + mask

        #if 's4.b1.f.c_bn' in norm_layer_name_list:
        #    mask_dict ={}
        #    for name , mask in zip(norm_layer_name_list ,  mask_list): 
        #        mask_dict[name] = mask
        #        print(mask.sum())
        #    for i in mask_list:
        #        print([ _.item()   for _ in i])
        #    import pdb;pdb.set_trace() 

        for i in range(len(mask_list),0,-1):#保守选择
            prune_mask = (prune_mask_ >= i)
            if prune_mask.sum() >= min_prunenum:
                break
        prune_mask = ~prune_mask #剪枝mask转化为原来的保留mask


        if get_mask:
            for norm_layer_name in norm_layer_name_list:
                prune_mask_dict[norm_layer_name] = prune_mask

        if prune_mask.sum().item() == scale_weight.size(0): #要剪枝的通道数（输出）为零，则跳过
            continue
        prune_indices = torch.nonzero(prune_mask).flatten()#nonzero得到数组中非零值的坐标
        
        # 1. prune source normalization layer，先剪掉剪掉源层（BN）
        for norm_layer_name in norm_layer_name_list:
            for weight_name in grouped_weight_names[norm_layer_name]:
                weights[weight_name] = weights[weight_name].masked_select(prune_mask)
      
        # 2. prune target succeeding conv/linear/... layers     剪掉源层（BN）后继层的（输入通道）
        
        for prune_layer_name in succ_layers_list:   #遍历后继
            for weight_name in grouped_weight_names[prune_layer_name]: #遍历后继的各层(如bn层的 weights ，bias ，meaning ，var)
                if weight_name.endswith(WEIGHT_POSTFIX):#endswith以“.weight”结尾
                    transpose_flag = is_transposed(model,weight_name[:-6])
                    #print(weight_name)#,prune_indices)
                    
                    if (weights[weight_name].size(IN_CHANNEL_DIM) == 1):#deepwise卷积，剪枝掩码存入字典
                        # print("                        ",norm_layer_name,weight_name)
                        deepwise_mask[weight_name]=[prune_indices , prune_mask]
                        continue
                    if transpose_flag:#反卷积
                        weights[weight_name] = weights[weight_name].index_select(OUT_CHANNEL_DIM, prune_indices)#剪输入通道
                    else:#正常卷积
                        weights[weight_name] = weights[weight_name].index_select(IN_CHANNEL_DIM, prune_indices)#剪输入通道
        # 3. prune target preceding conv/linear/... layers    #剪掉源层（BN）前继层的（输入通道）也就是该modules的卷积
        for prune_layer_name in prec_layers_list: #同上
            for weight_name in grouped_weight_names[prune_layer_name]:#同上
                if weight_name.endswith(WEIGHT_POSTFIX):#endswith以“.weight”结尾
                    transpose_flag = is_transposed(model,weight_name[:-6])
                    if transpose_flag:#反卷积
                        weights[weight_name] = weights[weight_name].index_select(IN_CHANNEL_DIM, prune_indices)#剪输出通道
                    else:#正常卷积
                        weights[weight_name] = weights[weight_name].index_select(OUT_CHANNEL_DIM, prune_indices)#剪输出通道
                elif weight_name.endswith(BIAS_POSTFIX):#endswith以“.bias”结尾
                    weights[weight_name] = weights[weight_name].index_select(0, prune_indices)#剪输出通道


#######################################################################################
    return prune_mask_dict , weights


liu2017_normalized_by_layer = partial(liu2017, per_layer_normalization=True)


def _dirty_fix(module, param_name, pruned_shape):
    module_param = getattr(module, param_name)

    # identify the dimension to prune
    pruned_dim = 0
    for original_size, pruned_size in zip(module_param.shape, pruned_shape):
        if original_size != pruned_size:
            keep_indices = torch.LongTensor(range(pruned_size)).to(module_param.data.device)
            module_param.data = module_param.data.index_select(pruned_dim, keep_indices)

            # modify number of features/channels
            if param_name == "weight":
                if isinstance(module, nn.modules.batchnorm._BatchNorm) or \
                        isinstance(module, nn.modules.instancenorm._InstanceNorm) or \
                        isinstance(module, nn.GroupNorm):
                    module.num_features = pruned_size
                elif isinstance(module, nn.modules.conv._ConvNd):
                    if pruned_dim == OUT_CHANNEL_DIM:
                        module.out_channels = pruned_size
                    elif pruned_dim == IN_CHANNEL_DIM:
                        module.in_channels = pruned_size
                elif isinstance(module, nn.Linear):
                    if pruned_dim == OUT_CHANNEL_DIM:
                        module.out_features = pruned_size
                    elif pruned_dim == IN_CHANNEL_DIM:
                        module.in_features = pruned_size
                else:
                    pass
        pruned_dim += 1


def load_pruned_model(model, pruned_weights, prefix='', load_pruned_weights=True, inplace=True):
    """load pruned weights to a unpruned model instance

    Arguments:
        model (pytorch model): the model instance
        pruned_weights (OrderedDict): pruned weights
        prefix (string optional): prefix (if has) of pruned weights
        load_pruned_weights (bool optional): load pruned weights to model according to the ICLR 2019 paper:
            "Rethinking the Value of Network Pruning", without finetuning, the model may achieve comparable or even
            better results
        inplace (bool, optional): if return a copy of the model

    Returns:
        a model instance with pruned structure (and weights if load_pruned_weights==True)
    """
    model_weight_names = model.state_dict().keys()
    pruned_weight_names = pruned_weights.keys()

    # check if module names match
    assert set([prefix + _ for _ in model_weight_names]) == set(pruned_weight_names) , print (set([prefix + _ for _ in model_weight_names])  ,"\n\n\n", set(pruned_weight_names)) 
    
    #scratch:   ~    stem  
    #test :    stem  stem

    # inplace or return a new copy
    if not inplace:
        pruned_model = copy.deepcopy(model)
    else:
        pruned_model = model

    # update modules with mis-matched weight
    model_weights = pruned_model.state_dict()
    for model_weight_name in model_weight_names:
       # print(model_weight_name,":(",model_weights[model_weight_name].shape ,pruned_weights[prefix + model_weight_name].shape,")")
        if model_weights[model_weight_name].shape != pruned_weights[prefix + model_weight_name].shape:
            # print("   ",model_weight_name," ",prefix + model_weight_name)
            *container_names, module_name, param_name = model_weight_name.split('.')
            container = model
            for container_name in container_names:
                container = container._modules[container_name]
            module = container._modules[module_name]
            _dirty_fix(module, param_name, pruned_weights[prefix + model_weight_name].shape)

    # print(pruned_model.state_dict()['conf.3.weight'].shape ,pruned_weights['conf.3.weight'].shape)


    if load_pruned_weights:
        pruned_model.load_state_dict({k: v for k, v in pruned_weights.items()})
    return pruned_model


def prune(model, input_shape,  prune_ratio=OT_DISCARD_PERCENT, prune_method=liu2017, get_mask=False, pruning_mask = None, min_channels = 1 ,Threshold_method = 'OT', super_norm_layers = None , super_succ_layers = None ,super_prec_layers = None  ):
    """prune a model

    Arguments:
        model (pytorch model): the model instance
        input_shape (tuple): shape of the input tensor
        prune_ratio (float): ratio of be pruned channels to total channels
        prune_method (method): algorithm to prune weights

    Returns:
        a model instance with pruned structure (and weights if load_pruned_weights==True)

    Pipeline:
        1. generate mapping from tensors connected to BNs by parsing torch script traced graph
        2. identify corresponding BN and conv/linear like:
            conv/linear --> ... --> BN --> ... --> conv/linear
                                     |
                                    ...
                                     | --> relu --> ... --> conv/linear
                                    ...
                                     | --> ... --> maxpool --> ... --> conv/linear
            , where ... represents per channel operations. all the floating nodes must be conv/linear
        3. prune the weights of BN and connected conv/linear
        4. load weights to a unpruned model with pruned weights
    """
    global MIN_CHANNELS
    MIN_CHANNELS = min_channels

    # convert to CPU for simplicity
    src_device = next(model.parameters()).device
    model = model.cpu()
    # parse & generate mappings to BN layers
    prec_layers, succ_layers = get_pruning_layers(model, input_shape)
 
    # prune weights
    prune_mask_dict,pruned_weights = prune_method(model=model,weights=model.state_dict(),  percent=prune_ratio, prec_layers = prec_layers,succ_layers=succ_layers,get_mask=get_mask,pruning_mask = pruning_mask, method = Threshold_method, super_norm_layers = super_norm_layers  , super_succ_layers = super_succ_layers ,super_prec_layers = super_prec_layers  )
    #liu2017(model,weights, percent, prec_layers, succ_layers, per_layer_normalization=False,get_mask=False):
    # prune model according to pruned weights
    if MIN_CHANNELS ==0:
        return prune_mask_dict , pruned_weights
    else :
        pruned_model = load_pruned_model(model, pruned_weights)
        return prune_mask_dict , pruned_model.to(src_device)
