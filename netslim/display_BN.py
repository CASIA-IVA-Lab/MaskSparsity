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

import numpy as np


def display_BN(model,norm_layer_names,ngpu=False,grad=False):
    if norm_layer_names == []:
        return    
    loss_s_1=0
    loss_s_2=0
    loss_s_3=0
    loss_s_4=0
    loss_s_5=0
    loss_s_6=0
    loss_s_7=0
    loss_s_8=0
    loss_s_9=0
    loss_s_10=0
    loss_s_11=0
    loss_s_12=0
    loss_s_13=0
    sum_bn =0
    flag =0
   

    for norm_layer_name in norm_layer_names:
        *container_names, module_name = norm_layer_name.split('.')
        if ngpu :
            container_names.insert(0,'module')
        container = model
        for container_name in container_names:
            container = container._modules[container_name]
        m = container._modules[module_name]

        if flag==0 and  m.weight.data.shape[0]>100 :
        #if norm_layer_name == 'layer1.7.bn1':
            print("\nbn_weight:\n",m.weight.data)
            if grad :
                print("\nbn_grad:\n",m.weight.grad.data)
            flag =1
                
        aaa1 = torch.sum(m.weight.data.abs().lt(1e-1)).item()
        aaa2 = torch.sum(m.weight.data.abs().lt(1e-2)).item()
        aaa3 = torch.sum(m.weight.data.abs().lt(1e-3)).item()
        aaa4 = torch.sum(m.weight.data.abs().lt(1e-4)).item()
        aaa5 = torch.sum(m.weight.data.abs().lt(1e-5)).item()
        aaa6 = torch.sum(m.weight.data.abs().lt(1e-6)).item()
        aaa7 = torch.sum(m.weight.data.abs().lt(1e-7)).item()
        aaa8 = torch.sum(m.weight.data.abs().lt(1e-8)).item()
        aaa9 = torch.sum(m.weight.data.abs().lt(1e-9)).item()
        aaa10 = torch.sum(m.weight.data.abs().lt(1e-10)).item()
        aaa11 = torch.sum(m.weight.data.abs().lt(1e-11)).item()
        aaa12 = torch.sum(m.weight.data.abs().lt(1e-12)).item()
        aaa13 = torch.sum(m.weight.data.abs().lt(1e-13)).item()
        
        loss_s_1 += aaa1
        loss_s_2 += aaa2
        loss_s_3 += aaa3
        loss_s_4 += aaa4
        loss_s_5 += aaa5
        loss_s_6 += aaa6
        loss_s_7 += aaa7
        loss_s_8 += aaa8
        loss_s_9 += aaa9
        loss_s_10 += aaa10
        loss_s_11 += aaa11
        loss_s_12 += aaa12
        loss_s_13 += aaa13
        sum_bn += m.weight.data.shape[0]
                # loss_s_b += torch.sum((m.weight.data.abs() + m.bias.data.abs()).abs()).item()
    
    print(norm_layer_names , '\nsum_bn:', sum_bn ,'\nloss_s_1:', loss_s_1)
    #import pdb;pdb.set_trace()
    loss_s_1 = loss_s_1 / sum_bn
    loss_s_2 = loss_s_2 / sum_bn
    loss_s_3 = loss_s_3 / sum_bn
    loss_s_4 = loss_s_4 / sum_bn
    loss_s_5 = loss_s_5 / sum_bn
    loss_s_6 = loss_s_6 / sum_bn
    loss_s_7 = loss_s_7 / sum_bn
    loss_s_8 = loss_s_8 / sum_bn
    loss_s_9 = loss_s_9 / sum_bn
    loss_s_10 = loss_s_10 / sum_bn
    loss_s_11 = loss_s_11 / sum_bn
    loss_s_12 = loss_s_12 / sum_bn
    loss_s_13 = loss_s_13 / sum_bn
   

    print("\n<1e-1:",loss_s_1 ,
          "\n<1e-2:",loss_s_2 ,
          "\n<1e-3:",loss_s_3 ,
          "\n<1e-4:",loss_s_4 ,
          "\n<1e-5:",loss_s_5 ,
          "\n<1e-6:",loss_s_6 ,
          "\n<1e-7:",loss_s_7 ,
          "\n<1e-8:",loss_s_8 ,
          "\n<1e-9:",loss_s_9 ,
          "\n<1e-10:",loss_s_10 ,
          "\n<1e-11:",loss_s_11 ,
          "\n<1e-12:",loss_s_12 ,
          "\n<1e-13:",loss_s_13
          )
    print("\n",loss_s_1 ,
          "\n",loss_s_2 ,
          "\n",loss_s_3 ,
          "\n",loss_s_4 ,
          "\n",loss_s_5 ,
          "\n",loss_s_6 ,
          "\n",loss_s_7 ,
          "\n",loss_s_8 ,
          "\n",loss_s_9 ,
          "\n",loss_s_10 ,
          "\n",loss_s_11 ,
          "\n",loss_s_12 ,
          "\n",loss_s_13
          )


def display_BN_graddata(model,norm_layer_names,ngpu=False,grad=True):

    sum_bn=0
    sum_zero_grad =0
    flag =0

    for norm_layer_name in norm_layer_names:
        *container_names, module_name = norm_layer_name.split('.')
        if ngpu :
            container_names.insert(0,'module')
        container = model
        for container_name in container_names:
            container = container._modules[container_name]
        m = container._modules[module_name]

        sum_zero_grad  += (m.weight.grad.data == 0).sum().item()
        sum_bn += m.weight.data.shape[0]
        #import pdb;pdb.set_trace()
        #if flag==0 and  m.weight.data.shape[0]>20 :
        if norm_layer_name == 'layer1.7.bn1':    
            if grad :
                print(norm_layer_name,"\n")
                print("\nbn_weight:\n",m.weight.data)#.tolist())
                print("\nbn_grad:\n",m.weight.grad.data)#.tolist())
                flag =1

    print("zero_grad_channels:",sum_zero_grad / sum_bn)


def display_BN_data(model,norm_layer_names,ngpu=False,grad=True):


    for norm_layer_name in norm_layer_names:
        *container_names, module_name = norm_layer_name.split('.')
        if ngpu :
            container_names.insert(0,'module')
        container = model
        for container_name in container_names:
            container = container._modules[container_name]
        m = container._modules[module_name]

        if norm_layer_name == 'layer3.7.bn1':
            print("\nbn_weight:\n",m.weight.data)
            if grad :
                print("\nbn_grad:\n",m.weight.grad.data)


def getBNmask(model,norm_layer_names,ngpu=False,grad=True,mask = None ,uniform =0.5):

    mask_BN = dict()
    for norm_layer_name in norm_layer_names:
        *container_names, module_name = norm_layer_name.split('.')
        if ngpu :
            container_names.insert(0,'module')
        container = model
        for container_name in container_names:
            container = container._modules[container_name]
        m = container._modules[module_name]
        num_ = int(m.weight.data.shape[0]*uniform)
        mask_BN[norm_layer_name] = torch.Tensor(num_*[False] + (m.weight.data.shape[0] - num_) * [True]).bool()

    return mask_BN



def getBNmask_2(model,norm_layer_names,ngpu=False,grad=True,mask = None):
    ff = 0
    ffzeroed =0
    ffnonezeroed =0
    mask_BN = dict()
    for norm_layer_name in norm_layer_names:
        *container_names, module_name = norm_layer_name.split('.')
        if ngpu :
            container_names.insert(0,'module')
        container = model
        for container_name in container_names:
            container = container._modules[container_name]
        m = container._modules[module_name]
        #import pdb;pdb.set_trace()
        mask_ = (m.weight.grad.data != 0).cpu()
        if mask is not None:#####compare the mask and warning
            #import pdb;pdb.set_trace()
            if mask_.shape[0] != (mask[norm_layer_name] == mask_).sum().item():
                mask1 = mask_.tolist()
                mask2 = mask[norm_layer_name].tolist()
                for i,j in zip(mask1,mask2):
                    if i and not j: #this iter ！=0 and iter one = 0
                      ffnonezeroed += 1
                    if not i and j:
                      ffzeroed += 1
                   
                
                #print('iter1:',mask_,'\niterother:',mask[norm_layer_name])
                #exit(0)
                ff += 1
            mask_ = mask_  |  mask[norm_layer_name]   # prune the False ,we need save the channels is false in every batch  False | true =true
        else:#get the mask
            mask_BN[norm_layer_name] = mask_.cpu()
    if ff!=0:
        print("!!!!!!!!!!!!!!!!!!!!!!!!diff layer:",ff,"   ffnonezeroed:",ffnonezeroed,"   ffzeroed:",ffzeroed)
    return mask_BN






