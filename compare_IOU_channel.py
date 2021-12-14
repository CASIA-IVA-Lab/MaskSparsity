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

from tensorboardX import SummaryWriter




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-ma', type=str, required=True, help='net type')
    parser.add_argument('-mb', type=str, default=True, help='use gpu or not')
    args = parser.parse_args()
  
    a = torch.load(args.ma)
    b = torch.load(args.mb)
   
    for i in a.keys():
        print(i,'\n',a[i])
        if i in b:
            print(b[i])
        else:
            print(torch.Tensor(len(a[i])*[True]).bool()) 
    
    mka = []
    mkb = []

    if len(a.keys()) > len(b.keys()):
        for i in a.keys():
            mka+= a[i].tolist()
            if i in b.keys():
                mkb += b[i].tolist()
            else:
                mkb +=len(a[i]) * [True]
    elif len(a.keys()) < len(b.keys()):
        for i in b.keys():
            mkb+= b[i].tolist()
            if i in a.keys():
                mka += a[i].tolist()
            else:
                mka +=len(b[i]) * [True]

    else:
        for i in a.keys():
            mka+= a[i].tolist()
            mkb += b[i].tolist() 

    mkb = ~(torch.Tensor(mkb).bool())
    mka = ~(torch.Tensor(mka).bool()) 
    
    print(args.ma,':',mka.sum().item() ,'\n',args.mb,':',mkb.sum().item(),'\n','&:',(mka&mkb).sum().item())




































