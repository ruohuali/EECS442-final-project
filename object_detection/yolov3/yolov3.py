#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:44:29 2021

@author: jessica
"""

import json
import itertools
import string
import math
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

CONFIG_ROUTE = 'darknet53.cfg'



def create_comps(file):
    f = open(file, 'r')
    lines = f.read().split('\n')
    comps = []
    comp = {}
    for line in lines:
        l = line.rstrip().lstrip()
        if len(l) == 0 or l[0] == '#': 
            continue
        # detect comp
        if l[0] == '[':
            if len(comp) != 0:
                comps.append(comp)
                comp = {}
            comp['layer'] = l[1:-1]
        else:
            key, val = l.split('=')
            comp[key]=val
    comps.append(comp)
    
    return comps

    
class ShortCutLayer(nn.Module):
    def __init__(self):
        super(ShortCutLayer, self).__init__()
   
        
class DarkNet53(nn.Module):
    def __init__(self, comps):
        super(DarkNet53, self).__init__()
        self.comps = comps
        self.net = self.comps[0]
        self.layers = self.comps[1:]
        self.in_channels = 3
        self.modules = self._build_modules()
       
        
    def _build_modules(self):
        
        in_channels = self.in_channels
        modules = nn.ModuleList()
        
        for i, layer in enumerate(self.layers):
            module = nn.Sequential()
            
            if layer['layer'] == 'convolutional':
                ## conv layer
                filters = int(layer['filters'])
                kernel = int(layer['size'])
                stride = int(layer['stride'])
                padding = (stride-1) // 2 if layer['pad'] else 0
                conv = nn.Conv2d(in_channels, filters, kernel, stride, padding)
                module.add_module('conv{}'.format(i), conv)
                
                ## batch norm
                norm = nn.BatchNorm2d(filters)
                module.add_module('norm{}'.format(i), norm)
                
                ## activation
                leaky = nn.LeakyReLU(0.1, inplace = True)
                module.add_module('leakyReLU{}'.format(i), leaky)
                
                in_channels = filters
                
            elif layer['layer'] == 'shortcut':
                shortcut = ShortCutLayer()
                module.add_module("shortcut{}".format(i), shortcut)
            modules.append(module)
            
        return modules
    
    def forward(self, x):
        tmp = []
        for i, layer in enumerate(self.layers):
            if layer['layer'] == 'convolutional':
                x = self.modules[i](x)
                tmp.append(x)
            elif layer['layer'] == 'shortcut':
                x = tmp[i-1]+tmp[i+int(layer["from"])]
        return x
        
       
comps = create_comps(CONFIG_ROUTE)     
dn = DarkNet53(comps)    
    
    
    
    



        
        
    
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
