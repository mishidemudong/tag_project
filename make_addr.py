#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 16:23:13 2020

@author: liang
"""
addr = dict()
with open("/media/liang/Project/Hemei-Project/tag_project/datasets/addr_dict.txt") as f:
#    l = f.readline()
    i = 12001
    for l in f:
        if i < 12014:
            addr[l.replace('"', '').strip()] = i
            i += 1 
        elif i == 12014:
            addr[l.replace('"', '').strip()] = i+1
            i+=2
        else:
    
              
        
            addr[l.replace('"', '').strip()] = i
            i += 1 
        