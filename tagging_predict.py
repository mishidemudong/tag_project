#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:43:13 2020

@author: liang
"""

from tagging_train import TagModel
from tqdm import tqdm
import json


def predict(data, model_save_path):
    
    
    config = 
    model = TagModel()
    result = []
    for text in tqdm(data):
        pred_arguments = extract_arguments(trainmodel, tokenizer, schema_dict, text)
        result.append(pred_arguments)
    model.load_weights('best_model.weights') 
    
    
    return result
