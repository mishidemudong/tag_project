#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 14:11:42 2020

@author: liang
"""

import os
#os.environ['RECOMPUTE'] = '1'

import json
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, ViterbiDecoder
from bert4keras.layers import ConditionalRandomField,MaximumEntropyMarkovModel
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm
import pylcs

def parse_data(filename):
    D = []
    with open(filename) as f:
        for l in f:
            l = json.loads(l)
            arguments = {}
            for event in l['event_list']:
                for argument in event['arguments']:
                    key = argument['argument']
#                    value = (event['event_type'], argument['role'])
                    value = '#'.join([event['event_type'], argument['role']])
                    arguments[key] = value
            D.append((l['text'], arguments))
    return D




def load_data():
        # 读取数据
    train_data = parse_data('./datasets/train.json')
    valid_data = parse_data('./datasets/dev.json')
    
    schema_dict = {}
    
    # 读取schema
    with open('./datasets/event_schema.json') as f:
        id2label, label2id, n = {}, {}, 0
        for l in f:
            l = json.loads(l)
            for role in l['role_list']:
                key = '#'.join([l['event_type'], role['role']])
                id2label[n] = key
                label2id[key] = n
                n += 1
        num_labels = len(id2label) * 2 + 1
        
    schema_dict['id2label'] = id2label    
    schema_dict['label2id'] = label2id 
    schema_dict['num_labels'] = num_labels
        
    return train_data,valid_data,schema_dict


class data_generator(DataGenerator):
    """数据生成器
    """
    
    def __init__(self, data, batch_size, tokenizer,label2id, maxlen):#64,256
        self.data = data
        self.batch_size = batch_size
        self.label2id = label2id
        self.maxlen = maxlen
        self.tokenizer = tokenizer
        if hasattr(self.data, '__len__'):
            self.steps = len(self.data) // self.batch_size
            if len(self.data) % self.batch_size != 0:
                self.steps += 1
        else:
            self.steps = None
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, arguments) in self.sample(random):
            token_ids, segment_ids = self.tokenizer.encode(text, max_length=self.maxlen)
            labels = [0] * len(token_ids)
            for argument in arguments.items():
                a_token_ids = self.tokenizer.encode(argument[0])[0][1:-1]
                start_index = self.search(a_token_ids, token_ids)
                if start_index != -1:
                    labels[start_index] = self.label2id[argument[1]] * 2 + 1
                    for i in range(1, len(a_token_ids)):
                        labels[start_index + i] = self.label2id[argument[1]] * 2 + 2
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []
                
                

def viterbi_decode(nodes, trans, num_labels):
    """Viterbi算法求最优路径
    其中nodes.shape=[seq_len, num_labels],
        trans.shape=[num_labels, num_labels].
    """
    labels = np.arange(num_labels).reshape((1, -1))
    scores = nodes[0].reshape((-1, 1))
    scores[1:] -= np.inf  # 第一个标签必然是0
    paths = labels
    for l in range(1, len(nodes)):
        M = scores + trans + nodes[l].reshape((1, -1))
        idxs = M.argmax(0)
        scores = M.max(0).reshape((-1, 1))
        paths = np.concatenate([paths[:, idxs], labels], 0)
    return paths[:, scores[0].argmax()]


def extract_arguments(Model, tokenizer, schema_dict, text):
    """命名实体识别函数
    """
    tokens = tokenizer.tokenize(text)
    while len(tokens) > 512:
        tokens.pop(-2)
    token_ids = tokenizer.tokens_to_ids(tokens)
    segment_ids = [0] * len(token_ids)
    nodes = Model.model.predict([[token_ids], [segment_ids]])[0]
    trans = K.eval(Model.CRF.trans)
    labels = viterbi_decode(nodes, trans, schema_dict['num_labels'])[1:-1]
    arguments, starting = [], False
    for token, label in zip(tokens[1:-1], labels):
        if label > 0:
            if label % 2 == 1:
                starting = True
                arguments.append([[token], schema_dict['id2label'][(label - 1) // 2]])
            elif starting:
                arguments[-1][0].append(token)
            else:
                starting = False
        else:
            starting = False

    return {tokenizer.decode(w, w): l for w, l in arguments}


def evaluate(data, Model, tokenizer, schema_dict):
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for text, arguments in tqdm(data):
        inv_arguments = {v: k for k, v in arguments.items()}
        pred_arguments = extract_arguments(Model, tokenizer, schema_dict, text)
        pred_inv_arguments = {v: k for k, v in pred_arguments.items()}
        Y += len(pred_inv_arguments)
        Z += len(inv_arguments)
        for k, v in pred_inv_arguments.items():
            if k in inv_arguments:
                # 用最长公共子串作为匹配程度度量
                l = pylcs.lcs(v, inv_arguments[k])
                X += 2. * l / (len(v) + len(inv_arguments[k]))
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return {'f1':f1, 'precision':precision, 'recall':recall}

##albert model
class TagModel():
    def __init__(self, config):
              
        self.albert = config['albert']
        self.config_path = config['config_path']
        self.checkpoint_path = config['checkpoint_path']
        self.crf_lr_multiplier = config['crf_lr_multiplier']
        
        self.num_labels = config['schema_dict']['num_labels']
        
        self.build()
        
    def build(self):
        if self.albert:
            model = build_transformer_model(
                self.config_path,
                self.checkpoint_path,
                model='albert',
            )
            output_layer = 'Transformer-FeedForward-Norm'
            bert_out = model.get_layer(output_layer).get_output_at(4 - 1)
            
        else:
            model = build_transformer_model(
                self.config_path,
                self.checkpoint_path,
            )
            bert_out = model.output
            
        output = Dense(self.num_labels)(bert_out)
        
        self.CRF = MaximumEntropyMarkovModel(lr_multiplier=self.crf_lr_multiplier)
        output = self.CRF(output)
        
        self.model = Model(model.input, output)
        self.model.summary()
        



#def train(train_corpus, eval_corpus, train_param, model_save_path, logger):
def train(train_param, model_save_path):
#    logger.info()
    train_data,valid_data,schema_dict= load_data()
    train_param['schema_dict'] = schema_dict
    
#    print(train_param)
    
    # 建立分词器
    tokenizer = Tokenizer(train_param['dict_path'], do_lower_case=True)
    trainmodel = TagModel(train_param)
    
    trainmodel.model.compile(
            loss=trainmodel.CRF.sparse_loss,
            optimizer=Adam(train_param['learing_rate']),
            metrics=[trainmodel.CRF.sparse_accuracy]
        )
    
    train_generator = data_generator(train_data, train_param['batch_size'], tokenizer,schema_dict['label2id'], train_param['maxlen'])
    
    trainmodel.model.fit_generator(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=train_param['epochs'],
        )
    savemodel_name = os.path.join(model_save_path ,'best_model.weights')
    trainmodel.model.save_weights(savemodel_name)
    
    params_file = os.path.join(model_save_path,'config.json')
    with open(params_file,'w',encoding='utf-8') as json_file:
        json.dump(train_param,json_file, indent=4 ,ensure_ascii=False)

    
    eval_result = evaluate(valid_data, trainmodel, tokenizer, schema_dict)
    
    
    
    return eval_result



if __name__ == '__main__':
    
    train_param = {}
    # 基本信息
    train_param['maxlen'] = 64
    train_param['epochs'] = 5
    train_param['batch_size'] = 32
    train_param['learing_rate'] = 2e-5  # bert_layers越小，学习率应该要越大
    train_param['crf_lr_multiplier'] = 1000  # 必要时扩大CRF层的学习率
    
    #albert配置
    bert_path = '/media/liang/Nas/PreTrainModel'
    train_param['albert'] = True
    train_param['config_path'] = bert_path + '/albert/albert_tiny_zh_google/albert_config_tiny_g.json' #albert_xlarge_google_zh_183k
    train_param['checkpoint_path'] = bert_path + '/albert/albert_tiny_zh_google/albert_model.ckpt'
    train_param['dict_path'] = bert_path + '/albert/albert_tiny_zh_google/vocab.txt'
    
    
    model_save_path = './model'
    eval_result = train(train_param, model_save_path)
    
    


#    model.load_weights('best_model.weights')
#    predict_to_file('./datasets/test1.json', 'ee_pred.json')