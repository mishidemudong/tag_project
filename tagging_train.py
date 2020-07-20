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
#from bert4keras.snippets import open, ViterbiDecoder
from ViterbiDecoderClass import ViterbiDecoder

from bert4keras.layers import ConditionalRandomField,MaximumEntropyMarkovModel
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm


def read_addr_dict():
    with open("./datasets/addr_dict.dict") as f:
        for l in f:
            addr_dict = l

    return { j:i for i,j in addr_dict.items()}

def parse_data(filename):
    D, schema_dict = [], {}
    id2label, label2id, n = {}, {}, 0
    labels = set()
    with open(filename) as f:
        for l in f:
            l = json.loads(l)
#            print(l)
            for item in l:
                tmp = []
                for entity_type in item['entity_result']:
                    start = int(entity_type['start_pos'])
                    end = int(entity_type['end_pos']) + 1
                    labels.add(entity_type['entity_type'])
#                    print(entity_type)
                    tmp.append([item['text'][start:end], entity_type['entity_type']])
                D.append(tmp)
                
    for label in list(labels):
        id2label[n] = label
        label2id[label] = n
        n += 1
        
    num_labels = len(id2label) * 2 + 1
        
    schema_dict['id2label'] = id2label    
    schema_dict['label2id'] = label2id 
    schema_dict['num_labels'] = num_labels
    
    return D, schema_dict

def parse_labeleddata(filename):
    D, schema_dict = [], {}
    id2label, label2id, n = {}, {}, 0
    labels = set()
    with open(filename) as f:
        for l in f:
            l = json.loads(l)
#            print(l)
            for item in l:
                for entity_type in item['entity_result']:
                    labels.add(entity_type['entity_type'])
                    
                D.append((item['text'], item['entity_result']))
                
    for label in list(labels):
        id2label[n] = label
        label2id[label] = n
        n += 1
        
    num_labels = len(id2label) * 2 + 1
        
    schema_dict['id2label'] = id2label    
    schema_dict['label2id'] = label2id 
    schema_dict['num_labels'] = num_labels
    
    return D, schema_dict


def parse_data1(filename):
    addr_dict = read_addr_dict()
    D, schema_dict = [], {}
    id2label, label2id, n = {}, {}, 0
    labels = set()
    with open(filename) as f:
        for l in f:
            l = json.loads(l)
#            print(l)
            for item in l['txt']:
                tmp = []
                for entity_type in item['entityContent']:
                    start = int(entity_type['start_pos'])
                    end = int(entity_type['end_pos']) + 1
                    labels.add(entity_type['label_id'])
#                    print(entity_type)
                    tmp.append([item['txt'][start:end], addr_dict[entity_type['label_id']]])
                D.append(tmp)
                
    for id_ in list(labels):
        label = addr_dict[id_]
        id2label[n] = label
        label2id[label] = n
        n += 1
        
    num_labels = len(id2label) * 2 + 1
        
    schema_dict['id2label'] = id2label    
    schema_dict['label2id'] = label2id 
    schema_dict['num_labels'] = num_labels
    
    return D, schema_dict


def load_data():
    # 读取数据
    train_data, schema_dict = parse_data('./labeled_data/train.json')
    valid_data,_ = parse_data('./labeled_data/dev.json')
        
    return train_data,valid_data,schema_dict

def transform2geshi(length, arguments, label2id):
    labels = [0] * length
    for argument in arguments:
#        print(argument['word'])
        start_index = int(argument['start_pos'])
        if start_index != -1:
            labels[start_index] = label2id[argument['entity_type']] * 2 + 1
            for i in range(1, len(argument['word'])):
                labels[start_index + i] = label2id[argument['entity_type']] * 2 + 2
    assert len(labels) == length
    
    return labels

def ziptextlabels(text, arguments):
    labels = [0] * len(text)
    for argument in arguments:
#        print(argument['word'])
        start_index = int(argument['start_pos'])
        if start_index != -1:
            labels[start_index] = argument['entity_type']
            for i in range(1, len(argument['word'])):
                labels[start_index + i] = argument['entity_type']
    assert len(labels) == len(text)
    
    return (text, labels)

                
                
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
    
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.sample(random):
            token_ids, labels = [self.tokenizer._token_start_id], [0]
            for w, l in item:
                w_token_ids = self.tokenizer.encode(w)[0][1:-1]
                if len(token_ids) + len(w_token_ids) < self.maxlen:
                    token_ids += w_token_ids
                    if l == 'O':
                        labels += [0] * len(w_token_ids)
                    else:
                        B = self.label2id[l] * 2 + 1
                        I = self.label2id[l] * 2 + 2
                        labels += ([B] + [I] * (len(w_token_ids) - 1))
                else:
                    break
            token_ids += [self.tokenizer._token_end_id]
            labels += [0]
            segment_ids = [0] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []
                

            

class NamedEntityRecognizer(ViterbiDecoder):
    """命名实体识别器
    """
    
    def __init__(self, trans, model, tokenizer, id2label,starts=None, ends=None):
        self.trans = trans
        self.num_labels = len(trans)
        self.non_starts = []
        self.non_ends = []
        self.model = model
        self.tokenizer = tokenizer
        self.id2label = id2label
        
        
        if starts is not None:
            for i in range(self.num_labels):
                if i not in starts:
                    self.non_starts.append(i)
        if ends is not None:
            for i in range(self.num_labels):
                if i not in ends:
                    self.non_ends.append(i)
    
    def recognize(self, text):
        tokens = self.tokenizer.tokenize(text)
        while len(tokens) > 512:
            tokens.pop(-2)
        mapping = self.tokenizer.rematch(text, tokens)
        token_ids = self.tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        nodes = self.model.predict([[token_ids], [segment_ids]])[0]
        labels, score, scoresarray = self.decode(nodes)
        print(score)
        print(scoresarray)

        print(labels.shape)
        print(labels[0])
        
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], self.id2label[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False

        # return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], {l:sum(scoresarray[mapping[w[0]][0]:mapping[w[-1]][-1] + 1])})
        #         for w, l in entities], score
        return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1],
                 l) for w, l in entities], [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], {l:sum(scoresarray[mapping[w[0]][0]:mapping[w[-1]][-1] + 1])})
                for w, l in entities], score



def evaluate(data,NER):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data):
        text = ''.join([i[0] for i in d])
        R = set(NER.recognize(text)[0])
        T = set([tuple(i) for i in d if i[1] != 'O'])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall



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

    NER = NamedEntityRecognizer(K.eval(trainmodel.CRF.trans), trainmodel.model, tokenizer, schema_dict['id2label'], starts=[0], ends=[0])
    eval_result = evaluate(valid_data, NER)
    
    return eval_result

def predict(data, model_path):
    result = []
    
    with open(os.path.join(model_save_path,'config.json')) as f:
        config = json.load(f)

    tokenizer = Tokenizer(train_param['dict_path'], do_lower_case=True)
    Pmodel = TagModel(config)
    Pmodel.model.load_weights(os.path.join(model_save_path,'best_model.weights'))
    
    NER = NamedEntityRecognizer(K.eval(Pmodel.CRF.trans), Pmodel.model, tokenizer, schema_dict['id2label'], starts=[0], ends=[0])
    
    for d in tqdm(data):
        text = ''.join([i[0] for i in d])
        result.append(NER.recognize(text)[1:])

    return result

if __name__ == '__main__':
    
    train_param = {}
    # 基本信息
    train_param['maxlen'] = 64
    train_param['epochs'] = 50
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
    
    #test parse data
    train_data,valid_data,schema_dict = load_data()
    eval_result = train(train_param, model_save_path)
    
#    realdata=[]
#    predictdata = []
#    for text, entities in valid_data:
#        realdata.append(ziptextlabels(text, entities))
    
    Presult = predict(valid_data, model_save_path)
    
#    with open(os.path.join(model_save_path,'config.json')) as f:
#        config = json.load(f)
#
#    tokenizer = Tokenizer(train_param['dict_path'], do_lower_case=True)
#    trainmodel,tokenizer = train(train_param, model_save_path)
    
#    X, Y, Z = 1e-10, 1e-10, 1e-10
#    for text, arguments in tqdm(valid_data):
#        R = transform2geshi(len(text), arguments, schema_dict['label2id'])
#        tokens = tokenizer.tokenize(text)
#        while len(tokens) > 512:
#            tokens.pop(-2)
#        token_ids = tokenizer.tokens_to_ids(tokens)
#        segment_ids = [0] * len(token_ids)
#        nodes = trainmodel.model.predict([[token_ids], [segment_ids]])[0]
#        trans = K.eval(trainmodel.CRF.trans)
#        P = viterbi_decode(nodes, trans, schema_dict['num_labels'])[1:]
#        
##        T = extract_arguments(trainmodel, tokenizer, schema_dict, text)
#        print(P)
#        X += len(R & P)
#        Y += len(R)
#        Z += len(P)
#    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    
    


#    model.load_weights('best_model.weights')
#    predict_to_file('./datasets/test1.json', 'ee_pred.json')