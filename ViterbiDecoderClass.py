#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 15:10:57 2020

@author: liang
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops, array_ops


class ViterbiDecoder(object):
    """Viterbi解码算法基类
    """
    def __init__(self, trans, starts=None, ends=None):
        self.trans = trans
        self.num_labels = len(trans)
        self.non_starts = []
        self.non_ends = []
        if starts is not None:
            for i in range(self.num_labels):
                if i not in starts:
                    self.non_starts.append(i)
        if ends is not None:
            for i in range(self.num_labels):
                if i not in ends:
                    self.non_ends.append(i)

    def decode(self, nodes):
        """nodes.shape=[seq_len, num_labels]
        """
        # 预处理
        nodes[0, self.non_starts] -= 0#np.inf
        nodes[-1, self.non_ends] -= 0#np.inf

        # 动态规划
        labels = np.arange(self.num_labels).reshape((1, -1))
        scores = nodes[0].reshape((-1, 1))
        paths = labels
        scoresarray = []
        pathscore = []
        totalscore = 0.0
        
#        best_score = math_ops.reduce_max(scores, axis=1)
#        viterbi_sequence, best_score = tf.contrib.crf.crf_decode(nodes, self.trans,
#                                                        nodes.shape[0])
        
#        sequences = tf.convert_to_tensor(inputs, dtype=self.dtype)
#        shape = tf.shape(inputs)
#        self.sequence_lengths = tf.ones(shape[0], dtype=tf.int32) * (shape[1])
#        
#        viterbi_sequence, _ = tf.contrib.crf.crf_decode(sequences, self.transitions,
#                                                        self.sequence_lengths)


        for item in scores + self.trans:
            item = [abs(x) for x in item]
            totalscore += sum(item)

        for l in range(1, len(nodes)):
            M = scores + self.trans + nodes[l].reshape((1, -1))
            idxs = M.argmax(0)
            scores = M.max(0).reshape((-1, 1))
            paths = np.concatenate([paths[:, idxs], labels], 0)
            scoresarray.append((max(scores)[0] / sum(scores))[0])

            pathscore.append(max(scores)[0])

            # for item in scores:
            #     totalscore += sum(item)


        # 最优路径
        # return paths[:, scores[:, 0].argmax()], sum(pathscore)/totalscore, pathscore, scoresarray

        return paths[:, scores[:, 0].argmax()], scoresarray