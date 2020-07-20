#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 15:10:57 2020

@author: liang
"""

import numpy as np

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
        nodes[0, self.non_starts] -= np.inf
        nodes[-1, self.non_ends] -= np.inf

        # 动态规划
        labels = np.arange(self.num_labels).reshape((1, -1))
        scores = nodes[0].reshape((-1, 1))
        paths = labels
        scoresarray = []
        pathscore = []
        totalscore = 0.0
        for l in range(1, len(nodes)):
            M = scores + self.trans + nodes[l].reshape((1, -1))
            idxs = M.argmax(0)
            scores = M.max(0).reshape((-1, 1))
            paths = np.concatenate([paths[:, idxs], labels], 0)
            scoresarray.append((max(scores)[0] / sum(scores))[0])

            totalscore.append(max(scores)[0])
            totalscore += sum(scores)

        # 最优路径
        return paths[:, scores[:, 0].argmax()], sum(pathscore)/totalscore, scoresarray