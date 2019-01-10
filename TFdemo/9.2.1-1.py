#!/usr/bin/env python
# -*- coding:utf-8 -*-

import codecs
import collections
import tensorflow as tf
from operator import itemgetter



RAW_DATA = "./data/ptb/ptb.train.txt"

VOCAB_OUTPUT = './data/ptb/ptb.vocab'

# 统计单词出现的频率
counter = collections.Counter()
with codecs.open(RAW_DATA,'r','utf-8') as f:
    for line in f :
        for word in line.strip().split():
            counter[word] += 1
# 按照词频顺序对单词进行排序
sorted_word_to_cnt = sorted(counter.items(),key=itemgetter(1),reverse=True)
# 排序好的单词
sorted_words = [x[0] for x in sorted_word_to_cnt]

sorted_words = ['<eos>'] + sorted_words # + ['<eos>']

# sorted_words = ['<eos>','<unk>','<sos>'] + sorted_words
# if len(sorted_words) > 10000:
#     sorted_words = sorted_words[:10000]

with codecs.open(VOCAB_OUTPUT,'w','utf-8') as file_output:
    for word in sorted_words:
        file_output.write(word + '\n')


