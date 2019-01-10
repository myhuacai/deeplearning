#!/usr/bin/env python
# -*- coding:utf-8 -*-

import codecs
import collections
import tensorflow as tf
from operator import itemgetter

'''
 先按照词频顺序为每个词汇分配一个编号，然后将词汇表保存到一个独立的covab文件中
'''
# RAW_DATA = "./data/ptb/ptb.train.txt"
# VOCAB_OUTPUT = './data/ptb/ptb.vocab'
#
# # 统计单词出现的频率
# counter = collections.Counter()
# with codecs.open(RAW_DATA,'r','utf-8') as f:
#     for line in f :
#         for word in line.strip().split():
#             counter[word] += 1
#         # print(counter)
#         # break
# # 按照词频顺序对单词进行排序,排好序后是放在一个list中的
# sorted_word_to_cnt = sorted(counter.items(),key=itemgetter(1),reverse=True)
# # # 取前10个验证下
# # print(sorted_word_to_cnt[:10])
# # 排序好的单词，只取单词，舍弃词频元素
# sorted_words = [x[0] for x in sorted_word_to_cnt]
# # 加入句子结束符 <eos>
# sorted_words = ['<eos>'] + sorted_words # + ['<eos>']
#
# # sorted_words = ['<eos>','<unk>','<sos>'] + sorted_words
# # if len(sorted_words) > 10000:
# #     sorted_words = sorted_words[:10000]
# # 将排好序的words存储到本地文件中,词汇表
# with codecs.open(VOCAB_OUTPUT,'w','utf-8') as file_output:
#     for word in sorted_words:
#         file_output.write(word + '\n')

'''
在确定了词汇表之后，再将训练文件、测试文件等都根据词汇文件转化为单词编号
'''

import sys

RAW_DATA = "./data/ptb/ptb.train.txt"
VOCAB = './data/ptb/ptb.vocab'
OUTPUT_DATA = "./data/ptb/ptb.train"

# 读取词汇表，并建立词汇到单词编号的映射
with codecs.open(VOCAB,'r','utf-8') as f_vocab:
    vocab = [w.strip() for w in f_vocab.readlines()]
# 词汇编码
word_to_id = {k:v for k,v in zip(vocab,range(len(vocab)))}

# 获取词汇id 如果出现了被删除的低频此，则替换为"<unk>"
def get_id(word):
    return word_to_id[word] if word in word_to_id else word_to_id['unk']
fin = codecs.open(RAW_DATA,'r','utf-8')
fout = codecs.open(OUTPUT_DATA,'w','utf-8')
for line in fin:
    # 读取单词并添加<eos>结束符
    words = line.strip().split() + ['<eos>']
    # 将每个单词替换为词汇表中的编号
    out_line = " ".join(str(get_id(w)) for w in words) + '\n'
    # 写入到文件
    fout.write(out_line)
fin.close()
fout.close()



