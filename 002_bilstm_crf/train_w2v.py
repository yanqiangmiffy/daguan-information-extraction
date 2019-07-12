#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: train_w2v.py 
@time: 2019-07-12 23:00
@description:
"""
from collections import Counter


def analysis_data():
    """
    分析数据
    :return:
    """

    chars_list = []
    sentences_list = []
    with open('../datagrand/train.txt', 'r') as train:
        for sent in train:
            tmp_sent = []
            sent = sent.strip()
            for word in sent.split('  '):
                word = word.rstrip('/a').strip('/b').rstrip('/c').rstrip('/o')
                tmp_sent.extend(word.split('_'))
                chars_list.extend(word.split('_'))
            sentences_list.append(tmp_sent)

    with open('../datagrand/test.txt', 'r') as train:
        for sent in train:
            tmp_sent = []
            sent = sent.strip()
            for word in sent.split('  '):
                tmp_sent.extend(word.split('_'))
                chars_list.extend(word.split('_'))
                # for char in word.split('_'):
                #     chars_list.append(char)
            sentences_list.append(tmp_sent)
    # with open('../datagrand/corpus.txt', 'r') as train:
    #     for sent in train:
    #         sent = sent.strip()
    #         for word in sent.split('  '):
    #             for char in word.split('_'):
    #                 chars_list.append(char)
    char_freq = Counter(chars_list)
    print(char_freq.most_common(10))
    print(len(char_freq))

    # 句子长度
    print(sentences_list[3], len(sentences_list[3]))
    sen_lens = [len(s) for s in sentences_list]
    sent_len_freq = Counter(sen_lens)
    print(sent_len_freq)

    maxlen = max([len(s) for s in sentences_list])
    print('Maximum sequence length:', maxlen)
    import matplotlib.pyplot as plt
    plt.style.use("ggplot")
    plt.hist([len(s) for s in sentences_list], bins=50)
    plt.show()


analysis_data()
