#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: prepare_data.py 
@time: 2019-07-12 23:00
@description:
"""
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


def transfer_format(open_path, save_path, flag='0train'):
    """
    将原始文件转为标注的csv
    :param file_path:     
    :return: 
    """

    df = open(save_path, 'w', encoding='utf-8')
    df.write('sent_idx' + ',' + 'char' + ',' + 'tag' + '\n')
    with open(open_path, 'r') as train:
        for index, sent in enumerate(train.readlines()):
            sent = sent.strip()
            for word in sent.split('  '):
                if word.endswith('/a'):
                    word = word.strip('/a')
                    for char_idx, char in enumerate(word.split('_')):
                        if char_idx == 0:
                            df.write(flag + str(index) + ',' + char + ',' + 'B_a' + '\n')
                        else:
                            df.write(flag + str(index) + ',' + char + ',' + 'I_a' + '\n')
                elif word.endswith('/b'):
                    word = word.strip('/b')
                    for char_idx, char in enumerate(word.split('_')):
                        if char_idx == 0:
                            df.write(flag + str(index) + ',' + char + ',' + 'B_b' + '\n')
                        else:
                            df.write(flag + str(index) + ',' + char + ',' + 'I_b' + '\n')
                elif word.endswith('/c'):
                    word = word.strip('/c')
                    for char_idx, char in enumerate(word.split('_')):
                        if char_idx == 0:
                            df.write(flag + str(index) + ',' + char + ',' + 'B_c' + '\n')
                        else:
                            df.write(flag + str(index) + ',' + char + ',' + 'I_c' + '\n')
                else:
                    word = word.strip('/o')
                    for char_idx, char in enumerate(word.split('_')):
                        df.write(flag + str(index) + ',' + char + ',' + 'O' + '\n')
    df.close()


class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["char"].values.tolist(),
                                                     s["tag"].values.tolist())]
        self.grouped = self.data.groupby("sent_idx").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


def bulid_dataset(train_path='data/train.csv',
                  test_path='data/test.csv',
                  dataset_dir='data/dataset.pkl'):
    """
    构建数据
    :param data:
    :return:
    """
    df_train = pd.read_csv(train_path)
    train_size = len(df_train['sent_idx'].unique())
    print(train_size)
    df_test = pd.read_csv(test_path)
    test_size = len(df_test['sent_idx'].unique())
    print(test_size)

    df = pd.concat([df_train, df_test], axis=0)
    df_size = len(df['sent_idx'].unique())
    print(df_size)
    df['char'] = df['char'].astype('str')
    if os.path.exists(dataset_dir):
        print("正在加载旧数据")
        with open(dataset_dir, 'rb') as in_data:
            data = pickle.load(in_data)
            return data

    # 标签和单词
    words = list(set(df["char"].values))
    words = [str(w) for w in words]
    words.append("ENDPAD")
    n_words = len(words)
    tags = list(set(df["tag"].values))
    n_tags = len(tags)
    getter = SentenceGetter(df)
    sentences = getter.sentences
    # print(len(sentences))
    print(len(sentences[:train_size]))
    print(len(sentences[train_size:]))
    # plt.hist([len(s) for s in sentences], bins=50)
    # plt.show()

    # 输入长度等长，统一设置为50
    max_len = 556
    word2idx = {str(w): i for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}

    # print(word2idx['3445'])
    # print(tag2idx['I_c'])

    print("训练数据")
    # 填充句子
    x_train = [[word2idx[w[0]] for w in s] for s in sentences[:train_size]]
    x_train = pad_sequences(maxlen=max_len, sequences=x_train, padding="post", value=n_words - 1)
    # print(x[1])

    # 填充标签
    y_train = [[tag2idx[w[1]] for w in s] for s in sentences[:train_size]]
    y_train = pad_sequences(maxlen=max_len, sequences=y_train, padding="post", value=tag2idx["O"])
    # print(y[1])

    # 将label转为categorial
    y_train = [to_categorical(i, num_classes=n_tags) for i in y_train]

    print("测试数据")
    x_test = [[word2idx[w[0]] for w in s] for s in sentences[train_size:]]
    x_test = pad_sequences(maxlen=max_len, sequences=x_test, padding="post", value=n_words - 1)

    # 填充标签
    y_test = [[tag2idx[w[1]] for w in s] for s in sentences[train_size:]]
    y_test = pad_sequences(maxlen=max_len, sequences=y_test, padding="post", value=tag2idx["O"])

    # 将label转为categorial
    y_test = [to_categorical(i, num_classes=n_tags) for i in y_test]
    with open(dataset_dir, 'wb') as out_data:
        pickle.dump([n_words, n_tags, max_len, words, tags,
                     x_train, y_train,
                     x_test, y_test],
                    out_data, pickle.HIGHEST_PROTOCOL)

    return n_words, n_tags, max_len, words, tags, \
           x_train, y_train, x_test, y_test


if __name__ == '__main__':
    open_path = '../datagrand/train.txt'
    save_path = 'data/train.csv'
    transfer_format(open_path, save_path, flag='0train')

    open_path = '../datagrand/test.txt'
    save_path = 'data/test.csv'
    transfer_format(open_path, save_path, flag='1test')

    data_dir = 'data/ner_dataset.csv'
    bulid_dataset()
