from utils import extend_maps, prepocess_data_for_lstmcrf
from evaluate import bilstm_train_and_eval
import pickle
import numpy as np
from gensim.models import Word2Vec
import torch
import random


def main():
    """训练模型，评估结果"""
    # 读取数据
    print("读取数据...")
    # 训练评估ｈｍｍ模型
    # print("正在训练评估HMM模型...")
    with open('corpus_word2id', 'rb') as f:
        word2id = pickle.load(f)
    with open('tag2id', 'rb') as f:
        tag2id = pickle.load(f)
    # print('word2id',word2id)
    # print('tag2id',tag2id)
    train_word_lists = []
    train_tag_lists = []
    f = open('normal_daguan_train.txt')
    datas = f.readlines()
    f.close()
    for data in datas:
        temps = data.strip().split('|||')
        words = temps[0]
        tags = temps[1]
        train_word_lists.append(words.split(' '))
        train_tag_lists.append(tags.split(' '))
    test_word_lists = []
    test_tag_lists = []
    f = open('normal_daguan_test.txt')
    datas = f.readlines()
    f.close()
    for data in datas:
        words = data.strip()
        test_word_lists.append(words.split(' '))
        test_tag_lists.append([None for i in range(len(words))])
    '''前3000数据集划分'''
    dev_word_lists = train_word_lists[:1500]
    dev_tag_lists = train_tag_lists[:1500]
    train_word_lists = train_word_lists[1500:]
    train_tag_lists = train_tag_lists[1500:]
    '''后3000数据集划分'''

    print("正在训练评估Bi-LSTM+CRF模型...")
    # 如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)
    crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
    # 还需要额外的一些数据处理
    train_word_lists, train_tag_lists = prepocess_data_for_lstmcrf(
        train_word_lists, train_tag_lists
    )


    dev_word_lists, dev_tag_lists = prepocess_data_for_lstmcrf(
        dev_word_lists, dev_tag_lists,
    )
    test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(
        test_word_lists, test_tag_lists,
    )
    model = Word2Vec.load('charEmbedding_300dim')
    weights = np.zeros(shape=(len(crf_word2id), 300))
    for word in word2id.keys():
        if word in model.wv.vocab:
            weights[word2id[word]] = model[word]
        else:
            weights[word2id[word]] = np.random.uniform(-0.1, 0.1, size=(300,))
    lstmcrf_pred = bilstm_train_and_eval(
        (train_word_lists, train_tag_lists),
        (dev_word_lists, dev_tag_lists),
        (test_word_lists, test_tag_lists),
        # (dev_word_lists, dev_tag_lists),
        weights, crf_word2id, crf_tag2id
    )


if __name__ == "__main__":
    seed = 66666
    print('seed:', seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    main()
