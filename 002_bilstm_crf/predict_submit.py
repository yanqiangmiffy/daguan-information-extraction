#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: predict_submit.py 
@time: 2019-07-13 11:33
@description:
"""
import numpy as np
from prepare_data import bulid_dataset
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils

# 2 构建数据集
n_words, n_tags, max_len, words, tags, \
x_train, y_train, x_test, y_test, x_valid, y_valid = bulid_dataset()


def predict(x, y, pred_path):
    """
    利用已经训练好的数据进行预测
    :return:
    """
    # 重新初始化模型，构建配置信息，和train部分一样
    input = Input(shape=(max_len,))
    model = Embedding(input_dim=n_words + 1, output_dim=100,
                      input_length=max_len, mask_zero=True)(input)  # 20-dim embedding
    model = Bidirectional(LSTM(units=50, return_sequences=True,
                               recurrent_dropout=0.1))(model)  # variational biLSTM
    model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
    crf = CRF(n_tags)  # CRF layer
    out = crf(model)  # output
    model = Model(input, out)

    # 恢复权重
    save_load_utils.load_all_weights(model, filepath="models/bilstm-crf.h5")

    # 预测
    i = 40
    # p = model.predict(np.array([x_train[i]]))
    # p = np.argmax(p, axis=-1)
    # true = np.argmax(y_train[i], -1)
    # print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
    # print(30 * "=")
    # for w, t, pred in zip(x_train[i], true, p[0]):
    #     print("{:15}: {:5} {}".format(words[w], tags[t], tags[pred]))

    y_preds = model.predict(np.array(x))
    with open(pred_path, 'w', encoding='utf-8') as f:
        for input_x, y_true, y_pred in zip(x, y, y_preds):
            y_true = np.argmax(y_true, -1)
            y_pred = np.argmax(y_pred, axis=-1)
            for w, t, pred in zip(input_x, y_true, y_pred):
                # print("{:15}: {:5} {}".format(words[w], tags[t], tags[pred]))
                if words[w] != 'ENDPAD':
                    f.write(words[w] + '\t' + tags[t] + '\t' + tags[pred] + '\n')
            f.write('\n')


def submit(pred_path, result_path):
    f1 = open(pred_path)
    all_data = f1.readlines()

    rs_all_data1 = []
    rs_str = ''
    pre_tag = ''
    rs_all_data = []
    idx = 0
    all_data.append('\n')
    for str in all_data:
        str = str.strip()
        if len(str) == 0:
            idx = idx + 1
            if len(rs_str) > 0:
                if rs_str.startswith('_'):
                    rs_str = rs_str[1:]
                if rs_str.endswith('_'):
                    rs_str = rs_str[:-2]
                rs_str = rs_str + '/' + pre_tag
                rs_all_data.append(rs_str)
                rs_str = ''
                pre_tag = ''

                rs_all_data1.append(rs_all_data)
                rs_all_data = []
            else:
                if len(rs_all_data) > 0:
                    rs_all_data1.append(rs_all_data)
                    rs_all_data = []
        else:
            sss = str.split("\t")

            if len(rs_str) == 0:
                if '_' in sss[2]:
                    sss[2] = sss[2].split("_")[1]
                rs_str = rs_str + '_' + sss[0]
                pre_tag = sss[2]
            else:
                if sss[2].startswith('B'):
                    if '_' in sss[2]:
                        sss[2] = sss[2].split("_")[1]
                    if len(rs_str) == 0:
                        rs_str = rs_str + '_' + sss[0]
                        pre_tag = sss[2]
                    else:
                        if rs_str.startswith('_'):
                            rs_str = rs_str[1:]
                        if rs_str.endswith('_'):
                            rs_str = rs_str[:-2]
                        rs_str = rs_str + '/' + pre_tag
                        rs_all_data.append(rs_str)
                        rs_str = ''
                        rs_str = rs_str + '_' + sss[0]
                        pre_tag = sss[2]
                else:
                    if '_' in sss[2]:
                        sss[2] = sss[2].split("_")[1]
                    if pre_tag == sss[2]:
                        rs_str = rs_str + '_' + sss[0]
                    else:
                        if rs_str.startswith('_'):
                            rs_str = rs_str[1:]
                        if rs_str.endswith('_'):
                            rs_str = rs_str[:-2]
                        rs_str = rs_str + '/' + pre_tag
                        rs_all_data.append(rs_str)
                        rs_str = ''
                        rs_str = rs_str + '_' + sss[0]
                        pre_tag = sss[2]

    print(idx)
    print(len(rs_all_data1))
    f2 = open(result_path, 'w')
    for ss in rs_all_data1:
        f2.write('  '.join(ss).lower() + "\n")
    f2.close()


if __name__ == '__main__':
    predict(x_train, y_train, pred_path='submit/train_pred.txt')
    predict(x_valid, y_valid, pred_path='submit/valid_pred.txt')
    predict(x_test, y_test, pred_path='submit/test_pred.txt')
    submit('submit/train_pred.txt','submit/train_result.txt')
    submit('submit/valid_pred.txt','submit/valid_result.txt')
    submit('submit/test_pred.txt','submit/test_result.txt')
