#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincyqiang
@license: Apache Licence
@file: 002_bilstm_crf.py
@time: 2019-07-12 22:58
@description:
"""
import numpy as np
import pandas as pd
from prepare_data import bulid_dataset
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils

import matplotlib.pyplot as plt

plt.style.use("ggplot")

# 2 构建数据集
n_words, n_tags, max_len, words, tags, \
x_train, y_train, x_test, y_test,x_valid,y_valid = bulid_dataset()


def train():
    input = Input(shape=(max_len,))
    model = Embedding(input_dim=n_words + 1, output_dim=100,
                      input_length=max_len, mask_zero=True)(input)  # 20-dim embedding
    model = Bidirectional(LSTM(units=50, return_sequences=True,
                               recurrent_dropout=0.1))(model)  # variational biLSTM
    model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
    crf = CRF(n_tags)  # CRF layer
    out = crf(model)  # output
    model = Model(input, out)

    model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
    model.summary()

    history = model.fit(x_train, np.array(y_train), batch_size=64, epochs=5,
                        validation_split=0.1, verbose=1)
    save_load_utils.save_all_weights(model, filepath="models/bilstm-crf.h5")

    hist = pd.DataFrame(history.history)
    print(hist)
    plt.figure(figsize=(12, 12))
    plt.plot(hist["crf_viterbi_accuracy"])
    plt.plot(hist["val_crf_viterbi_accuracy"])
    plt.show()


if __name__ == '__main__':
    train()
