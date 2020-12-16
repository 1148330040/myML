# *- coding: utf-8 -*-

# =================================
# time: 2020.8.25
# author: @tangzhilin
# function: Xlnet_model fit
# =================================

import os
import numpy as np
import yaml

from collections import namedtuple
from datetime import datetime

from keras_xlnet.backend import keras
from keras_bert.layers import Extract
from keras_xlnet import load_trained_model_from_checkpoint, ATTENTION_TYPE_BI

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from transformers import AutoTokenizer

# 调用xlnet 相关模型参数

with open('../config/config.yml', 'r', encoding='utf-8') as f:
    xlnet_config = yaml.safe_load(f)
    xlnet_config = xlnet_config['xlnet_model']

EPOCH = xlnet_config['epoch']
BATCH_SIZE = xlnet_config['batch_size']
TEXT_LEN = xlnet_config['text_len']
MEMORY_LEN = xlnet_config['memory_len']
LEARNING_RATE = xlnet_config['lr']
# print(EPOCH, BATCH_SIZE, TEXT_LEN, MEMORY_LEN, LEARNING_RATE)
pretrained_path = '../models/Xlnet_model/'

PretrainedPaths = namedtuple('PretrainedPaths', ['config', 'model', 'vocab'])

config_path = os.path.join(pretrained_path, 'xlnet_config.json')
model_path = os.path.join(pretrained_path, 'xlnet_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'spiece.model')
paths = PretrainedPaths(config_path, model_path, vocab_path)
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-xlnet-base")

time_month = datetime.now().month
time_day = datetime.now().day
log_dir = '../Transformer/logs/xlnet-{}-{}-log'.format(time_month, time_day)
checkpoint_path = '../Transformer/models/xlnet-{}-{}-model/xlnet_model.h5'.format(time_month, time_day)


class DataSequence(keras.utils.Sequence):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return (len(self.y) + BATCH_SIZE - 1) // BATCH_SIZE

    def __getitem__(self, index):
        s = slice(index * BATCH_SIZE, (index + 1) * BATCH_SIZE)
        return [item[s] for item in self.x], self.y[s]


def generate_sequence(df):
    """
    :param df:
    :return:
    """
    tokens, classes = [], []
    for _, row in df.iterrows():
        # text, solve
        text, cls = row['text'], row['solve']
        encoded = tokenizer.encode(text)[:TEXT_LEN - 1]
        encoded = [5] * (TEXT_LEN - 1 - len(encoded)) + encoded + [3]
        tokens.append(encoded)
        classes.append(int(cls))
    tokens, classes = np.array(tokens), np.array(classes)
    segments = np.zeros_like(tokens)
    segments[:, -1] = 1
    lengths = np.zeros_like(tokens[:, :1])
    return DataSequence([tokens, segments, lengths], classes)


def build_model():
    model = load_trained_model_from_checkpoint(
        config_path=paths.config,
        checkpoint_path=paths.model,
        batch_size=BATCH_SIZE,
        memory_len=MEMORY_LEN,
        target_len=TEXT_LEN,
        in_train_phase=False,
        attention_type=ATTENTION_TYPE_BI
    )

    # 加载预训练权重
    # Build classification model
    last = model.output
    extract = Extract(index=-1, name='Extract')(last)
    dense = keras.layers.Dense(units=768, name='Dense')(extract)
    norm = keras.layers.BatchNormalization(name='Normal')(dense)
    output = keras.layers.Dense(units=2, activation='softmax', name='Softmax')(norm)
    model = keras.models.Model(inputs=model.inputs, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'],
    )

    return model


def fit_model(train, test, valid):
    import pandas as pd
    model = build_model()

    train_ = generate_sequence(train)
    test_ = generate_sequence(test)
    valid_ = generate_sequence(valid)

    # 模型训练
    model.fit_generator(
        generator=train_,
        validation_data=valid_,
        epochs=EPOCH,
        callbacks=[TensorBoard(log_dir)]
    )

    model.save_weights(filepath=checkpoint_path)
    return checkpoint_path


# def xlnet_predict_datas(datas, path):
#     # 待预测的数据
#     datas = generate_sequence(datas)
#     model = build_model()
#     model.load_weights(path)
#     result = model.predict(datas)
#     return result



