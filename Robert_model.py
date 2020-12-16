# *- coding: utf-8 -*-

# =================================
# time: 2020.9.17
# author: @tangzhilin
# function: Robert模型训练, 用于进行标签输出
# =================================

import numpy as np
import pandas as pd

from datetime import datetime

from tensorflow.keras.layers import Dropout, Activation, Flatten, Conv1D
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

from bert4keras.models import build_transformer_model
from bert4keras.snippets import DataGenerator

from transformers import BertTokenizer

import yaml
import logging
logging.basicConfig(level=logging.ERROR)

# 调用robert 相关模型参数
with open('../config/config.yml', 'r', encoding='utf-8') as f:
    robert_config = yaml.safe_load(f)
    robert_config = robert_config['robert_model']

TEXT_LEN = robert_config['text_len']
BATCH_SIZE = robert_config['batch_size']
EPOCH = robert_config['epoch']
LEARNING_RATE = robert_config['lr']
VERBOSE = robert_config['verbose']

model_path = '../models/Robert_model/bert_model.ckpt'
config_path = '../models/Robert_model/bert_config.json'
vocab_path = '../models/Robert_model/vocab.txt'

tokenizer = BertTokenizer.from_pretrained(vocab_path, do_lower_case=True)

time_month = datetime.now().month
time_day = datetime.now().day
log_dir = '../Transformer/logs/robert-{}-{}-log'.format(time_month, time_day)
checkpoint_path = '../Transformer/models/robert-{}-{}-model/robert_model.h5'.format(time_month, time_day)


class data_generator(DataGenerator):
    """数据生成器"""
    def __iter__(self, random=True):
        ids, mask, tids, starts, ends = [], [], [], [], []
        for is_end, (text, label) in self.sample(random):
            start = np.zeros(TEXT_LEN)
            end = np.zeros(TEXT_LEN)
            encode_result = tokenizer.encode_plus(
                text,
                max_length=TEXT_LEN,
                pad_to_max_length=True,
                return_token_type_ids=True,
                add_special_tokens=True
            )
            ids.append(encode_result['input_ids'])
            mask.append(encode_result['attention_mask'])
            tids.append(encode_result['token_type_ids'])
            if label[0] >= 0 and label[1] > 0:
                start[label[0]] = 1
                end[label[1]] = 1
            starts.append(start)
            ends.append(end)
            if len(ids) == self.batch_size or is_end:
                ids = np.array(ids)
                mask = np.array(mask)
                tids = np.array(tids)
                starts = np.array(starts)
                ends = np.array(ends)
                yield [ids, mask, tids], [starts, ends]
                ids, mask, tids, starts, ends = [], [], [], [], []


def build_model():

    robert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=model_path,
        model='roberta',
        return_keras_model=False
    )
    # outputs = Lambda(lambda x: x[:, 0], name='CLS-token')(robert.model.output)
    outputs = robert.model.output
    # 获取start
    dropout1 = Dropout(0.1)(outputs)
    dense1 = Conv1D(1, 1)(dropout1)
    flatten1 = Flatten()(dense1)
    activation1 = Activation(activation='softmax')(flatten1)
    # 获取end
    dropout2 = Dropout(0.1)(outputs)
    dense2 = Conv1D(1, 1)(dropout2)
    flatten2 = Flatten()(dense2)
    activation2 = Activation(activation='softmax')(flatten2)

    model = Model(inputs=robert.model.input, outputs=[activation1, activation2])

    optimizer = Adam(learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy'
    )

    return model


def fit_model(datas):
    train, test = datas[:100], datas[100:]
    train = data_generator(train, BATCH_SIZE)
    test = data_generator(test, BATCH_SIZE)
    model = build_model()
    model.fit_generator(
        train.forfit(),
        steps_per_epoch=len(train),
        epochs=EPOCH,
        verbose=VERBOSE,
        callbacks=[TensorBoard(log_dir)]
    )

    model.save_weights(filepath=checkpoint_path)

    return checkpoint_path

