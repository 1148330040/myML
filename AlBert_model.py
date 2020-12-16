# *- coding: utf-8 -*-

# =================================
# time: 2020.8.6
# author: @tangzhilin
# function: AlBert模型训练流程
# =================================
from collections import namedtuple
from datetime import datetime

from bert4keras.backend import keras
from bert4keras.snippets import DataGenerator, sequence_padding
from bert4keras.models import build_transformer_model, Lambda
from bert4keras.tokenizers import Tokenizer
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam

import yaml
import tensorflow as tf
import bert4keras
import os
print(tf.__version__)
print(bert4keras.__version__)

# 调用albert 先关模型参数
with open('../config/config.yml', 'r', encoding='utf-8') as f:
    albert_config = yaml.safe_load(f)
    albert_config = albert_config['albert_model']

TEXT_LEN = albert_config['text_len']
BATCH_SIZE = albert_config['batch_size']
EPOCH = albert_config['epoch']
LEARNING_RATE = albert_config['lr']
VERBOSE = albert_config['verbose']

pretrained_path = '../models/albert_small_cn/'

PretrainedPaths = namedtuple('PretrainedPaths', ['config', 'model', 'vocab'])

config_path = os.path.join(pretrained_path, 'albert_config_small_google.json')
model_path = os.path.join(pretrained_path, 'albert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')
paths = PretrainedPaths(config_path, model_path, vocab_path)


time_month = datetime.now().month
time_day = datetime.now().day
log_dir = '../Transformer/logs/albert-{}-{}-log'.format(time_month, time_day)
checkpoint_path = '../Transformer/models/albert-{}-{}-model/albert_weights'.format(time_month, time_day)

tokenizer = Tokenizer(paths.vocab, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=True):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=TEXT_LEN)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def build_model():
    bert = build_transformer_model(
        config_path=paths.config,
        checkpoint_path=paths.model,
        model='albert',
        return_keras_model=False
    )
    output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
    output = keras.layers.Dense(
        units=2,
        activation='softmax',
        kernel_initializer=bert.initializer)(output)
    model = keras.models.Model(bert.model.input, output)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics=['accuracy'],
    )
    # print(model.summary())
    return model


def fit_model(train, test, valid):
    """
    :param train: 训练数据
    :param test:  测试数据
    :param valid: 验证数据
    :return: 模型最新的保存地址
    """
    train = data_generator(train, BATCH_SIZE)
    valid = data_generator(valid, BATCH_SIZE)
    test = data_generator(test, BATCH_SIZE)

    model = build_model()

    model.fit_generator(
        train.forfit(),
        steps_per_epoch=len(train),
        epochs=EPOCH,
        verbose=VERBOSE,
        callbacks=[TensorBoard(log_dir)]
    )

    model.save_weights(checkpoint_path)

    return checkpoint_path
