# *- coding: utf-8 -*-
import os
from collections import namedtuple

import yaml
import tensorflow as tf
import pandas as pd
import numpy as np

from bert4keras.models import build_transformer_model
from keras_self_attention import SeqSelfAttention

from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from transformers import BertTokenizer
from datetime import datetime
from sklearn.model_selection import train_test_split


# 调用bert 相关模型参数
with open('../config/config.yml', 'r', encoding='utf-8') as f:
    new_bert_config = yaml.safe_load(f)
    new_bert_config = new_bert_config['bert_model']

EPOCHS = new_bert_config['epochs']
BATCH_SIZE = new_bert_config['batch_size']
SHUFFLE = new_bert_config['shuffle']
NUM_WORKS = new_bert_config['num_works']
LEARNING_RATE = new_bert_config['lr']
TEXT_LEN = new_bert_config['text_len']
MAX_TO_KEEP = new_bert_config['max_to_keep']

MODEL_NAME = "hfl/chinese-roberta-wwm-ext-large"
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

time_month = datetime.now().month
time_day = datetime.now().day
checkpoint_path = '../Transformer/models/bert-{}-{}-model'.format(time_month, time_day)


class DataSequence(tf.keras.utils.Sequence):

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


pretrained_path = '../models/BERT_model'

PretrainedPaths = namedtuple('PretrainedPaths', ['config', 'model', 'vocab'])

config_path = os.path.join(pretrained_path, 'bert_config.json')
model_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')
paths = PretrainedPaths(config_path, model_path, vocab_path)


def build_model():
    bert = build_transformer_model(
        config_path=paths.config,
        checkpoint_path=paths.model,
        model='bert',
        return_keras_model=False
    )
    output = LSTM(32, return_sequences=True)(bert.model.output)
    output = SeqSelfAttention(attention_activation='sigmoid')(output)
    output = Lambda(lambda x: x[:, 0], name='CLS-token-Txt')(output)
    output = Dense(32)(output)
    output = Dropout(0.5)(output)
    output = Dense(32)(output)
    output = Dense(units=2, activation='softmax', name='Softmax')(output)

    model = Model(inputs=bert.inputs, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=1.4e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )

    return model


def fit_model(train, test, valid):
    train = pd.read_excel('../test_data/train.xlsx')
    test = pd.read_excel('../test_data/test.xlsx')
    train = generate_sequence(train)
    valid = generate_sequence(valid)
    test = generate_sequence(test)

    model = build_model()
    model.fit_generator(
        generator=train,
        validation_data=valid,
        epochs=10
    )

    model.save_weights(checkpoint_path)
