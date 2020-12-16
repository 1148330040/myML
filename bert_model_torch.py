# *- coding: utf-8 -*-

# =================================
# time: 2020.8.9
# author: @tangzhilin
# function: Bert模型训练流程
# =================================

import pandas as pd
import numpy as np
import logging
import yaml
import matplotlib.pyplot as plt

import torch
from torch import cuda, tensor, long, float, save, load, no_grad, squeeze, unsqueeze
from torch.nn import Linear, LSTM, Dropout, BCEWithLogitsLoss, Module
from torch.cuda import device
from torch.nn.functional import sigmoid, softmax
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer, BertModel

from sklearn.metrics import roc_curve, classification_report, auc, accuracy_score
from datetime import datetime

logging.basicConfig(level=logging.ERROR)
pd.set_option('display.max_columns', None)

# 调用bert 相关模型参数
with open('../config/config.yml', 'r', encoding='utf-8') as f:
    bert_config = yaml.safe_load(f)
    bert_config = bert_config['bert_model']

EPOCHS = bert_config['epochs']
BATCH_SIZE = bert_config['batch_size']
SHUFFLE = bert_config['shuffle']
NUM_WORKS = bert_config['num_works']
LEARNING_RATE = bert_config['lr']
TEXT_LEN = bert_config['text_len']
MAX_TO_KEEP = bert_config['max_to_keep']

token = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")

time_month = datetime.now().month
time_day = datetime.now().day

checkpoint_path = '../Transformer/models/bert-{}-{}-model'.format(time_month, time_day)
log_dir = '../Transformer/logs/bert-{}-{}-log'.format(time_month, time_day)

devices = 'cuda' if cuda.is_available() else 'cpu'


class CustomDataset(Dataset):
    """
    function: 数据准备
    """
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe['text']
        self.targets = self.data['solve']
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        return {
            'ids': tensor(ids, dtype=long),
            'mask': tensor(mask, dtype=long),
            'token_type_ids': tensor(token_type_ids, dtype=long),
            'targets': tensor(self.targets[index], dtype=float)
        }


class BertClass(Module):
    """
    function: 模型
    ABC库用于给类提供一个抽象类的权限，具体的抽象方法还需要从abc里调用相关函数
    """
    def __init__(self):
        super(BertClass, self).__init__()
        # self.bert_1 = TFBertModel.from_pretrained("bert-base-chinese")
        self.bert_1 = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext')
        self.lstm = LSTM(768, 384)
        self.batch = torch.nn.BatchNorm1d(384)
        self.linear = Linear(384, 32)
        self.dropout = Dropout(0.5)
        self.linear_2 = Linear(32, 1)

    def forward(self, ids, mask, token_type_ids):
        _, output = self.bert_1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        # print("bert: ", output.size())
        output = self.lstm(unsqueeze(output, 1))[0]
        # squeeze(output, 1) 去除新增的维度
        output = self.batch(squeeze(output, 1))
        output = self.linear(output)
        output = self.dropout(output)
        output = self.linear_2(output)
        result = sigmoid(output)

        return result


def loss_fn(target, output):
    """
    function: 获取损失值
    :param target: label
    :param output: prediction
    :return: loss
    """
    loss = BCEWithLogitsLoss()
    output = squeeze(output)
    try:
        loss_value = loss(target, output)
    except:
        loss_value = loss(target, output[0])

    return loss_value


def accuracy_fn(target, output):
    """
    function: 获取准确值
    :param target: label
    :param output: prediction
    :return: accuracy
    """
    output = squeeze(torch.round(output))
    accuracy_value = accuracy_score(y_true=target, y_pred=output.detach().numpy())

    return accuracy_value


def train_fit(data_set):
    """
    function: 开启模型训练过程
    :return: Model, Log
    """
    model = BertClass()
    # model.to(devices)

    model.train()

    data_params = {'batch_size': BATCH_SIZE,
                   'shuffle': SHUFFLE,
                   'num_workers': NUM_WORKS}

    optimizer = Adam(params=model.parameters(), lr=1.5e-5)
    train_set = CustomDataset(data_set, token, TEXT_LEN)
    train_set_pt = DataLoader(train_set, **data_params)
    for epoch in range(5, EPOCHS+1):
        for _, data in enumerate(train_set_pt, 0):
            ids = data['ids'].to(dtype=long)
            mask = data['mask'].to(dtype=long)
            token_type_ids = data['token_type_ids'].to(dtype=long)
            targets = data['targets'].to(dtype=float)
            pred = model(ids, mask, token_type_ids)

            optimizer.zero_grad()
            loss = loss_fn(targets, pred)
            accuracy = accuracy_fn(targets, pred)
            if _ % 20 == 0:
                print("epoch is {}, step is {}, loss is {}, acuuracy is {}".format(
                    epoch, _, loss, accuracy
                ))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 2 == 0:
            # 模型每训练两轮保存一次
            print("fit epoch is: {}".format(epoch))
            save(model.state_dict(), checkpoint_path)


def validation(validata_set):
    val_model = BertClass()
    val_model.load_state_dict(load(checkpoint_path))
    val_model.to(devices)
    val_model.eval()

    validata_set = CustomDataset(validata_set, token, 60)
    validata_params = {'batch_size': 64, 'shuffle': True, 'num_workers': 0}
    validata_set_pt = DataLoader(validata_set, **validata_params)
    targets_all = []
    preds_all = []
    with no_grad():
        for _, data in enumerate(validata_set_pt, 0):
            ids = data['ids'].to(dtype=long)
            mask = data['mask'].to(dtype=long)
            token_type_ids = data['token_type_ids'].to(dtype=long)
            targets = data['targets'].to(dtype=float)
            outputs = val_model(ids, mask, token_type_ids)

            targets_all.extend(targets.cpu().detach().numpy().tolist())
            preds_all.extend(sigmoid(outputs).cpu().detach().numpy().tolist())

    print("val targets: ", targets_all, "val pred: ", preds_all)
    metrics_des = classification_report(y_true=targets_all, y_pred=preds_all)
    fpr, tpr, thresholds = roc_curve(y_true=targets_all, y_score=preds_all)
    auc_value = auc(fpr, tpr)
    print("all_des: \n", metrics_des)
    print("auc_value: \n", auc_value)
    plt.plot(fpr, tpr, marker='o')
    plt.show()


def fit_model(train, test):
    train.index = np.arange(len(train))
    test.index = np.arange(len(test))
    train_fit(data_set=train)
    validation(test)
    return checkpoint_path


train = pd.read_excel('../test_data/train.xlsx')
test = pd.read_excel('../test_data/test.xlsx')
train = train.sample(frac=0.99)
fit_model(train, test)
