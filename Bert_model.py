# *- coding: utf-8 -*-

# =================================
# time: 2020.8.9
# author: @tangzhilin
# function: Bert模型训练流程
# =================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import yaml

from tensorflow.keras.layers import Conv1D, Dropout, Activation, Flatten, LSTM, Dense, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow import expand_dims, device, summary, GradientTape, round
from tensorflow import train, convert_to_tensor, round, function, distribute
from keras_bert.layers import Extract


from keras_self_attention import SeqSelfAttention
from tensorflow.python.keras.layers import BatchNormalization

from torch import cuda, tensor, long, float
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, TFBertModel, BertModel
from datetime import datetime
from sklearn.metrics import roc_curve, classification_report, auc
from abc import ABC

logging.basicConfig(level=logging.ERROR)
pd.set_option('display.max_columns', None)
strategy = distribute.MirroredStrategy()

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

# token = BertTokenizer.from_pretrained("bert-base-chinese")
token = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")


time_month = datetime.now().month
time_day = datetime.now().day

checkpoint_path = '../Transformer/models/bert-{}-{}-model'.format(time_month, time_day)
log_dir = '../Transformer/logs/bert-{}-{}-log'.format(time_month, time_day)


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


class BertClass(Model, ABC):
    """
    function: 模型
    ABC库用于给类提供一个抽象类的权限，具体的抽象方法还需要从abc里调用相关函数
    """

    # Model 需要def call()调用
    def __init__(self):
        super(BertClass, self).__init__()
        # self.bert_1 = TFBertModel.from_pretrained("bert-base-chinese")
        self.bert_1 = TFBertModel.from_pretrained('hfl/chinese-bert-wwm-ext')
        self.layer_1 = Dropout(0.3)
        self.layer_2 = Conv1D(1, 1)
        self.layer_3 = Flatten()
        self.layer_4 = Activation('sigmoid')

    @function
    def call(self, ids, mask, token_type_ids):
        print(self.bert_1(ids, attention_mask=mask, token_type_ids=token_type_ids))
        _, output_1 = self.bert_1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output = self.layer_1(output_1)
        output = expand_dims(output, 0)
        output = self.layer_2(output)
        output = self.layer_3(output)
        result = self.layer_4(output)

        return result


with strategy.scope():
    def loss_fn(target, output):
        """
        function: 获取损失值
        :param target: label
        :param output: prediction
        :return: loss
        """
        loss = BinaryCrossentropy(from_logits=True, name='binary_crossentropy')
        try:
            loss_value = loss(y_true=target, y_pred=output)
        except:
            loss_value = loss(y_true=target, y_pred=output[0])

        return loss_value


with strategy.scope():
    def accuracy_fn(target, output):
        """
        function: 获取准确值
        :param target: label
        :param output: prediction
        :return: accuracy
        """
        accuracy = Accuracy()
        output = round(output)
        accuracy.update_state(y_true=target, y_pred=output)
        accuracy_value = accuracy.result().numpy()
        return accuracy_value


with strategy.scope():
    def train_fit(data_set):
        """
        function: 开启模型训练过程
        :return: Model, Log
        """
        summary_writer = summary.create_file_writer(log_dir)

        data_params = {'batch_size': BATCH_SIZE,
                       'shuffle': SHUFFLE,
                       'num_workers': NUM_WORKS}

        devices = 'cuda' if cuda.is_available() else 'cpu'
        device(devices)

        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.99)

        train_set = CustomDataset(data_set, token, TEXT_LEN)
        train_set_pt = DataLoader(train_set, **data_params)
        model = BertClass()
        ckpt = train.Checkpoint(transformer=model.trainable_variables,
                                optimizer=optimizer)

        ckpt_manager = train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=MAX_TO_KEEP)

        def train_step(model_, id_, mk_, type_ids_, optimizer_, target_):
            with GradientTape() as tp:
                y_pred = model_(id_, mk_, type_ids_)
                loss_value = loss_fn(target=target_, output=y_pred)
                # y_pred = [round(y_p) for y_p in y_pred]
                acc = accuracy_fn(target_, y_pred)
            gradient = tp.gradient(loss_value, model.trainable_variables)

            optimizer_.apply_gradients(zip(gradient, model.trainable_variables))

            return loss_value, np.array(acc).mean(), y_pred
        for epoch in range(1, EPOCHS+1):
            for _, batch_data in enumerate(train_set_pt):
                ids = convert_to_tensor(batch_data['ids'].detach().numpy())
                mask = convert_to_tensor(batch_data['mask'].detach().numpy())
                token_type_ids = convert_to_tensor(batch_data['token_type_ids'].detach().numpy())
                targets = convert_to_tensor(batch_data['targets'].detach().numpy())
                loss, accuracy, pred = train_step(model_=model, id_=ids, mk_=mask,
                                                  type_ids_=token_type_ids, optimizer_=optimizer,
                                                  target_=targets)

                if _ % 20 == 0 and _ > 0:
                    # 将loss和accuracy写入日志文件
                    # 日志每训练十批数据保存一次日志文件
                    print("epoch: {}, fit step: {}, loss: {}, accuracy: {}".format(
                        epoch, _, loss, accuracy
                    ))
                    print("epoch is {}, predict: {}".format(epoch, pred))

            if epoch % 2 == 0:
                # 模型每训练两轮保存一次
                ckpt_manager.save(check_interval=True)

                with summary_writer.as_default():
                    summary.scalar(name="loss_value_step:{}".format(epoch),
                                   data=loss, step=epoch)

                with summary_writer.as_default():
                    summary.scalar(name='accuracy_value_step:{}'.format(epoch),
                                   data=accuracy, step=epoch)

with strategy.scope():
    def validation(validata_set):
        val_model = BertClass()
        ckpt = train.Checkpoint(transformer=val_model)
        ckpt.restore(train.latest_checkpoint(checkpoint_path))
        # val_model.load_weights(checkpoint_path)
        # 恢复模型的训练检查点
        validata_set = CustomDataset(validata_set, token, 60)
        validata_params = {'batch_size': 64, 'shuffle': True, 'num_workers': 0}
        validata_set_tf = DataLoader(validata_set, **validata_params)
        targets_all = []
        preds_all = []
        for _, batch_data in enumerate(validata_set_tf):
            ids = convert_to_tensor(batch_data['ids'].detach().numpy())
            mask = convert_to_tensor(batch_data['mask'].detach().numpy())
            token_type_ids = convert_to_tensor(batch_data['token_type_ids'].detach().numpy())
            targets = batch_data['targets'].numpy().tolist()
            y_pred = val_model(ids, mask, token_type_ids)
            y_pred = round(y_pred).numpy().tolist()

            preds_all = preds_all + y_pred
            targets_all = targets_all + targets

        print("val targets: ", targets_all, "val pred: ", preds_all)
        metrics_des = classification_report(y_true=targets_all, y_pred=preds_all)
        fpr, tpr, thresholds = roc_curve(y_true=targets_all, y_score=preds_all)
        auc_value = auc(fpr, tpr)
        print("all_des: \n", metrics_des)
        print("auc_value: \n", auc_value)
        plt.plot(fpr, tpr, marker='o')
        plt.show()


def fit_model(train, test, valid):
    train.index = np.arange(len(train))
    test.index = np.arange(len(test))
    valid.index = np.arange(len(valid))
    train_fit(data_set=train)
    validation(test)
    return checkpoint_path
