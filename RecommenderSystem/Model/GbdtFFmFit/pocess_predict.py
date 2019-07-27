# coding:gbk

'''
此文件的目的
对待预测数据做出最后的处理
'''

from Bmob.bmob import *
import numpy as np

def predict_result_to_Bmob(predict_result):
    bmob = Bmob("dbac9a625c262e782e519cb092c26f91", "661aceb04b058aeb0b5cb9e6feb5f24e")
    predict_result.index = np.arange(len(predict_result))
    result_dict = predict_result.to_dict()
    deliver_dict = dict()
    for i in range(len(predict_result)):
        for n in predict_result.columns:
            deliver_dict[n] = result_dict[n][i]
        bmob.insert(
            'Recommend',
            deliver_dict
        )

def end_processing(train):
    train.fillna(0, inplace=True)
    # 删除部分时间特征
    train.drop(['year', 'days'], axis=1, inplace=True)
    # 删除四种行为特征
    lag_columns = ['browse', 'collect', 'addcart', 'bought',
                   'user_days_browse_count', 'user_days_collect_count',
                   'user_days_addcart_count', 'user_days_bought_count',
                   'user_days_cate_probability', 'days',
                   'user_days_cate_browse_count', 'user_days_cate_collect_count',
                   'user_days_cate_addcart_count', 'user_days_cate_bought_count'
                   ]
    for f in lag_columns:
        try:
            train.drop(f, axis=1, inplace=True)
        except:
            pass
    return train

