# coding:gbk

import pandas as pd
import numpy as np

from TrainingData.GBDT_Data_Clean import get_fit_train_datas

import warnings
warnings.filterwarnings(action = 'ignore') #忽略警告


def get_fit_datas():
    datas = get_fit_train_datas(variable=1)
    return datas
datas = get_fit_datas()
# 4:1 划分训练数据和测试数据
size = round(len(datas)*0.8)
# 这里的特征名称因为要帮助新用户数据进行滞后值的构建
# 因此在数据清洗文件中不能直接删除
# 所以在训练模型时需要再次删除这几个特征列
lag_columns = ['user_days_browse_count', 'user_days_collect_count',
               'user_days_addcart_count', 'user_days_bought_count',
               'user_days_cate_probability', 'days',
               'user_days_cate_browse_count', 'user_days_cate_collect_count',
               'user_days_cate_addcart_count', 'user_days_cate_bought_count'
               ]

for feature in lag_columns:
    try:
        datas.drop(feature, axis=1, inplace=True)
    except:
        raise ValueError(
            "please check this feature whether surive" % feature
        )

datas.fillna(0, inplace=True)
# 将数据分为两类-> 产生行为4的和没有产生行为4的
datas.loc[datas['behavior_type']==2, 'behavior_type'] = 1
datas.loc[datas['behavior_type']==3, 'behavior_type'] = 1

trains = datas[:size]
train = trains.sample(frac=0.99)

train_label = train.pop('behavior_type')
train_label = train_label.apply(lambda x: 0 if x==1 else 1)
from imblearn.combine import SMOTEENN
train, train_label = SMOTEENN(0.33).fit_sample(train, train_label)

tests = datas[size:]
test = tests.sample(frac=0.99)
test_label = test.pop('behavior_type')
test_label = test_label.apply(lambda x: 0 if x==1 else 1)

from sklearn.model_selection import train_test_split

train, valida, train_label, valida_label = train_test_split(train, train_label,
                                                        test_size=0.3, random_state=1)
train = pd.DataFrame(train)
valida = pd.DataFrame(valida)
train_label = pd.Series(train_label)
valida_label = pd.Series(valida_label)
import joblib
from xgboost import XGBClassifier

XGB = XGBClassifier(
    n_estimators=105,
    learning_rate=1.3,
    max_depth=3,
    boosting_type='gbdt',
    booster='gbtree',
    subsample=0.7,
    max_delta_step=10  # 每个叶子输出的最大步长,通常用来解决数据不平衡的问题
)
XGB.fit(train.values, train_label.values)
# 保存模型
joblib.dump(XGB,'XGB_FFM.model')



# 这里的train, valida, test将来全都是从数据库内获取数据
def get_datas(model, train, valida, test):
    new_train = model.apply(train.values)
    new_valida = model.apply(valida.values)
    new_test = model.apply(test.values)
    return new_train, new_valida, new_test

new_train, new_valida, new_test = get_datas(XGB, train, valida, test)
# 训练集,验证集,测试集


from sklearn.metrics import classification_report,roc_auc_score
import xlearn as xl

ffm_train = xl.DMatrix(new_train, train_label)
ffm_valida = xl.DMatrix(new_valida, valida_label)
ffm_test = xl.DMatrix(new_test, test_label)

print("开始训练FFm模型")
ffm_model = xl.create_ffm()

ffm_model.setTrain(ffm_train)
ffm_model.setValidate(ffm_valida)
# 帮助提前收敛setValidate
ffm_model.setSign()
ffm_model.setNoBin()
ffm_model.setQuiet()

param = {
    'task':'binary',
    'lr':0.01,
    'lambda':0.0001,
    'k':2,
    'stop_window':3
}
ffm_model.fit(param, './model_dm.out')
print("FFm模型训练完毕！")
ffm_model.setTest(ffm_test)
res = ffm_model.predict("./model_dm.out")
print(pd.Series(test_label).value_counts())
print(pd.Series(res).value_counts())
print(roc_auc_score(test_label, res))
print(classification_report(test_label, res))

