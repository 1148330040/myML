# coding:gbk

import pandas as pd
import numpy as np

from TrainingData.GBDT_Data_Clean import get_fit_train_datas

import warnings
warnings.filterwarnings(action = 'ignore') #���Ծ���


def get_fit_datas():
    datas = get_fit_train_datas(variable=1)
    return datas
datas = get_fit_datas()
# 4:1 ����ѵ�����ݺͲ�������
size = round(len(datas)*0.8)
# ���������������ΪҪ�������û����ݽ����ͺ�ֵ�Ĺ���
# �����������ϴ�ļ��в���ֱ��ɾ��
# ������ѵ��ģ��ʱ��Ҫ�ٴ�ɾ���⼸��������
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
# �����ݷ�Ϊ����-> ������Ϊ4�ĺ�û�в�����Ϊ4��
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
    max_delta_step=10  # ÿ��Ҷ���������󲽳�,ͨ������������ݲ�ƽ�������
)
XGB.fit(train.values, train_label.values)
# ����ģ��
joblib.dump(XGB,'XGB_FFM.model')



# �����train, valida, test����ȫ���Ǵ����ݿ��ڻ�ȡ����
def get_datas(model, train, valida, test):
    new_train = model.apply(train.values)
    new_valida = model.apply(valida.values)
    new_test = model.apply(test.values)
    return new_train, new_valida, new_test

new_train, new_valida, new_test = get_datas(XGB, train, valida, test)
# ѵ����,��֤��,���Լ�


from sklearn.metrics import classification_report,roc_auc_score
import xlearn as xl

ffm_train = xl.DMatrix(new_train, train_label)
ffm_valida = xl.DMatrix(new_valida, valida_label)
ffm_test = xl.DMatrix(new_test, test_label)

print("��ʼѵ��FFmģ��")
ffm_model = xl.create_ffm()

ffm_model.setTrain(ffm_train)
ffm_model.setValidate(ffm_valida)
# ������ǰ����setValidate
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
print("FFmģ��ѵ����ϣ�")
ffm_model.setTest(ffm_test)
res = ffm_model.predict("./model_dm.out")
print(pd.Series(test_label).value_counts())
print(pd.Series(res).value_counts())
print(roc_auc_score(test_label, res))
print(classification_report(test_label, res))

