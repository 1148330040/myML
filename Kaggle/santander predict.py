# coding:utf-8
from __future__ import division

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import re

pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings(action = 'ignore') #忽略警告

import seaborn as sns
import scipy.stats as sts
from sklearn.metrics import classification_report,f1_score
from sklearn.metrics import confusion_matrix

"""
path = 'kaggle\\SantanderPredict\\'
trains = pd.read_csv(path+'train.csv', encoding='gbk', iterator=True)
tests = pd.read_csv(path+'test.csv', encoding='gbk', iterator=True)
trains = trains.get_chunk(80000)
#tests = tests.get_chunk(40000)
#trains = pd.concat([trains, tests])
trains.index = np.arange(len(trains))
trains_features = trains.columns[2:]

trains0 = trains.loc[trains.target == 0]
trains1 = trains.loc[trains.target == 1]
# 针对于每一位用户的所有特征的std标准差，skew偏斜值，mean平均值，sum数据总和来说
# 无论是target=0或者target=1的用户都没有明显的差异

trains_ = trains[trains_features].copy()
trains['sum'] = trains_.apply(lambda x: x.sum(), axis=1)
trains['std'] = trains_.apply(lambda x: x.std(), axis=1)
trains['skew'] = trains_.apply(lambda x: x.skew(), axis=1)
trains['kurt'] = trains_.apply(lambda x: x.kurt(), axis=1)
trains['med'] = trains_.apply(lambda x: x.median(), axis=1)
trains['mean'] = trains['sum'] / len(trains_.columns)
'''
trains0 = trains.loc[trains.target == 0]
trains1 = trains.loc[trains.target == 1]
print("开始绘制")
trains0 = trains0.sample(frac=0.1)
trains1 = trains1.sample(frac=0.1)

plt.subplots(2, 2)
plt.subplot(2, 2, 1)
sns.distplot(trains1['sum'], color="magenta", kde=True, bins=15, label='target = 1')
sns.distplot(trains0['sum'], color="darkblue", kde=True, bins=15, label='target = 0')
plt.legend()
plt.subplot(2, 2, 2)
sns.distplot(trains1['skew'], color="magenta", kde=True, bins=15, label='target = 1')
sns.distplot(trains0['skew'], color="skyblue", kde=True, bins=15, label='target = 0')
plt.legend()
plt.subplot(2, 2, 3)
sns.distplot(trains1['mean'], color="darkred", kde=True, bins=15, label='target = 1')
sns.distplot(trains0['mean'], color="darkblue", kde=True, bins=15, label='target = 0')
plt.legend()
plt.subplot(2, 2, 4)
sns.distplot(trains1['sum'], color="darkgreen", kde=True, bins=15, label='target = 1')
sns.distplot(trains0['sum'], color="darkblue", kde=True, bins=15, label='target = 0')
plt.legend()
plt.show()
'''
# 我试图找到target=0和target=1的每一个特征的频数最多的值
# 结果令人意想不到,round()处理后的结果来看,某些数值的频数
# 虽然是最多但是相比按四舍五入取整来看好像并没有多大的价值

special_feature = []
enhance_feature_dict = dict()
weaken_feature_dict = dict()
# 用来添加那些在target=1中出现次数很多但在target=0中出现次数并不多的round值
for f in trains_features:
    # 找到部分不进行round频数最多的和进行过round频数最多的明显有差异的特征进行处理
    a_0 = list(round(trains0[f]).value_counts().sort_values(ascending=False).index)[:3]
    b_0 = list(trains0[f].value_counts().sort_values(ascending=False).index)[:3]
    a_1 = list(round(trains1[f]).value_counts().sort_values(ascending=False).index)[:3]
    b_1 = list(trains1[f].value_counts().sort_values(ascending=False).index)[:3]
    if abs((a_0[0] - b_0[0]))>1 or abs((a_1[0] - b_1[0]))>1:
        special_feature.append(f)
    enhance_feature_dict[f] = [0]
    weaken_feature_dict[f] = [0]
    for i in a_1:
        if i not in a_0:
            enhance_feature_dict[f].append(i)
    for i in a_1:
        if i in a_0:
            weaken_feature_dict[f].append(i)
# 根据对比我发现某些round值无论是针对target=0或是target=1他都是非常多
# 因此我推测该类值只是数量太多但是对于target=0或是1并没有明显的帮助
# 因此构建新的列保证那些target=1独有的且数量足够多的那部分值有足够的表现力
for f in trains_features:
    if len(enhance_feature_dict[f]) > 1:
        trains['special_'+f] = round(trains[f]).apply(lambda x:
                                                      1 if x in enhance_feature_dict[f] else
                                                      0)
columns = []
for i in trains.columns:
    if 'special' in i:
        columns.append(i)
trains_ = trains[columns].copy()
trains['special_value_sum'] = trains_.apply(lambda x: x.sum(), axis=1)
trains['special_value_std'] = trains_.apply(lambda x: x.std(), axis=1)
trains['special_value_skew'] = trains_.apply(lambda x: x.skew(), axis=1)
trains['special_value_kurt'] = trains_.apply(lambda x: x.kurt(), axis=1)
trains['special_value_med'] = trains_.apply(lambda x: x.median(), axis=1)
trains['special_value_mean'] = trains['special_value_sum'] / len(trains_.columns)


for sf in special_feature:
    index = list(round(trains1[sf]).value_counts().sort_values(ascending=False).index)
    trains['special_most_value_'+sf] = round(trains[sf]).apply(lambda x: 5 if x in index[:(len(index)//4)] else
                                                                   1 )


# 就数列的正负数据比率来说无论是target = 0的还是target = 1大多都没有什么很大差别
# 就数据的部分特征来说，数据的分级使用有一定的影响的

trains0_1 = trains0.copy()
trains1_1 = trains1.copy()
for f in trains_features:
   trains0_1[f] = trains0[f].apply(lambda x: 0 if x<0 else 1)
   trains1_1[f] = trains1[f].apply(lambda x: 0 if x<0 else 1)

special_features = []
error_features = []
for f in trains_features:
    try:
        target0_rate = trains0_1[f].value_counts()[1]/trains0_1[f].value_counts()[0]
        target1_rate = trains1_1[f].value_counts()[1]/trains1_1[f].value_counts()[0]
        if (target1_rate / target0_rate) > 1.3:
            special_features.append(f)
    except:
        if len(trains0_1[f][trains0_1[f]==0]) >=1 or len(trains1_1[f][trains1_1[f]==0]) >=1:
            error_features.append(f)
# 我从上一行代码中找到了target=0和target=1两块数据针对同一个特征
# 正负比率之差1/3之上的部分特征
for feature in error_features:
    trains[feature].loc[trains[feature]<0] = np.mean(trains[feature])

for feature in special_features:
    index = 'special_rate_'+feature
    trains[index] = trains[feature]
    positive = trains[index][trains[index]>0]
    negative = trains[index][trains[index]<0]
    twenty_five_percent, fifty_percent, seventy_five_percent, ninety_percent = np.percentile(positive, (25, 50, 75, 90))
    twenty_five_negative, fifty_negative, seventy_five_negative, ninety_negative = np.percentile(negative, (25, 50, 75, 90))
    positives = positive.apply(lambda x: 1 if x <= twenty_five_percent else
                               2 if x <= fifty_percent else
                               3 if x <= seventy_five_percent else
                               4 if x <= ninety_percent else
                               5)
    negatives = negative.apply(lambda x: -1 if x<=twenty_five_negative else
                               -2 if x<=fifty_negative else
                               -3 if x<=seventy_five_negative else
                               -4 if x<=ninety_negative else
                               -5)
    connect_p_n = positives.append(negatives)
    trains[index] = connect_p_n.sort_index()

for feature in trains_features:
    trains['round2_'+feature] = round(trains[feature], 2)
    trains['round3_'+feature] = round(trains[feature], 3)
trains.drop(trains_features, axis=1, inplace=True)

trains.to_csv(path+'new_checkss.csv', index=False)
print("写入完毕！")
'''
trains1s = trains.loc[trains.target == 1]
trains0s = trains.loc[trains.target == 0]
plt.subplots(2, 3)
for num in range(6):
    plt.subplot(2, 3, num+1)
    sns.barplot(x=trains1s['new_var_'+str(speci_features_num[num])].value_counts().index,
                y=trains1s['new_var_'+str(speci_features_num[num])].value_counts())
plt.show()

plt.figure(figsize=(16,6))
plt.title("Distribution of skew per column in the train and test set")
sns.distplot(trains0[features].skew(axis=0),color="magenta", kde=True, bins=120, label='target = 0')
sns.distplot(trains1[features].skew(axis=0),color="darkblue", kde=True, bins=120, label='target = 1')
plt.legend()
plt.show()
'''


trains = pd.read_csv(path+'checks.csv', encoding='gbk', iterator=True)
datas = trains.get_chunk(40000)
from sklearn.preprocessing import StandardScaler
for col in datas.columns[2:]:
    datas[col].fillna(datas[col].mean(), inplace=True)
    if 'special' not in col:
        datas[col] = StandardScaler().fit_transform(datas[col].values.reshape(-1, 1))
    if 'special_most' in col:
        datas.drop(col, axis=1, inplace=True)

try:
    datas.drop('Unnamed: 0', axis=1, inplace=True)
    print("清除成功！")
except:
    pass

train = datas[:30000]
train_label = train.pop('target')
train_id = train.pop('ID_code')
X_check = datas[30000:]
y_check = X_check.pop('target')
check_id = X_check.pop('ID_code')
'''
test = datas[200000:]
test_id = test.pop('ID_code')
test_label = test.pop('target')
'''
from imblearn.combine import SMOTETomek
clf = SMOTETomek(0.33)
# 显然，针对于高维数据平衡性处理的效率十分低下
new_train , new_train_label = train, train_label
from sklearn.model_selection import train_test_split

X_train, y_train =  new_train, new_train_label,

X_train = pd.DataFrame(X_train)
y_train = pd.Series(y_train)
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import joblib
from sklearn.model_selection import StratifiedKFold
folds = StratifiedKFold(n_splits=3, shuffle=False, random_state=123)
param = {
    'bagging_freq': 5,
    'bagging_fraction': 0.4,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.05,
    'learning_rate': 0.01,
    'max_depth': -1,
    'metric':'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary',
    'verbosity': 1
}

num_round = 100000
yp = np.zeros(len(X_check))
for i , (train_index, validate_index) in enumerate(folds.split(X_train.values, y_train.values)):
    print("folds: ",i)
    train_data_lgb = lgb.Dataset(X_train.iloc[train_index], label=y_train[train_index])
    validate_date_lgb = lgb.Dataset(X_train.iloc[validate_index], label=y_train[validate_index])
    LGB = lgb.train(param, train_data_lgb, num_round,
                    valid_sets = [train_data_lgb, validate_date_lgb],
                    verbose_eval=1000, early_stopping_rounds = 3000)
    yp += LGB.predict(X_check)
print(folds.n_splits)
try:
    result = pd.DataFrame({
        'ID_code':test_id,
        'target':yp
    })
    result.to_csv(path+'result.csv', index=False)
except:
    r = pd.DataFrame({
        'id': np.arange(len(yp)),
        'yo':yp
    })
    r.to_csv(path+'zxc.csv', index=False)

import xlearn as xl
X_train_leaves = xl.DMatrix(X_train_leaves, y_train)
X_validate_leaves =xl.DMatrix(X_validate_leaves, y_validate)
test_leaves = xl.DMatrix(test_leaves, test_label)

ffm_model = xl.create_ffm()

ffm_model.setTrain(X_train_leaves)
ffm_model.setValidate(X_validate_leaves)
ffm_model.setSign()
ffm_model.setNoBin()
ffm_model.setQuiet()



print(i)
param = {
    'task':'binary',
    'lr':1.5,
    'lambda':i,
    'metric': 'auc'
}
ffm_model.fit(param, './model_dm.out')
print("FFm模型训练完毕！")
ffm_model.setTest(test_leaves)
res = ffm_model.predict("./model_dm.out")
print(res)
print(roc_auc_score(test_label, res))
"""