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

train = pd.read_csv("kaggle\\MoveBoxoffice\\train.csv")
test = pd.read_csv("kaggle\\MoveBoxoffice\\test.csv")

TrainAdditionalFeatures = pd.read_csv("kaggle\\MoveBoxoffice\\TrainAdditionalFeatures.csv")
TestAdditionalFeatures = pd.read_csv("kaggle\\MoveBoxoffice\\TestAdditionalFeatures.csv")
TrainAdditionalFeatures.dropna(inplace=True) #空值经查看并不属于train内
TestAdditionalFeatures.dropna(inplace=True)
train = pd.merge(train,TrainAdditionalFeatures,on=['imdb_id'],how='left')
test = pd.merge(test,TestAdditionalFeatures,on=['imdb_id'],how='left')
ntrain = train.shape[0]

'''
 ['id', 'belongs_to_collection', 'budget', 'genres', 'homepage',
'imdb_id', 'original_language', 'original_title', 'overview',
'popularity', 'poster_path', 'production_companies',
'production_countries', 'release_date', 'runtime', 'spoken_languages',
'status', 'tagline', 'title', 'Keywords', 'cast', 'crew', 'revenue']
'''

#手动补充现有的部分数据
def fill_train(train):
    train.loc[train.title == 'Logan', 'revenue'] = 6*10*9
    train.loc[train['id'] == 16,'revenue'] = 192864
    train.loc[train['id'] == 90,'budget'] = 30000000
    train.loc[train['id'] == 118,'budget'] = 60000000
    train.loc[train['id'] == 149,'budget'] = 18000000
    train.loc[train['id'] == 313,'revenue'] = 12000000
    train.loc[train['id'] == 451,'revenue'] = 12000000
    train.loc[train['id'] == 464,'budget'] = 20000000
    train.loc[train['id'] == 470,'budget'] = 13000000
    train.loc[train['id'] == 513,'budget'] = 930000
    train.loc[train['id'] == 797,'budget'] = 8000000
    train.loc[train['id'] == 819,'budget'] = 90000000
    train.loc[train['id'] == 850,'budget'] = 90000000
    train.loc[train['id'] == 1112,'budget'] = 7500000
    train.loc[train['id'] == 1131,'budget'] = 4300000
    train.loc[train['id'] == 1359,'budget'] = 10000000
    train.loc[train['id'] == 1542,'budget'] = 15800000
    train.loc[train['id'] == 1571,'budget'] = 4000000
    train.loc[train['id'] == 1714,'budget'] = 46000000
    train.loc[train['id'] == 1721,'budget'] = 17500000
    train.loc[train['id'] == 1865,'revenue'] = 25000000
    train.loc[train['id'] == 2268,'budget'] = 17500000
    train.loc[train['id'] == 2491,'revenue'] = 6800000
    train.loc[train['id'] == 2602,'budget'] = 31000000
    train.loc[train['id'] == 2612,'budget'] = 15000000
    train.loc[train['id'] == 2696,'budget'] = 10000000
    train.loc[train['id'] == 2801,'budget'] = 10000000
    train.loc[train.id == 391,'runtime'] = 86
    train.loc[train.id == 592,'runtime'] = 90
    train.loc[train.id == 925,'runtime'] = 95
    train.loc[train.id == 978,'runtime'] = 93
    train.loc[train.id == 1256,'runtime'] = 92
    train.loc[train.id == 1542,'runtime'] = 93
    train.loc[train.id == 1875,'runtime'] = 86
    train.loc[train.id == 2151,'runtime'] = 108
    train.loc[train.id == 2499,'runtime'] = 108
    train.loc[train.id == 2646,'runtime'] = 98
    train.loc[train.id == 2786,'runtime'] = 111
    train.loc[train.id == 2866,'runtime'] = 96
def fill_test(test):
    test.loc[test['id'] == 3889,'budget'] = 15000000
    test.loc[test['id'] == 6733,'budget'] = 5000000
    test.loc[test['id'] == 3197,'budget'] = 8000000
    test.loc[test['id'] == 6683,'budget'] = 50000000
    test.loc[test['id'] == 5704,'budget'] = 4300000
    test.loc[test['id'] == 6109,'budget'] = 281756
    test.loc[test['id'] == 7242,'budget'] = 10000000
    test.loc[test['id'] == 7021,'budget'] = 17540562
    test.loc[test['id'] == 5591,'budget'] = 4000000
    test.loc[test['id'] == 4282,'budget'] = 20000000
    test.loc[test.id == 4074,'runtime'] = 103
    test.loc[test.id == 4222,'runtime']= 93
    test.loc[test.id == 4431,'runtime']= 100
    test.loc[test.id == 5520,'runtime'] = 86
    test.loc[test.id == 5845,'runtime'] = 83
    test.loc[test.id == 5849,'runtime']= 140
    test.loc[test.id == 6210,'runtime']= 104
    test.loc[test.id == 6804,'runtime'] = 145
    test.loc[test.id == 7321,'runtime'] = 87
fill_train(train)
fill_test(test)
test['revenue'] = 0
train['log1pRevenue'] = np.log1p(train.revenue).astype(np.float16)
train = pd.concat((train,test),sort=False).reset_index(drop=True)
def drop_feature(datas):
    datas['log1pBudget'] = np.log1p(datas.budget).astype(np.float16)
    datas.drop('budget',axis=1,inplace=True)
    datas['popularity'] = datas['popularity'].astype(np.float16)
    datas['popularity2'] = datas['popularity2'].astype(np.float16)
    datas.drop(['poster_path','title'],axis=1,inplace=True) #删除链接和图片格式
    datas.drop(['overview','original_title'],axis=1,inplace=True) #删除文字介绍,名称之类保留关键字
    return datas
train = drop_feature(train)
#缺失值查看
def show_Nan(datas):
    total = datas.isnull().sum().sort_values(ascending=False)
    percent = (datas.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data)

import ast
#找到Nan值的位置  print(train.loc[train.crew.isnull()].index)
#关于original_language不同值的数目
#print(train.original_language.value_counts())
#将其en置为1其他为0

#查看名称是否有重叠train.original_language = train.original_language.apply(lambda x: 1 if x=='en' else 0)

#print("标题总数 : ",len(train.title))
#print("唯一的标题总数 :",len(train.title.unique()))
#长度不一致说明部分数据(前30个数据)不唯一
time_start = time()

train.belongs_to_collection = train.belongs_to_collection.apply(
    lambda x: 1 if x!= None else 0
).astype(np.int8)
'''
title_train = train.title.value_counts()[:30].index
title_test = test.title.value_counts()[:55].index
def Change_title(datas,title):
    for name in title:
        index = datas.loc[datas.title==name].index
        for i in range(2):
            datas.loc[index[i],'title'] = [name+'_'+str(i+1)]
    return

train =Change_title(train,title_train)
test = Change_title(test,title_test)
'''



#观察数据关键字按照重要顺序排列这里我选择将其前三个重要的关键字id提取出来

from collections import Counter
#Counter 返回各个值出现的次数,list(Counter().elements())则将Counter对象返回原始情况
#Counter([i for j in train.genres for i in j])这种式子也很有意思



def get_times(datas):
    '''
    datas['release_date'] = pd.to_datetime(datas['release_date'], format='%m/%d/%y')
    #Series的一个处理日期序列的好方法
    datas['year'] = datas.release_date.dt.year
    pattern = r'20'
    datas['year'] = datas.year.apply(lambda x:re.sub(pattern,'19',str(x)) if int(x) > 2017 else x)
    datas['year'] = datas['year'].astype(int)
    datas['month'] = datas.release_date.dt.month
    datas['day'] = datas.release_date.dt.day
    '''
    datas[['month', 'day', 'year']] = datas['release_date'].str.split('/', expand=True).replace(
        np.nan, 0).astype(int)
    datas['year'] = datas['year']
    datas.loc[(datas['year'] <= 19) & (datas['year'] < 100), "year"] += 2000
    datas.loc[(datas['year'] > 19) & (datas['year'] < 100), "year"] += 1900
    releaseDate = pd.to_datetime(datas['release_date'])
    datas['release_dayofweek'] = releaseDate.dt.dayofweek #获取当前日期是周几
    datas['release_quarter'] = releaseDate.dt.quarter #获取当前月份是第几季度
    print("get_times 完成")
    datas.drop('release_date',axis=1,inplace=True)
    return datas

train = get_times(train)

from sklearn.preprocessing import LabelEncoder

def clean_feature(datas):
    datas.spoken_languages = datas.spoken_languages.apply(
        lambda x: list(map(lambda d: list(d.values())[0], ast.literal_eval(x)) if isinstance(x, str) else []))
    for i in datas.spoken_languages.index:
        if len(datas.loc[i,'spoken_languages'])>1:
            datas.loc[i, 'spoken_languages_others'] = 1 #表明有其他语言
            datas.loc[i, 'spoken_languages'] = datas.loc[i, 'spoken_languages'][0]
        else:
            datas.loc[i, 'spoken_languages_others'] = 0
            try:
                datas.loc[i,'spoken_languages'] = datas.loc[i,'spoken_languages'][0]
            except:
                datas.loc[i,'spoken_languages'] = 0
    datas.spoken_languages = LabelEncoder().fit_transform(datas.spoken_languages.astype(np.str))

    print("spoken_languages 完成")

    for num,id_order in enumerate(datas.crew.values):
        pattern = r"'id':.(\d+)"
        id = re.findall(pattern,str(id_order))
        for i in range(8):
            try:
                datas.loc[num,'crew_id_'+str(i+1)] = id[i]
            except:
                datas.loc[num,'crew_id_'+str(i+1)] = 0

    print("crew_id 完成")

    datas.cast = datas.cast.apply(
        lambda x: list(map(lambda d: list(d.values())[3], ast.literal_eval(x)) if isinstance(x, str) else []))
    datas['Female_cast_nums'] = datas.cast.apply(lambda x: x.count(1)).astype(np.int16)
    datas['Male_cast_nums'] = datas.cast.apply(lambda x: x.count(2)).astype(np.int16)
    datas['Ungender_cast_nums'] = datas.cast.apply(lambda x: x.count(0)).astype(np.int16)
    datas['cast_len'] = datas.cast.apply(lambda x: len(x)).astype(np.int16)

    print("cast_gender 完成")

    datas.crew = datas.crew.apply(
        lambda x: list(map(lambda d: list(d.values())[2], ast.literal_eval(x)) if isinstance(x, str) else []))
    datas['Female_crew_nums'] = datas.crew.apply(lambda x: x.count(1)).astype(np.int16)
    datas['Male_crew_nums'] = datas.crew.apply(lambda x: x.count(2)).astype(np.int16)
    datas['Ungender_crew_nums'] = datas.crew.apply(lambda x: x.count(0)).astype(np.int16)

    print("crew_gender 完成")

    datas.production_countries = datas.production_countries.apply(
        lambda x: list(map(lambda d: list(d.values())[0], ast.literal_eval(x)) if isinstance(x, str) else []))
    for num in datas.production_countries.index:
        try:
            datas.loc[num, 'produ_countries_1'] = datas.production_countries[num][0]
        except:
            datas.loc[num, 'produ_countries_1'] = 0
    print("produ_countries 完成")

    datas.genres = datas.genres.apply(
        lambda x: list(map(lambda d: list(d.values())[1], ast.literal_eval(x)) if isinstance(x, str) else []))
    datas['genres_nums'] = datas.genres.apply(lambda x: len(x)).astype(np.int16)
    datas['genres_main'] = datas.genres.apply(lambda x: x[0] if len(x) !=0 else 'None')
    top_genres = [m[0] for m in
                  Counter([i for j in train.genres for i in j]).most_common(15)]
    for g in top_genres:
        datas[g] = datas['genres'].apply(lambda x: 1 if g in x else 0)
        datas[g] = datas[g].astype(np.int8)
    print("genres 完成")

    datas.production_companies = datas.production_companies.apply(
        lambda x: list(map(lambda d: list(d.values())[0], ast.literal_eval(x)) if isinstance(x, str) else []))
    datas['production_companies_nums'] = datas.production_companies.apply(lambda x: len(x))
    datas['production_companies_main'] = datas.production_companies.apply(lambda x: x[0] if len(x) != 0 else 'None')
    top_production_companies = [m[0] for m in
                                Counter([i for j in datas.production_companies for i in j]).most_common(15)]
    for g in top_production_companies:
        datas[g] = datas['production_companies'].apply(lambda x: 1 if g in x else 0)
        datas[g] = datas[g].astype(np.int8)
    print("production_companies 完成")

    datas.Keywords = datas.Keywords.apply(
        lambda x: list(map(lambda d: list(d.values())[0], ast.literal_eval(x)) if isinstance(x, str) else []))
    for num in datas.Keywords.index:
        for i in range(3):
            try:
                datas.loc[num, 'Keywords_id_' + str(i + 1)] = datas.Keywords[num][i]
            except:
                datas.loc[num, 'Keywords_id_' + str(i + 1)] = 0
    print("key_words 完成")
    datas.drop(['Keywords','cast','crew','production_companies','genres','production_countries','spoken_languages'],axis=1,inplace=True)
    datas.runtime.fillna(90, inplace=True)
    datas.runtime = datas.runtime.astype(np.int16)
    datas.totalVotes.fillna(6, inplace=True)
    datas.totalVotes = datas.totalVotes.astype(np.float16)
    datas.rating.fillna(datas.rating.std(), inplace=True)
    datas.rating = datas.rating.astype(np.float16)
    datas.popularity2.fillna(datas.popularity2.std(), inplace=True)
    datas['_popularity_mean_year'] = datas['popularity'] / datas.groupby("year")["popularity"].transform('mean')
    datas['_log1pBudget_runtime_ratio'] = datas['log1pBudget'] / datas['runtime']
    datas['_log1pBudget_popularity_ratio'] = datas['log1pBudget'] / datas['popularity']
    datas['_log1pBudget_year_ratio'] = datas['log1pBudget'] / (datas['year'] * datas['year'])
    datas['_releaseYear_popularity_ratio2'] = datas['popularity'] / datas['year']
    datas['_rating_totalVotes_ratio'] = datas['totalVotes'] / datas['rating']
    datas['_totalVotes_releaseYear_ratio'] = datas['totalVotes'] / datas['year']
    return datas

train = clean_feature(train)
print(len(train.columns))
def other_feature(datas):
    datas.tagline = datas.tagline.apply(lambda x: 1 if isinstance(x,str) else 0).astype(np.int8)
    datas.homepage = datas.homepage.apply(lambda x: 1 if isinstance(x,str) else 0).astype(np.int8)
    datas.status = datas.status.apply(lambda x: 1 if x == 'Released' else 0).astype(np.int8)
    datas.original_language = datas.original_language.apply(lambda x: 1 if x=='en' else 2 if isinstance(x,str) else 0).astype(np.int8)
    datas.imdb_id = datas.imdb_id.apply(lambda x: re.findall(r'\d+', str(x))[0]).astype(np.int32)
    datas.runtime = datas.runtime.apply(lambda x: 90 if x <=60 else x)
    for i in range(8):
        datas['crew_id_' + str(i + 1)] = datas['crew_id_' + str(i + 1)].astype(np.int64)
    for i in range(3):
        datas['Keywords_id_'+str(i+1)] = datas['Keywords_id_'+str(i+1)].astype(np.float32)
    return datas

train = other_feature(train)
print(len(train.columns))

def Aggregate_value_(datas):
    #popularity

    popularity_year = datas.groupby(['year']).agg({
        'popularity':['mean']
    })
    popularity_year.columns = ['year_means_popularity']
    datas = pd.merge(datas,popularity_year,on=['year'],how='left')
    datas.year_means_popularity = datas.year_means_popularity.astype(np.float16)

    popularity_month = datas.groupby(['year','month']).agg({
        'popularity':['mean']
    })
    popularity_month.columns = ['month_means_popularity']
    popularity_month.reset_index(True)
    datas = pd.merge(datas,popularity_month,on=['year','month'],how='left')
    datas.month_means_popularity = datas.month_means_popularity.astype(np.float16)
    release_quarter = datas.groupby(['year', 'release_quarter']).agg({
        'popularity': ['mean']
    })
    release_quarter.columns = ['quarter_means_popularity']
    release_quarter.reset_index(True)
    datas = pd.merge(datas, release_quarter, on=['year', 'release_quarter'], how='left')
    datas.quarter_means_popularity = datas.quarter_means_popularity.astype(np.float16)

    #popularity2
    popularity2_year = datas.groupby(['year']).agg({
        'popularity2':['mean']
    })
    popularity2_year.columns = ['year_means_popularity2']
    datas = pd.merge(datas,popularity2_year,on=['year'],how='left')
    datas.year_means_popularity2 = datas.year_means_popularity2.astype(np.float16)

    popularity2_month = datas.groupby(['year','month']).agg({
        'popularity2':['mean']
    })
    popularity2_month.columns = ['month_means_popularity2']
    popularity2_month.reset_index(True)
    datas = pd.merge(datas,popularity2_month,on=['year','month'],how='left')
    datas.month_means_popularity2 = datas.month_means_popularity2.astype(np.float16)

    release_quarter = datas.groupby(['year', 'release_quarter']).agg({
        'popularity2': ['mean']
    })
    release_quarter.columns = ['quarter_means_popularity2']
    release_quarter.reset_index(True)
    datas = pd.merge(datas, release_quarter, on=['year', 'release_quarter'], how='left')
    datas.quarter_means_popularity2 = datas.quarter_means_popularity2.astype(np.float16)

    rating_year = datas.groupby(['year']).agg({
        'rating': ['mean']
    })
    rating_year.columns = ['year_means_rating']
    datas = pd.merge(datas, rating_year, on=['year'], how='left')
    datas.year_means_rating = datas.year_means_rating.astype(np.float16)

    rating_month = datas.groupby(['year', 'month']).agg({
        'rating': ['mean']
    })
    rating_month.columns = ['month_means_rating']
    rating_month.reset_index(True)
    datas = pd.merge(datas, rating_month, on=['year', 'month'], how='left')
    datas.month_means_rating = datas.month_means_rating.astype(np.float16)

    rating_quarter = datas.groupby(['year', 'release_quarter']).agg({
        'rating': ['mean']
    })
    rating_quarter.columns = ['quarter_means_rating']
    rating_quarter.reset_index(True)
    datas = pd.merge(datas, rating_quarter, on=['year', 'release_quarter'], how='left')
    datas.quarter_means_rating = datas.quarter_means_rating.astype(np.float16)

    log1pBudget_month = datas.groupby(['year', 'month']).agg({
        'log1pBudget': ['mean']
    })
    log1pBudget_month.columns = ['month_means_log1pBudget']
    log1pBudget_month.reset_index(True)
    datas = pd.merge(datas, log1pBudget_month, on=['year', 'month'], how='left')
    datas.month_means_log1pBudget = datas.month_means_log1pBudget.astype(np.float16)

    log1pBudget_quarter = datas.groupby(['year', 'release_quarter']).agg({
        'log1pBudget': ['mean']
    })
    log1pBudget_quarter.columns = ['quarter_means_log1pBudget']
    log1pBudget_quarter.reset_index(True)
    datas = pd.merge(datas, log1pBudget_quarter, on=['year', 'release_quarter'], how='left')
    datas.quarter_means_log1pBudget = datas.quarter_means_log1pBudget.astype(np.float16)

    log1pBudget_year = datas.groupby(['year']).agg({
        'log1pBudget': ['mean']
    })
    log1pBudget_year.columns = ['year_means_log1pBudget']
    datas = pd.merge(datas, log1pBudget_year, on=['year'], how='left')
    datas.year_means_log1pBudget = datas.year_means_log1pBudget.astype(np.float16)

    return datas

train = Aggregate_value_(train)
print(len(train.columns))

train.drop(['id','revenue'],axis=1,inplace=True)
train.fillna(0,axis=1,inplace=True)

train.produ_countries_1 = train.produ_countries_1.astype(np.str)
train.produ_countries_1 = LabelEncoder().fit_transform(train.produ_countries_1)
train['genres_main'] = LabelEncoder().fit_transform(train['genres_main'].astype(str))
train['production_companies_main'] = LabelEncoder().fit_transform(train['production_companies_main'].astype(str))

train.release_dayofweek = train.release_dayofweek.astype(np.int8)
train.release_quarter = train.release_quarter.astype(np.int8)
train.spoken_languages_others = train.spoken_languages_others.astype(np.int8)
train.produ_countries_1 = train.produ_countries_1.astype(np.int32)

print(train.info())
show_Nan(train)

X_train = train[:ntrain]
y_train = X_train.log1pRevenue
X_train.drop('log1pRevenue',axis=1,inplace=True)


X_test = train[ntrain:]
X_test.drop('log1pRevenue',axis=1,inplace=True)


from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso,ElasticNet
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score,KFold

X_valid,X_test,y_valid,y_test = train_test_split(
    X_train,y_train,test_size=0.2,random_state=12
)

n_folds = 5
def rmsle(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train.values)
    rmse = np.sqrt(-cross_val_score(model, X_train.values, y_train, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)

'''
alpha_nums = np.linspace(0.005,1,50)
alpha = {'alpha':alpha_nums}
clf = GridSearchCV(Lasso(),alpha,cv=5) #cv表示把数据分成五份拿一份作为测试数据
clf.fit(X_train,y_train)
best_alpha = clf.best_params_
print ("Lasso bets alpha: ",best_alpha)
'''

LAS = make_pipeline(RobustScaler(),Lasso(alpha =0.045, random_state=1))
#0.0005的系数使其对数据的影

#LAS.fit(X_valid,y_valid)
#y_predict = LAS.predict(X_test)
print("LAS 均方根误差(最终得分) : ",rmsle(LAS).mean())


'''
alpha_nums = np.linspace(10,20,5)
tol = np.linspace(0.001,0.01,5)
alpha = {'C':alpha_nums,'tol':tol}
clf = GridSearchCV(SVR(),alpha,cv=5) #cv表示把数据分成五份拿一份作为测试数据
clf.fit(X_train,y_train)
best_alpha = clf.best_params_
print ("SVR bets alpha: ",best_alpha)
'''

SVR = make_pipeline(RobustScaler(),SVR(C=10,tol=0.008,epsilon=0.008))

print("SVR 均方根误差(最终得分) : ",rmsle(SVR).mean())



'''
alpha_nums = np.linspace(0.002,0.2,10)
li_ratios = np.linspace(2,10,5)
alpha = {'alpha':alpha_nums,'l1_ratio':li_ratios}
clf = GridSearchCV(ElasticNet(),alpha,cv=5) #cv表示把数据分成五份拿一份作为测试数据
clf.fit(X_train,y_train)
best_alpha = clf.best_params_
print ("ENET  bets alpha: ",best_alpha)
'''
ENT = make_pipeline(RobustScaler(),ElasticNet(alpha=0.025, l1_ratio=2, random_state=3))

print("ENT 均方根误差(最终得分) : ",rmsle(ENT).mean())

from sklearn.ensemble import GradientBoostingRegressor
#随机数森和梯度提升(参数类似于决策树)

'''
learning_rate = np.linspace(0.1,1,10)
alpha = {'learning_rate':learning_rate}
clf = GridSearchCV(GradientBoostingRegressor(),alpha,cv=5) #cv表示把数据分成五份拿一份作为测试数据
clf.fit(X_train,y_train)
best_alpha = clf.best_params_
print ("GBR bets alpha: ",best_alpha)
'''
GBR = GradientBoostingRegressor(n_estimators=3000, #弱学习器的数量(人多力量大) Boosting思想
                                learning_rate=0.1,#学习率
                                max_depth=4, max_features='sqrt',
                                min_samples_leaf=15, min_samples_split=50,
                                loss='huber', random_state =5)

print("GBR 均方根误差(最终得分) : ",rmsle(GBR).mean())

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import ShuffleSplit,KFold


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)

model_stack = StackingAveragedModels(base_models=(ENT,SVR,LAS),meta_model=GBR,n_folds=5)
try:
    print("StackModel",rmsle(model_stack).mean())
except:
    print("还是有错")


LGB = LGBMRegressor(n_estimators=10000,objective='regression',
                    max_depth = 5,num_leaves=30,
                    min_child_samples=100,learning_rate=0.01,
                    boosting = 'gbdt',min_data_in_leaf= 10,
                    feature_fraction = 0.9,bagging_freq = 1,
                    bagging_fraction = 0.9,importance_type='gain',
                    lambda_l1 = 0.2,subsample=.8,
                    colsample_bytree=.9,use_best_model=True)

#print("LGB rmsle : ",rmsle(LGB))
LGB.fit(
    X_valid,
    y_valid,
    eval_metric="rmse",
    eval_set=[(X_valid, y_valid), (X_test, y_test)],
    verbose=True,
    early_stopping_rounds = 10 #可以通过设置参数 early_stopping_rounds 来解决因为迭代次数过多而过拟合的状态
    )
XGB = XGBRegressor(colsample_bytree=0.7, #构造每个树列的子采样率。子采样将在每次增强迭代中发生一次。
                   objective='reg:linear',
                   gamma=1, #在树的叶节点上进行进一步分区所需的最小损耗减少.越大.算法越保守(0~∞)
                   learning_rate=0.01, #更新使用中的步长缩小以防止过度拟合 (0,1)
                   max_depth=4,n_estimators=2000, #构建的子树的数量
                   reg_alpha=0.4640, #L1正则化项
                   reg_lambda=0.8571, #L2正则化项
                   subsample=0.5213, #训练时抽取的样本比率
                   silent=True, #1表示不打印训练的信息,True,0表示打印
                   random_state =7
                   )

#print("XGB rmsle : ",rmsle(XGB))

XGB.fit(
    X_valid,
    y_valid,
    eval_metric="rmse",
    eval_set=[(X_valid, y_valid), (X_test, y_test)],
    verbose=True,
    early_stopping_rounds = 10
    #可以通过设置参数 early_stopping_rounds 来解决因为迭代次数过多而过拟合的状态
    )
"""