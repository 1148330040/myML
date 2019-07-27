# conding:gbk

'''
此文件的目的
1 对新数据进行初步的处理
2 与训练数据进行合并处理
3 将新数据整理成模型可以识别的数据
4 传递给模型文件
5 用于GBDT+FFM
'''
import pandas as pd
import numpy as np
from ProcessingData.GBDT_Deliver_Data import *
pd.set_option('display.max_columns', None)
from sqlalchemy import create_engine

'''
import random
train = pd.read_csv("C:\\Users\\dell--pc\\Desktop\\py与贝叶斯\\kaggle\\UserBehavior\\tianchi_mobile_recommend_train_user.csv",
                    encoding='gbk', iterator = True, engine='python')

trains = train.get_chunk(1000)
trains.drop('user_geohash', axis=1, inplace=True)
trains.rename(columns={'item_category': 'category_id'}, inplace=True)

engine = create_engine("mysql+pymysql://root:123456@localhost:3306/mysql?charset=utf8")

trains.to_sql(name='new_user_data_gbdt', con=engine, if_exists='replace',
            index=False, index_label=False, chunksize=5000)
print("写入完毕")

trains = train.get_chunk(50000)
trains.drop('user_geohash', axis=1, inplace=True)
trains.rename(columns={'item_category': 'category_id'}, inplace=True)
engine = create_engine("mysql+pymysql://root:123456@localhost:3306/mysql?charset=utf8")
trains.to_sql(name='fit_datas_gbdt', con=engine, if_exists='replace',
            index=False, index_label=False, chunksize=5000)
print("写入完毕")
'''
# 获取new_user_datas,user_datas和cate_datas
# cate_datas是从user_datas中仅仅针对category_id的数据
def get_datas(variable):
    # variable = 1 指Android端传来的指令
    if variable == '1':
        new_user_data = deliver_new_user_data_and()
    else:
        new_user_data = deliver_new_user_data_java()
        new_user_data = java_process_data(new_user_data)
    user_id_list = list(new_user_data.user_id)
    # sql语句需要使用元组进行传递
    user_datas = deliver_user_datas(user_id_list, new_user_data)
    for feature in new_user_data.columns:
        try:
            new_user_data[feature] = new_user_data[feature].astype(user_datas[feature].dtype)
        except:
            pass
    category_id_list = list(new_user_data.category_id)
    cate_datas = pd.DataFrame(deliver_cate_datas(category_id_list, new_user_data))
    return user_datas, cate_datas, new_user_data

# java数据需要预先做一些处理
def java_process_data(new_user_data):
    browse_type = new_user_data.loc[new_user_data.behavior_type == 1].item_id
    browse_type = pd.DataFrame({'item_id': list(browse_type.unique()),
                                'browse_type': 1})
    new_user_data = pd.merge(new_user_data, browse_type, on='item_id', how='left')

    collect_type = new_user_data.loc[new_user_data.behavior_type == 2].item_id
    collect_type = pd.DataFrame({'item_id': list(collect_type.unique()),
                                 'collect_type': 1})
    new_user_data = pd.merge(new_user_data, collect_type, on='item_id', how='left')

    addcart_type = new_user_data.loc[new_user_data.behavior_type == 3].item_id
    addcart_type = pd.DataFrame({'item_id': list(addcart_type.unique()),
                                 'addcart_type': 1})
    new_user_data = pd.merge(new_user_data, addcart_type, on='item_id', how='left')

    bought_type = new_user_data.loc[new_user_data.behavior_type == 4].item_id
    bought_type = pd.DataFrame({'item_id': list(bought_type.unique()),
                                'bought_type': 1})
    new_user_data = pd.merge(new_user_data, bought_type, on='item_id', how='left')
    new_user_data.drop('behavior_type', axis=1, inplace=True)
    new_user_data.fillna(0, inplace=True)
    return new_user_data

# 预处理new_user_data
def preprocessing_data(new_user_data):
    releaseDate = pd.to_datetime(new_user_data['time'])
    new_user_data['year'] = 2014
    new_user_data['year'] = new_user_data['year'].astype(np.int16)
    new_user_data['month'] = releaseDate.dt.month.astype(np.int8)
    new_user_data['day'] = releaseDate.dt.day.astype(np.int8)
    new_user_data['week'] = releaseDate.dt.dayofweek.astype(np.int8) #获取当前日期是周几
    new_user_data['hour'] = releaseDate.dt.hour.astype(np.int8)
    new_user_data['days'] = new_user_data['day']
    new_user_data.loc[new_user_data.month==11, 'days'] = new_user_data.loc[new_user_data.month==11,'days']-17
    new_user_data.loc[new_user_data.month==12, 'days'] = new_user_data.loc[new_user_data.month==12,'days']+18

    new_user_data.drop(['time'], axis=1, inplace=True)


    morning_time = np.arange(6, 13)
    afternoon_time = np.arange(12, 20)
    night_time = np.arange(19, 25)
    smallhour_time = np.arange(0, 6)
    new_user_data['four_time_days'] = new_user_data.hour.apply(lambda x: 1 if x in morning_time else
                                                                         2 if x in afternoon_time else
                                                                         3 if x in night_time else
                                                                         4)
    return new_user_data

# 滞后函数
def user_days_lag_processing(user_datas, new_user_data, feature, nums):
    df = user_datas[['user_id', 'days', feature]]
    for i in nums:
        group = df.copy()
        group.columns = ['user_id', 'days', feature + '_lag_' + str(i)]
        group.drop_duplicates(
            subset=['user_id', 'days'], keep='first', inplace=True)
        group['days'] += i
        new_user_data = pd.merge(new_user_data, group, on=['user_id', 'days'], how='left', sort=False)
    return new_user_data

def user_cate_lag_processing(user_datas, new_user_data, feature, nums):
    group = user_datas[['user_id', 'days', 'category_id',feature]]
    for i in nums:
        group = group.copy()
        group.columns = ['user_id', 'days', 'category_id', feature+'_lag_'+str(i)]
        group['days'] +=1
        group.drop_duplicates(
            subset=['user_id', 'days', 'category_id'], keep='first', inplace=True)
        new_user_data = pd.merge(new_user_data, group, on=['user_id', 'days', 'category_id'], how='left')
    return new_user_data

# 将新数据与user_datas和cate_datas进行连接
# 获取需要的特征
def link_user_data(user_datas, cate_datas, new_user_data):
    new_user_data = preprocessing_data(new_user_data)
    user_feature = ['user_id',
                    'user_behavior_activity', 'user_probability', 'desire_buy',
                    'user_four_time_days_probability']
    group = user_datas[user_feature]
    group = group.drop_duplicates(
        subset=['user_id'], keep='first').copy()
    new_user_data = pd.merge(new_user_data, group, on='user_id', how='left')

    user_cate_feature = ['user_id', 'category_id',
                         'user_category_behavior_activity',
                         'user_category_first_click_browse', 'user_category_first_click_collect',
                         'user_category_first_click_addcart', 'user_category_first_click_bought',
                         'user_cate_probability', 'user_cate_allcate_rate']
    group1 = user_datas[user_cate_feature]
    group1= group1.drop_duplicates(
        subset=['user_id', 'category_id'], keep='first').copy()
    new_user_data = pd.merge(new_user_data, group1, on=['user_id', 'category_id'], how='left')

    user_cate_week_feature = ['user_id', 'category_id', 'week',
                              'user_week_cate_collect_count',
                              'user_week_probability', 'user_week_cate_browse_count', 'user_week_cate_addcart_count',
                              'user_week_cate_bought_count', 'user_week_cate_allcate_rate']
    group2 = user_datas[user_cate_week_feature]
    group2 = group2.drop_duplicates(
        subset=['user_id', 'category_id', 'week'], keep='first').copy()
    new_user_data = pd.merge(new_user_data, group2, on=['user_id', 'category_id', 'week'], how='left')

    user_item_feature = ['user_id', 'item_id',
                'user_item_behavior_activity']
    group3 = user_datas[user_item_feature]
    group3 = group3.drop_duplicates(
        subset=['user_id', 'item_id'], keep='first').copy()
    new_user_data = pd.merge(new_user_data, group3, on=['user_id', 'item_id'], how='left')

    special_days = [30]
    new_user_data['special_days'] = new_user_data.days.apply(lambda x: 10 if x in special_days else 1)

    lag_browse_feature = 'user_days_browse_count'
    lag_browse_nums = [1]
    new_user_data = user_days_lag_processing(user_datas, new_user_data, lag_browse_feature, lag_browse_nums)

    lag_collect_feature = 'user_days_collect_count'
    lag_collect_nums = [1, 2, 3]
    new_user_data = user_days_lag_processing(user_datas, new_user_data, lag_collect_feature, lag_collect_nums)

    lag_addcart_feature = 'user_days_addcart_count'
    lag_addcart_nums = [1, 2, 3]
    new_user_data = user_days_lag_processing(user_datas, new_user_data, lag_addcart_feature, lag_addcart_nums)

    lag_bought_feature = 'user_days_bought_count'
    lag_bought_nums = [1, 2, 3, 4, 5]
    new_user_data = user_days_lag_processing(user_datas, new_user_data, lag_bought_feature, lag_bought_nums)

    lag_category_feature = 'user_days_cate_probability'
    lag_category_nums = [1, 2, 3, 4, 5]
    new_user_data = user_cate_lag_processing(user_datas, new_user_data, lag_category_feature, lag_category_nums)

    cate_feature = ['category_id',
                    'cate_borwse_count', 'cate_collect_count', 'cate_addcart_count',
                    'cate_bought_count', 'cate_probability']
    group1_cate = cate_datas[cate_feature]
    group1_cate = group1_cate.drop_duplicates(
        subset=['category_id'], keep='first').copy()
    new_user_data = pd.merge(new_user_data, group1_cate, on=['category_id'], how='left')

    cate_week_feature = ['category_id', 'week',
                         'cate_allcate_week_rate', 'cate_week_probability']
    group2_cate = cate_datas[cate_week_feature]
    group2_cate = group2_cate.drop_duplicates(
        subset=['category_id', 'week'], keep='first').copy()
    new_user_data = pd.merge(new_user_data, group2_cate, on=['category_id', 'week'], how='left')

    cate_day_feature = ['category_id', 'day',
                        'cate_allcate_day_rate', 'cat_day_probability']
    group3_cate = cate_datas[cate_day_feature]
    group3_cate = group3_cate.drop_duplicates(
        subset=['category_id', 'day'], keep='first').copy()
    new_user_data = pd.merge(new_user_data, group3_cate, on=['category_id', 'day'], how='left')

    return new_user_data

# 获取到经过处理的用户的新的数据
def get_new_user_data(variable):
    user_datas, cate_datas, new_user_data = get_datas(variable=variable)
    end_user_data = link_user_data(user_datas, cate_datas, new_user_data)
    return end_user_data



