# coding:gbk

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from ProcessingData.GBDT_Deliver_Data import *


def show_Nan(datas):
    total = datas.isnull().sum().sort_values(ascending=False)
    percent = (datas.isnull().sum()/datas.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data)

# 构建滞后值 days
def Previous_days(df, feature, nums):
    tmp = df[['days', 'user_id', feature]]
    for i in nums:
        shifted = tmp.copy()
        shifted.columns = ['days', 'user_id', feature+'_lag_'+str(i)]
        # 为何要对shifted进行去重？因为类型号相同的商品其item_id可能是不同的,但是由于针对的是商品类型，
        # 所以又不能为其添加商品的id这就导致了shifted里面出现了同样的数据，实际上他们的item_id是不同的，
        # 因此数据只是表面相同，这就导致了在和存在item_id的df，即原始数据合并的时候，由于只针对商品类型和days，
        # 所以导致了shifted的一条数据可能与多个df行进行了多次的合并，因此出现了大量的重复行
        shifted.drop_duplicates(subset=['days', 'user_id'], keep='first', inplace=True)
        shifted['days'] += i
        df = pd.merge(df, shifted, on=['days', 'user_id'], how='left', sort=False)
    return df

def clean_train(trains):
    train = trains.copy()
    # print(pd.Series(behavior_type).value_counts())
    releaseDate = pd.to_datetime(train['time'])
    train['year'] = 2014
    train['year'] = train['year'].astype(np.int16)
    train['month'] = releaseDate.dt.month.astype(np.int8)
    train['day'] = releaseDate.dt.day.astype(np.int8)
    train['week'] = releaseDate.dt.dayofweek.astype(np.int8) #获取当前日期是周几
    train['hour'] = releaseDate.dt.hour.astype(np.int8)
    train['days'] = train['day']
    train.loc[train.month==11, 'days'] = train.loc[train.month==11,'days']-17
    train.loc[train.month==12, 'days'] = train.loc[train.month==12,'days']+18

    train.drop(['time'], axis=1, inplace=True)


    train['browse'] = train.behavior_type.apply(lambda x: 1 if x==1 else 0).astype(np.int8)
    train['collect'] = train.behavior_type.apply(lambda x: 1 if x==2 else 0).astype(np.int8)
    train['addcart'] = train.behavior_type.apply(lambda x: 1 if x==3 else 0).astype(np.int8)
    train['bought'] = train.behavior_type.apply(lambda x: 1 if x==4 else 0).astype(np.int8)

    morning_time = np.arange(6,13)
    afternoon_time = np.arange(12,20)
    night_time = np.arange(19,25)
    train['four_time_days'] = train.hour.apply(lambda x: 1 if x in morning_time else
                                                         2 if x in afternoon_time else
                                                         3 if x in night_time else
                                                         4 )

    # 获取用户的行为链
    browse_type = train.loc[train.behavior_type==1].item_id
    browse_type = pd.DataFrame({'item_id':list(browse_type.unique()),
                                'browse_type':1})
    train = pd.merge(train, browse_type, on='item_id', how='left')

    collect_type = train.loc[train.behavior_type==2].item_id
    collect_type = pd.DataFrame({'item_id':list(collect_type.unique()),
                                 'collect_type':1})
    train = pd.merge(train, collect_type, on='item_id', how='left')

    addcart_type = train.loc[train.behavior_type==3].item_id
    addcart_type = pd.DataFrame({'item_id':list(addcart_type.unique()),
                                 'addcart_type':1})
    train = pd.merge(train, addcart_type, on='item_id', how='left')

    bought_type = train.loc[train.behavior_type==4].item_id
    bought_type = pd.DataFrame({'item_id':list(bought_type.unique()),
                                'bought_type':1})
    train = pd.merge(train, bought_type, on='item_id', how='left')
    train.fillna(0, inplace=True)

    train.drop_duplicates(
        subset=['item_id', 'user_id', 'hour', 'collect_type', 'addcart_type', 'bought_type'],
        keep='first',
        inplace=True)


    # 消除爬虫用户产生的影响
    group = train.groupby('user_id').agg({
        'browse':['sum'],
        'collect':['sum'],
        'addcart':['sum'],
        'bought':['sum']
    })
    group.columns = ['browse_', 'collect_', 'addcart_', 'bought_']
    group.reset_index(inplace=True)
    group['bug'] = (group['collect_']+group['addcart_']+group['bought_'])/group['browse_']
    group = group.loc[(group['bug']==0)]
    bug_user = list(group.loc[(group['browse_']>group['browse_'].mean()),'user_id'])
    # 先将bug用户的id作为index,然后通过删除指定的index来侧面的删除爬虫用户
    train.index = train.user_id
    train.drop(index=bug_user, inplace=True)


    # 商品类型行为特征总数
    # 此特征值仅用来获取某类商品的购买转化率
    group = train.groupby(['category_id']).agg({
                'browse': ['sum'],
                'collect': ['sum'],
                'addcart': ['sum'],
                'bought': ['sum']
    })
    feature_names = ['cate_borwse_count', 'cate_collect_count',
                     'cate_addcart_count', 'cate_bought_count']
    group.columns = feature_names
    group.reset_index(inplace=True)
    train = pd.merge(train, group, on=['category_id'], how='left')
    train['cate_borwse_count'] =  train.cate_borwse_count.astype(np.int32)
    train['cate_collect_count'] = train.cate_collect_count.astype(np.int32)
    train['cate_addcart_count'] = train.cate_addcart_count.astype(np.int32)
    train['cate_bought_count'] =  train.cate_bought_count.astype(np.int32)


    # 总的购买行为与其他行为之比
    #-----------------------------------
    train['cate_probability'] = train.cate_borwse_count+train.cate_collect_count+train.cate_addcart_count
    train['cate_probability'] = train.cate_bought_count/train.cate_probability
    train['cate_probability'] = train.cate_probability.astype(np.float32)
    train.loc[train.cate_probability==np.inf, 'cate_probability'] = 1.0
    train.loc[train.cate_probability==1.0, 'cate_probability'] = train.cate_probability.mean()

    # 用星期号聚合
    group = train.groupby(['week']).agg({
                'browse': ['sum'],
                'collect': ['sum'],
                'addcart': ['sum'],
                'bought': ['sum']
    })
    feature_names_week = ['week_browse_count', 'week_collect_count',
                          'week_addcart_count', 'week_bought_count']
    group.columns = feature_names_week
    group.reset_index(inplace=True)
    train = pd.merge(train, group, on=['week'], how='left')
    train['week_browse_count'] = train.week_browse_count.astype(np.int32)
    train['week_collect_count'] = train.week_collect_count.astype(np.int32)
    train['week_addcart_count'] = train.week_addcart_count.astype(np.int32)
    train['week_bought_count'] = train.week_bought_count.astype(np.int32)

    # 商品按星期号进行类型聚合的行为总数
    group = train.groupby(['week', 'category_id']).agg({
        'browse' :['sum'],
        'collect':['sum'],
        'addcart':['sum'],
        'bought' :['sum']
    })
    feature_names = ['cate_week_browse_count', 'cate_week_collect_count',
                     'cate_week_addcart_count', 'cate_week_bought_count']
    group.columns = feature_names
    group.reset_index(inplace=True)
    train = pd.merge(train, group, on=['week', 'category_id'], how='left')

    # 某一商品星期号进行的行为总数占星期号总商品行为总数的比值->可得到该商品类型在当前星期号受欢迎的程度
    train['cate_allcate_week_rate'] = (train['cate_week_browse_count']+train['cate_week_collect_count']+
                                       train['cate_week_addcart_count']+train['cate_week_bought_count'])

    train['cate_allcate_week_rate']=train['cate_allcate_week_rate']/(train['week_browse_count']+train['week_collect_count']+
                                                                     train['week_addcart_count']+train['week_bought_count'])

    #-----------------------------------
    # 具体到星期号产品的购买转化率
    train['cate_week_probability'] = train.cate_week_browse_count+train.cate_week_collect_count+train.cate_week_addcart_count
    train['cate_week_probability'] = train.cate_week_bought_count/train.cate_week_probability
    train['cate_week_probability'] = train.cate_week_probability.astype(np.float64)
    train.loc[train.cate_week_probability==np.inf, 'cate_week_probability'] = 1.0
    train.loc[train.cate_week_probability==1.0, 'cate_week_probability'] = train.cate_week_probability.mean()

    for feature in feature_names:
        train.drop(feature, axis=1, inplace=True)
    for feature in feature_names_week:
        train.drop(feature, axis=1, inplace=True)


    # 商品类型按日期聚合的行为数目
    # 此特征用来构建滞后值

    # 以日期号进行聚合的行为总数，注意这里是day
    group = train.groupby(['day']).agg({
                'browse':  ['sum'],
                'collect': ['sum'],
                'addcart': ['sum'],
                'bought':  ['sum']
    })
    feature_names_day = ['day_browse_count', 'day_collect_count',
                         'day_addcart_count', 'day_bought_count']
    group.columns = feature_names_day
    group.reset_index(inplace=True)
    train = pd.merge(train, group, on=['day'], how='left')

    #sns.barplot(x=train['days'], y=train['days_all_type'] )
    #plt.show()
    # 由图可以明显看到双12那一天行为总数突发猛涨因此一些特殊的日子需要额外的赋予权值

    special_days = [30]
    train['special_days'] = train.days.apply(lambda x: 10 if x in special_days else 1)

    # 注意这里是day不是days
    group = train.groupby(['day', 'category_id']).agg({
        'browse': ['sum'],
        'collect':['sum'],
        'addcart':['sum'],
        'bought': ['sum']
    })
    feature_names = ['cate_day_browse_count', 'cate_day_collect_count',
                     'cate_day_addcart_count', 'cate_day_bought_count']
    group.columns = feature_names

    group.reset_index(inplace=True)
    train = pd.merge(train, group, on=['day','category_id'], how='left')


    # 某类商品在日期号进行的行为总数占日期号总商品类型的行为总数->可得到该商品在当日受欢迎的程度
    train['cate_allcate_day_rate'] = (train['cate_day_browse_count']+train['cate_day_collect_count']
                                     +train['cate_day_addcart_count']+train['cate_day_bought_count'])

    train['cate_allcate_day_rate'] = train['cate_allcate_day_rate'] / (train['day_browse_count']+train['day_collect_count']
                                                                      +train['day_addcart_count']+train['day_bought_count'])

    # 日期号day某种商品的购买转化率
    train['cat_day_probability'] = train['cate_day_bought_count'] / (train['cate_day_browse_count']+
                                                                    train['cate_day_collect_count']+
                                                                    train['cate_day_addcart_count'])
    train['cat_day_probability'] = train.cat_day_probability.astype(np.float64)

    for feature in feature_names_day:
        train.drop(feature, axis=1, inplace=True)
    for feature in feature_names:
        train.drop(feature, axis=1, inplace=True)


    # 用户在之前一段时间内按照商品类型的行为总和与36(日期总长)的比值 侧面反映该用户在这段时间内活跃度
    # 如果数据量增大的话(比如一年内有12个月)在groupby内增加月份特征即可

    train['user_behavior_activity'] = train.groupby(['user_id'])['behavior_type'].transform('sum') / 36

    train['user_category_behavior_activity'] = train.groupby(['user_id', 'category_id'])['behavior_type'].transform('sum') / 36

    train["user_item_behavior_activity"] = train.groupby(['user_id', 'item_id'])['behavior_type'].transform('sum') / 36

    # 关于用户的特征
    group = train.groupby(['user_id']).agg({
                'browse':  ['sum'],
                'collect': ['sum'],
                'addcart': ['sum'],
                'bought':  ['sum']
    })
    feature_names_user = ['user_browse_count', 'user_collect_count',
                          'user_addcart_count', 'user_bought_count']
    group.columns = feature_names_user
    group.reset_index(inplace=True)
    train = pd.merge(train, group, on=['user_id'], how='left')


    # 用户的购买转化率
    #-----------------------------------
    train['user_probability'] = train.user_browse_count+train.user_collect_count+train.user_addcart_count
    train['user_probability'] = train.user_bought_count/train.user_probability
    train['user_probability'] = train.user_probability.astype(np.float64)
    train.loc[train.user_probability==np.inf, 'user_probability'] = 1.0
    train.loc[train.user_probability==1.0, 'user_probability'] = train.user_probability.mean()

    # 用户购买力
    train['desire_buy'] = train['user_browse_count']+train['user_collect_count']+train['user_addcart_count']+train['user_bought_count']
    four_nums = train.describe().desire_buy.values[-4:]

    train['desire_buy'] = train['desire_buy'].apply(lambda x: 0.5 if x<=(four_nums[0]//2)
                                                    else 1 if x<(four_nums[2]+four_nums[3])//2
                                                    else 4
                                                    )

    #用户上次产生的是什么行为
    group = train.groupby(['user_id','category_id'])[['behavior_type','days']].transform('min')
    group.columns = ['get_behavior_type','zzzzzz']
    group.reset_index(inplace=True)
    train['user_category_first_click_browse'] = group.get_behavior_type.apply(lambda x: 1 if x==1 else 0)
    train['user_category_first_click_collect'] = group.get_behavior_type.apply(lambda x: 1 if x==2 else 0)
    train['user_category_first_click_addcart'] = group.get_behavior_type.apply(lambda x: 1 if x==3 else 0)
    train['user_category_first_click_bought'] = group.get_behavior_type.apply(lambda x: 1 if x==4 else 0)
    #print (train.groupby(['user_id','category_id'])['days'].transform('min').head(10))
    #print(train.groupby(['user_id','category_id'])[['behavior_type','days']].transform('min').head(10))
    feature_names = ['user_category_first_click_browse','user_category_first_click_collect',
                     'user_category_first_click_addcart','user_category_first_click_bought']
    for feature in feature_names:
        train[feature] = train[feature].astype(np.int8)

    # 聚合用户对某类商品的聚合值
    group = train.groupby(['user_id', 'category_id']).agg({
                'browse':  ['sum'],
                'collect': ['sum'],
                'addcart': ['sum'],
                'bought':  ['sum']
    })
    feature_names = ['user_cate_browse_count', 'user_cate_collect_count',
                    'user_cate_addcart_count', 'user_cate_bought_count']
    group.columns = feature_names
    group.reset_index(inplace=True)
    train = pd.merge(train, group, on=['user_id', 'category_id'], how='left')

    # 用户对某类商品的购买行为与这类商品的其他行为的比(即其他行为转变为购买行为的比率)
    train['user_cate_probability'] = train.user_cate_browse_count+train.user_cate_collect_count+train.user_cate_addcart_count
    train['user_cate_probability'] = train.user_cate_bought_count/train.user_cate_probability
    train['user_cate_probability'] = train.user_cate_probability.astype(np.float64)


    # 用户对某类商品的行为相比于用户总的行为的比->可判断用户对某类商品的偏好程度
    train['user_cate_allcate_rate'] = (train['user_cate_browse_count']+train['user_cate_collect_count']+
                                       train['user_cate_addcart_count']+train['user_cate_bought_count'])
    train['user_cate_allcate_rate'] = train['user_cate_allcate_rate']/(train['user_browse_count']+train['user_collect_count']+
                                                                       train['user_addcart_count']+train['user_bought_count'])

    for feature in feature_names_user:
        train.drop(feature, axis=1, inplace=True)
    for feature in feature_names:
        train.drop(feature, axis=1, inplace=True)


    # 用户在星期号的行为总数
    group = train.groupby(['week', 'user_id']).agg({
                'browse': ['sum'],
                'collect': ['sum'],
                'addcart': ['sum'],
                'bought': ['sum']
    })
    feature_names_week_user = ['user_week_browse_count', 'user_week_collect_count',
                               'user_week_addcart_count', 'user_week_bought_count']
    group.columns = feature_names_week_user
    group.reset_index(inplace=True)
    train = pd.merge(train,group,on=['week', 'user_id'],how='left')
    # 用户在星期号的购买转化率
    train['user_week_probability'] = train['user_week_bought_count'] / (train['user_week_browse_count']+
                                                                        train['user_week_addcart_count']+
                                                                        train['user_week_collect_count'])


    # 用户在星期号对某类商品的行为总数
    group = train.groupby(['week', 'user_id', 'category_id']).agg({
                'browse': ['sum'],
                'collect': ['sum'],
                'addcart': ['sum'],
                'bought': ['sum']
    })
    feature_names = ['user_week_cate_browse_count', 'user_week_cate_collect_count',
                     'user_week_cate_addcart_count', 'user_week_cate_bought_count']
    group.columns = feature_names
    group.reset_index(inplace=True)
    train = pd.merge(train, group, on=['week', 'user_id', 'category_id'], how='left')

    #-----------------------------------

    train['user_week_cate_allcate_rate'] = (train['user_week_cate_browse_count']+train['user_week_cate_collect_count']+
                                            train['user_week_cate_addcart_count']+train['user_week_cate_bought_count'])
    # 星期号用户对某类商品偏好程度
    train['user_week_cate_allcate_rate'] = train['user_week_cate_allcate_rate']/(train['user_week_browse_count'] + train['user_week_collect_count']+
                                                                                 train['user_week_addcart_count'] + train['user_week_bought_count'])


    train['user_week_cate_allcate_rate'] = train.user_week_probability.astype(np.float64)

    for feature in feature_names_week_user:
        train.drop(feature, axis=1, inplace=True)

    '''
    # 用户在日期号(day)进行的行为总数，注意是day因为用户针对的是日期(比如说10号,那么就有一年就有12个10号)
    # 此特征同样针对月份数目多的数据
    group = train.groupby(['day', 'user_id']).agg({
                'browse':  ['sum'],
                'collect': ['sum'],
                'addcart': ['sum'],
                'bought':  ['sum']
    })
    feature_names = ['user_day_browse_count', 'user_day_collect_count',
                     'user_day_addcart_count', 'user_day_bought_count']
    group.columns = feature_names
    
    group.reset_index(inplace=True)
    train = pd.merge(train, group, on=['day', 'user_id'], how='left')
    
    # 用户在日期号(day)的购买转化率
    # 周期短，因此不进行偏好程度的建立
    train['user_day_probability'] = train['user_day_bought_count'] / (train['user_day_browse_count']+
                                                                      train['user_day_collect_count']+
                                                                      train['user_day_addcart_count'])
    
    '''


    # 通过判断前N天用户的行为情况来判断当下用户产生某种行为的可能性
    group = train.groupby(['days', 'user_id']).agg({
                'browse':  ['sum'],
                'collect': ['sum'],
                'addcart': ['sum'],
                'bought':  ['sum']
    })
    feature_names = ['user_days_browse_count', 'user_days_collect_count',
                     'user_days_addcart_count', 'user_days_bought_count']
    group.columns = feature_names

    group.reset_index(inplace=True)
    train = pd.merge(train, group, on=['days', 'user_id'], how='left')

    for feature in feature_names:
        train[feature] = train[feature].astype(np.int16)
        # 只判断2,3,4这三种行为在前几天是否出现
        # 因为如果前几天出现过4那么接下来很有可能不会出现4
        # 相反如果前几天出现2,3,那么接下来这一天就很有可能出现4
        if 'bought' in feature:
            train = Previous_days(train, feature, [1, 2, 3, 4, 5])
        if 'collect' in feature:
            train = Previous_days(train, feature, [1, 2, 3])
        elif 'addcart' in feature:
            train = Previous_days(train, feature, [1, 2, 3])
        if 'browse' in feature:
            train = Previous_days(train, feature, [1])
    # 通过判断前N天用户的行为情况来判断当下用户产生某种行为的可能性
    group = train.groupby(['days', 'user_id', 'category_id']).agg({
                'browse':  ['sum'],
                'collect': ['sum'],
                'addcart': ['sum'],
                'bought':  ['sum']
    })
    feature_names = ['user_days_cate_browse_count', 'user_days_cate_collect_count',
                     'user_days_cate_addcart_count', 'user_days_cate_bought_count']
    group.columns = feature_names
    group.reset_index(inplace=True)
    train = pd.merge(train, group, on=['days', 'user_id', 'category_id'], how='left')
    # 用户在days上面的购买转化率(用来构建滞后值判断前n天的购买转化率侧面反映前n天的是否产生购买情况)
    train['user_days_cate_probability'] = train['user_days_cate_bought_count'] / (train['user_days_cate_browse_count']+
                                                                                  train['user_days_cate_collect_count']+
                                                                                  train['user_days_cate_addcart_count'])
    train = Previous_days(train, 'user_days_cate_probability', [1,2,3,4,5])

    '''
    # 当数据量涉及n个月的时候使用此特征
    # 用户在某月对某类商品的行为总数
    group = train.groupby(['month', 'user_id', 'category_id']).agg({
                'browse':  ['sum'],
                'collect': ['sum'],
                'addcart': ['sum'],
                'bought':  ['sum']
    })
    feature_names = ['user_month_cate_browse_count', 'user_month_cate_collect_count',
                     'user_month_cate_addcart_count', 'user_month_cate_bought_count']
    group.columns = feature_names
    
    group.reset_index(inplace=True)
    train = pd.merge(train, group, on=['month', 'user_id', 'category_id'], how='left')
    
    for feature in feature_names:
        train[feature] = train[feature].astype(np.int16)
    
    train['user_month_probability'] = train['user_month_cate_bought_count'] / (train['user_month_cate_browse_count']+
                                                                              train['user_month_cate_collect_count']+
                                                                              train['user_month_cate_addcart_count'])
    train['user_month_probability'] = train.user_month_probability.astype(np.float64)
    
    for feature in feature_names:
        train.drop(feature, axis=1, inplace=True)
    '''
    # 用户在当前日期的不同时间段的对商品类型进行的聚合总数
    group = train.groupby(['user_id', 'four_time_days', 'category_id']).agg({
            'browse': ['sum'],
            'collect': ['sum'],
            'addcart': ['sum'],
            'bought': ['sum']
    })
    feature_names = ['user_four_time_days_browse_count', 'user_four_time_days_collect_count',
                     'user_four_time_days_addcart_count', 'user_four_time_days_bought_count']
    group.columns = feature_names
    group.reset_index(inplace=True)
    train = pd.merge(train, group, on=['user_id', 'four_time_days', 'category_id'], how='left')
    train['user_four_time_days_probability'] = train['user_four_time_days_bought_count'] / (group['user_four_time_days_browse_count'] +
                                                                                            group['user_four_time_days_collect_count'] +
                                                                                            group['user_four_time_days_addcart_count'])
    train['user_four_time_days_probability'].fillna(0, inplace=True)
    train['user_four_time_days_probability'] = train['user_four_time_days_probability'].astype(np.float16)
    for feature in feature_names:
        train.drop(feature, axis=1, inplace=True)

    train = train.query('days > 5')

    train.fillna(0, inplace=True)
    # 删除部分时间特征
    train.drop(['year'], axis=1, inplace=True)
    # 删除四种行为特征
    train.drop(['browse', 'collect', 'addcart', 'bought'], axis=1, inplace=True)

    for col in train.columns:
        if '_lag_' in col:
            train[col] = train[col] * train['desire_buy']
        if train[col].max() == np.inf:
            train.loc[train[col]==np.inf, col] = 1.0
            train.loc[train[col]==1.0, col] = train[col].mean()
    return train

def get_fit_train_datas(variable):
    if variable==1:
        datas = get_check_fit_datas()
        datas = clean_train(datas)
        # 清洗后的训练数需要保存起来
        try:
            engine = create_engine("mysql+pymysql://root:123456@localhost:3306/mysql?charset=utf8")
            datas.to_sql(name='user_datas_gbdt', con=engine, if_exists='replace',
                          index=False, index_label=False, chunksize=5000)
        except:
            raise ValueError(
                "Data problems"
            )

        return datas
    else:
        raise ValueError(
            "Please check to confirm whether model training is conducted."
        )
