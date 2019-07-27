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

# �����ͺ�ֵ days
def Previous_days(df, feature, nums):
    tmp = df[['days', 'user_id', feature]]
    for i in nums:
        shifted = tmp.copy()
        shifted.columns = ['days', 'user_id', feature+'_lag_'+str(i)]
        # Ϊ��Ҫ��shifted����ȥ�أ���Ϊ���ͺ���ͬ����Ʒ��item_id�����ǲ�ͬ��,����������Ե�����Ʒ���ͣ�
        # �����ֲ���Ϊ�������Ʒ��id��͵�����shifted���������ͬ�������ݣ�ʵ�������ǵ�item_id�ǲ�ͬ�ģ�
        # �������ֻ�Ǳ�����ͬ����͵������ںʹ���item_id��df����ԭʼ���ݺϲ���ʱ������ֻ�����Ʒ���ͺ�days��
        # ���Ե�����shifted��һ�����ݿ�������df�н����˶�εĺϲ�����˳����˴������ظ���
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
    train['week'] = releaseDate.dt.dayofweek.astype(np.int8) #��ȡ��ǰ�������ܼ�
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

    # ��ȡ�û�����Ϊ��
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


    # ���������û�������Ӱ��
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
    # �Ƚ�bug�û���id��Ϊindex,Ȼ��ͨ��ɾ��ָ����index�������ɾ�������û�
    train.index = train.user_id
    train.drop(index=bug_user, inplace=True)


    # ��Ʒ������Ϊ��������
    # ������ֵ��������ȡĳ����Ʒ�Ĺ���ת����
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


    # �ܵĹ�����Ϊ��������Ϊ֮��
    #-----------------------------------
    train['cate_probability'] = train.cate_borwse_count+train.cate_collect_count+train.cate_addcart_count
    train['cate_probability'] = train.cate_bought_count/train.cate_probability
    train['cate_probability'] = train.cate_probability.astype(np.float32)
    train.loc[train.cate_probability==np.inf, 'cate_probability'] = 1.0
    train.loc[train.cate_probability==1.0, 'cate_probability'] = train.cate_probability.mean()

    # �����ںžۺ�
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

    # ��Ʒ�����ںŽ������;ۺϵ���Ϊ����
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

    # ĳһ��Ʒ���ںŽ��е���Ϊ����ռ���ں�����Ʒ��Ϊ�����ı�ֵ->�ɵõ�����Ʒ�����ڵ�ǰ���ں��ܻ�ӭ�ĳ̶�
    train['cate_allcate_week_rate'] = (train['cate_week_browse_count']+train['cate_week_collect_count']+
                                       train['cate_week_addcart_count']+train['cate_week_bought_count'])

    train['cate_allcate_week_rate']=train['cate_allcate_week_rate']/(train['week_browse_count']+train['week_collect_count']+
                                                                     train['week_addcart_count']+train['week_bought_count'])

    #-----------------------------------
    # ���嵽���ںŲ�Ʒ�Ĺ���ת����
    train['cate_week_probability'] = train.cate_week_browse_count+train.cate_week_collect_count+train.cate_week_addcart_count
    train['cate_week_probability'] = train.cate_week_bought_count/train.cate_week_probability
    train['cate_week_probability'] = train.cate_week_probability.astype(np.float64)
    train.loc[train.cate_week_probability==np.inf, 'cate_week_probability'] = 1.0
    train.loc[train.cate_week_probability==1.0, 'cate_week_probability'] = train.cate_week_probability.mean()

    for feature in feature_names:
        train.drop(feature, axis=1, inplace=True)
    for feature in feature_names_week:
        train.drop(feature, axis=1, inplace=True)


    # ��Ʒ���Ͱ����ھۺϵ���Ϊ��Ŀ
    # ���������������ͺ�ֵ

    # �����ںŽ��оۺϵ���Ϊ������ע��������day
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
    # ��ͼ�������Կ���˫12��һ����Ϊ����ͻ���������һЩ�����������Ҫ����ĸ���Ȩֵ

    special_days = [30]
    train['special_days'] = train.days.apply(lambda x: 10 if x in special_days else 1)

    # ע��������day����days
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


    # ĳ����Ʒ�����ںŽ��е���Ϊ����ռ���ں�����Ʒ���͵���Ϊ����->�ɵõ�����Ʒ�ڵ����ܻ�ӭ�ĳ̶�
    train['cate_allcate_day_rate'] = (train['cate_day_browse_count']+train['cate_day_collect_count']
                                     +train['cate_day_addcart_count']+train['cate_day_bought_count'])

    train['cate_allcate_day_rate'] = train['cate_allcate_day_rate'] / (train['day_browse_count']+train['day_collect_count']
                                                                      +train['day_addcart_count']+train['day_bought_count'])

    # ���ں�dayĳ����Ʒ�Ĺ���ת����
    train['cat_day_probability'] = train['cate_day_bought_count'] / (train['cate_day_browse_count']+
                                                                    train['cate_day_collect_count']+
                                                                    train['cate_day_addcart_count'])
    train['cat_day_probability'] = train.cat_day_probability.astype(np.float64)

    for feature in feature_names_day:
        train.drop(feature, axis=1, inplace=True)
    for feature in feature_names:
        train.drop(feature, axis=1, inplace=True)


    # �û���֮ǰһ��ʱ���ڰ�����Ʒ���͵���Ϊ�ܺ���36(�����ܳ�)�ı�ֵ ���淴ӳ���û������ʱ���ڻ�Ծ��
    # �������������Ļ�(����һ������12����)��groupby�������·���������

    train['user_behavior_activity'] = train.groupby(['user_id'])['behavior_type'].transform('sum') / 36

    train['user_category_behavior_activity'] = train.groupby(['user_id', 'category_id'])['behavior_type'].transform('sum') / 36

    train["user_item_behavior_activity"] = train.groupby(['user_id', 'item_id'])['behavior_type'].transform('sum') / 36

    # �����û�������
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


    # �û��Ĺ���ת����
    #-----------------------------------
    train['user_probability'] = train.user_browse_count+train.user_collect_count+train.user_addcart_count
    train['user_probability'] = train.user_bought_count/train.user_probability
    train['user_probability'] = train.user_probability.astype(np.float64)
    train.loc[train.user_probability==np.inf, 'user_probability'] = 1.0
    train.loc[train.user_probability==1.0, 'user_probability'] = train.user_probability.mean()

    # �û�������
    train['desire_buy'] = train['user_browse_count']+train['user_collect_count']+train['user_addcart_count']+train['user_bought_count']
    four_nums = train.describe().desire_buy.values[-4:]

    train['desire_buy'] = train['desire_buy'].apply(lambda x: 0.5 if x<=(four_nums[0]//2)
                                                    else 1 if x<(four_nums[2]+four_nums[3])//2
                                                    else 4
                                                    )

    #�û��ϴβ�������ʲô��Ϊ
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

    # �ۺ��û���ĳ����Ʒ�ľۺ�ֵ
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

    # �û���ĳ����Ʒ�Ĺ�����Ϊ��������Ʒ��������Ϊ�ı�(��������Ϊת��Ϊ������Ϊ�ı���)
    train['user_cate_probability'] = train.user_cate_browse_count+train.user_cate_collect_count+train.user_cate_addcart_count
    train['user_cate_probability'] = train.user_cate_bought_count/train.user_cate_probability
    train['user_cate_probability'] = train.user_cate_probability.astype(np.float64)


    # �û���ĳ����Ʒ����Ϊ������û��ܵ���Ϊ�ı�->���ж��û���ĳ����Ʒ��ƫ�ó̶�
    train['user_cate_allcate_rate'] = (train['user_cate_browse_count']+train['user_cate_collect_count']+
                                       train['user_cate_addcart_count']+train['user_cate_bought_count'])
    train['user_cate_allcate_rate'] = train['user_cate_allcate_rate']/(train['user_browse_count']+train['user_collect_count']+
                                                                       train['user_addcart_count']+train['user_bought_count'])

    for feature in feature_names_user:
        train.drop(feature, axis=1, inplace=True)
    for feature in feature_names:
        train.drop(feature, axis=1, inplace=True)


    # �û������ںŵ���Ϊ����
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
    # �û������ںŵĹ���ת����
    train['user_week_probability'] = train['user_week_bought_count'] / (train['user_week_browse_count']+
                                                                        train['user_week_addcart_count']+
                                                                        train['user_week_collect_count'])


    # �û������ںŶ�ĳ����Ʒ����Ϊ����
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
    # ���ں��û���ĳ����Ʒƫ�ó̶�
    train['user_week_cate_allcate_rate'] = train['user_week_cate_allcate_rate']/(train['user_week_browse_count'] + train['user_week_collect_count']+
                                                                                 train['user_week_addcart_count'] + train['user_week_bought_count'])


    train['user_week_cate_allcate_rate'] = train.user_week_probability.astype(np.float64)

    for feature in feature_names_week_user:
        train.drop(feature, axis=1, inplace=True)

    '''
    # �û������ں�(day)���е���Ϊ������ע����day��Ϊ�û���Ե�������(����˵10��,��ô����һ�����12��10��)
    # ������ͬ������·���Ŀ�������
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
    
    # �û������ں�(day)�Ĺ���ת����
    # ���ڶ̣���˲�����ƫ�ó̶ȵĽ���
    train['user_day_probability'] = train['user_day_bought_count'] / (train['user_day_browse_count']+
                                                                      train['user_day_collect_count']+
                                                                      train['user_day_addcart_count'])
    
    '''


    # ͨ���ж�ǰN���û�����Ϊ������жϵ����û�����ĳ����Ϊ�Ŀ�����
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
        # ֻ�ж�2,3,4��������Ϊ��ǰ�����Ƿ����
        # ��Ϊ���ǰ������ֹ�4��ô���������п��ܲ������4
        # �෴���ǰ�������2,3,��ô��������һ��ͺ��п��ܳ���4
        if 'bought' in feature:
            train = Previous_days(train, feature, [1, 2, 3, 4, 5])
        if 'collect' in feature:
            train = Previous_days(train, feature, [1, 2, 3])
        elif 'addcart' in feature:
            train = Previous_days(train, feature, [1, 2, 3])
        if 'browse' in feature:
            train = Previous_days(train, feature, [1])
    # ͨ���ж�ǰN���û�����Ϊ������жϵ����û�����ĳ����Ϊ�Ŀ�����
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
    # �û���days����Ĺ���ת����(���������ͺ�ֵ�ж�ǰn��Ĺ���ת���ʲ��淴ӳǰn����Ƿ�����������)
    train['user_days_cate_probability'] = train['user_days_cate_bought_count'] / (train['user_days_cate_browse_count']+
                                                                                  train['user_days_cate_collect_count']+
                                                                                  train['user_days_cate_addcart_count'])
    train = Previous_days(train, 'user_days_cate_probability', [1,2,3,4,5])

    '''
    # ���������漰n���µ�ʱ��ʹ�ô�����
    # �û���ĳ�¶�ĳ����Ʒ����Ϊ����
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
    # �û��ڵ�ǰ���ڵĲ�ͬʱ��εĶ���Ʒ���ͽ��еľۺ�����
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
    # ɾ������ʱ������
    train.drop(['year'], axis=1, inplace=True)
    # ɾ��������Ϊ����
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
        # ��ϴ���ѵ������Ҫ��������
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
