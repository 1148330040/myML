# coding:gbk

'''
此文件的目的
1 python与mysql的链接中转
2 传递不同的数据用于处理
3 获取Bmob云数据服务器的Android数据
'''

import pymysql
import pandas as pd
import cx_Oracle
pd.set_option('display.max_columns', None)
from Bmob.bmob import *



# 数据库连接参数
def link_mysql_db():
    params = {
        'host':'localhost',
        'port':3306,
        'db': 'mysql',
        'user':'root',
        'password':'123456', # 注意是字符串
        'charset':'utf8',
        'cursorclass':pymysql.cursors.DictCursor
    }

    return params

# 传递new_user_data
def deliver_new_user_data_and():
    # 链接Bmob
    b = Bmob("dbac9a625c262e782e519cb092c26f91", "661aceb04b058aeb0b5cb9e6feb5f24e")
    new_user_data = b.find(
        "Commodity"  # 表名
    ).queryResults  # 输出string格式的内容

    new_user_data = pd.DataFrame(new_user_data, columns=['objectId', 'user_id', 'item_id',
                                                         'category_id', 'browse_type', 'collect_type',
                                                         'addcart_type','bought_type',
                                                         'createdAt', 'updatedAt'
                                                         ])
    new_user_data.rename(columns={'updatedAt': 'time'}, inplace=True)
    new_user_data.drop(['createdAt', 'objectId'], axis=1, inplace=True)
    return new_user_data
def deliver_new_user_data_java():
    params = link_mysql_db()
    db = pymysql.connect(**params)
    columns = ['user_id', 'item_id', 'category_id',
               'behavior_type', 'time']
    try:
        with db.cursor() as cursor:
            sql = "select * from new_user_data_gbdt limit 0,100"
            try:
                cursor.execute(sql)
                result = cursor.fetchall()
                test_datas = pd.DataFrame(result)
                test_datas = test_datas[columns]
                return test_datas
            except:
                raise ValueError(
                    "Please check your database to make sure"
                    "that there is data in the table."
                )
    finally:
        db.close()


def connect_item_name(item_ids):
    params = link_mysql_db()
    db = pymysql.connect(**params)
    result = list()
    try:
        with db.cursor() as cursor:
            for id in item_ids:
                sql = "select item_name from item_table where item_id = " + str(id)
                try:
                    cursor.execute(sql)
                    result.append(cursor.fetchone()['item_name'])
                except:
                    raise ValueError(
                        "There is no name for the product"
                    )
            return result
    finally:
        db.close()



# 传递关于user_id的train_data 用于合并
def deliver_user_datas(user_ids, new_user_data):
    paramse = link_mysql_db()
    db = pymysql.connect(**paramse)
    columns = ['behavior_type', 'category_id', 'item_id', 'user_id', 'month', 'day',
       'week', 'hour', 'days', 'four_time_days', 'browse_type', 'collect_type',
       'addcart_type', 'bought_type', 'cate_borwse_count',
       'cate_collect_count', 'cate_addcart_count', 'cate_bought_count',
       'cate_probability', 'cate_allcate_week_rate', 'cate_week_probability',
       'special_days', 'cate_allcate_day_rate', 'cat_day_probability',
       'user_behavior_activity', 'user_category_behavior_activity',
       'user_item_behavior_activity', 'user_probability', 'desire_buy',
       'user_category_first_click_browse', 'user_category_first_click_collect',
       'user_category_first_click_addcart', 'user_category_first_click_bought',
       'user_cate_probability', 'user_cate_allcate_rate',
       'user_week_probability', 'user_week_cate_browse_count',
       'user_week_cate_collect_count', 'user_week_cate_addcart_count',
       'user_week_cate_bought_count', 'user_week_cate_allcate_rate',
       'user_days_browse_count', 'user_days_collect_count',
       'user_days_addcart_count', 'user_days_bought_count',
       'user_days_browse_count_lag_1', 'user_days_collect_count_lag_1',
       'user_days_collect_count_lag_2', 'user_days_collect_count_lag_3',
       'user_days_addcart_count_lag_1', 'user_days_addcart_count_lag_2',
       'user_days_addcart_count_lag_3', 'user_days_bought_count_lag_1',
       'user_days_bought_count_lag_2', 'user_days_bought_count_lag_3',
       'user_days_bought_count_lag_4', 'user_days_bought_count_lag_5',
       'user_days_cate_browse_count', 'user_days_cate_collect_count',
       'user_days_cate_addcart_count', 'user_days_cate_bought_count',
       'user_days_cate_probability', 'user_days_cate_probability_lag_1',
       'user_days_cate_probability_lag_2', 'user_days_cate_probability_lag_3',
       'user_days_cate_probability_lag_4', 'user_days_cate_probability_lag_5',
       'user_four_time_days_probability']
    try:
        with db.cursor() as cursor:
            result = pd.DataFrame()
            for id in user_ids:
                sql = "select * from user_datas_gbdt where user_id = " + str(id)
                try:
                    cursor.execute(sql)
                    r_2 = pd.DataFrame(cursor.fetchall())
                except:
                    # There is no data of the user in the user_datas
                    # So use the original data
                    # Connect and fill with the user_datas's mean value.
                    r_2 = pd.DataFrame(new_user_data.loc[new_user_data.user_id==id])
                result = pd.concat([result, r_2], ignore_index=False, sort=True)
    finally:
        db.close()
    try:
        user_datas = pd.DataFrame(result[columns])
    except:
        raise ValueError(
            "These user_id are all new"
            "so choose to give up prediction"
        )
    return user_datas

# 传递关于category_id的train_data用于合并
def deliver_cate_datas(category_ids, new_user_data):
    paramse = link_mysql_db()
    db = pymysql.connect(**paramse)
    columns_cate = ['category_id', 'day', 'week',
                    'cate_borwse_count', 'cate_collect_count', 'cate_addcart_count',
                    'cate_bought_count', 'cate_probability', 'cate_allcate_week_rate',
                    'cate_week_probability', 'cate_allcate_day_rate', 'cat_day_probability']
    try:
        with db.cursor() as cursor:
            result = pd.DataFrame()
            for id in category_ids:
                sql = "select distinct category_id,day,week,cate_borwse_count,cate_collect_count," \
                      "cate_addcart_count,cate_bought_count,cate_probability," \
                      "cate_allcate_week_rate,cate_week_probability," \
                      "cate_allcate_day_rate,cat_day_probability from user_datas_gbdt where category_id = "+str(id)
                try:
                    cursor.execute(sql)
                    c_2 = pd.DataFrame(cursor.fetchall())
                except:
                    # There is no data of the category in the user_datas
                    # So use the original data
                    # Connect and fill with the mean value."
                    c_2 = pd.DataFrame(new_user_data.loc[new_user_data.category_id==id])
                result = pd.concat([result, c_2], ignore_index=False, sort=False)
    finally:
        db.close()
    try:
        cate_datas=pd.DataFrame(result[columns_cate])
    except:
        raise ValueError(
            "These category_id are all new,"
            "so choose to give up prediction"
        )
    return cate_datas

# 获取最新的15天内的所有用户产生的数据
def deliver_train_data():
    params = link_mysql_db()
    db = pymysql.connect(**params)
    try:
        with db.cursor() as cursor:
            sql = "select * from fit_datas_gbdt"
            try:
                cursor.execute(sql)
                result = cursor.fetchall()
                train_data = pd.DataFrame(result)
                return train_data
            except:
                raise ValueError(
                    "Please check your database to make sure"
                    "that there is data in the table."
                )
    finally:
        db.close()


# 判断15天之内的数据是否有问题
# 将数据传递给GBDT_FFm.py文件用于训练模型
def get_check_fit_datas():
    user_datas = deliver_train_data()
    base_feature = ['user_id', 'item_id', 'behavior_type', 'category_id', 'time']
    if len(user_datas.columns) != len(base_feature):
        raise LookupError(
            "Check that the data feature length if is five"
        )
    for feature in user_datas.columns:
        if feature not in base_feature:
            raise ValueError(
                "Make sure the features belong to "
                "['user_id', 'item_id', 'behavior_type', 'category_id', 'time']"
            )
    return user_datas
'''
from matplotlib import pyplot as plt
import numpy as np
def plt_parameter(_2,_2std,_3,_3std,names1,names2):
    X_axis = [4,6,10,14,18,20]

    ind = np.arange(len(X_axis))  # the x locations for the groups
    width = 0.4  # the width of the bars

    fig, ax = plt.subplots()

    rects1 = ax.bar(ind - width/2, _2, width/2,yerr=_2std,
                    color='SkyBlue', label='precision')
    rects2 = ax.bar(ind, _3, width/2,yerr=_3std,
                    color='IndianRed', label='recall')

    ax.set_ylabel(names1)
    ax.set_xlabel(names2+"  %ten thousand")
    ax.set_xticks(ind)
    y_ticks = np.linspace(0,1.2,4)
    ax.set_yticks(y_ticks)
    ax.set_title("this is parameters of "+names2)
    ax.set_xticklabels(X_axis)
    ax.legend()
    def autolabel(rects, xpos='center'):
        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                    '{}'.format(height), ha=ha[xpos], va='bottom')
    autolabel(rects1, "left")
    autolabel(rects2)
    plt.show()


precision,p_std = (0.91,0.79,0.83,0.70,0.74,0.85),(0.03,0.04,0.05,0.02,0.01,0.02)
recall,r_std = (0.82,0.75,0.84,0.74,0.81,0.83),(0.04,0.04,0.05,0.02,0.03,0.02)
plt_parameter(precision, p_std, recall, r_std, 'score' ,'datas')
'''


