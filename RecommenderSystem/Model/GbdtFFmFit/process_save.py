# coding:gbk
'''
此文件的目的
处理准备放入训练数据库的数据
'''
import os

def del_files(path):
    for root,dirs,files in os.walk(path):
            for name in files:
                if '_log' in name:
                 os.remove(os.path.join(root,name))

def concat_data(new_user_data, feature, num):
    '''
    次函数的目的
    将新用户的数据变成训练数据的格式
    即将xx_type最终合并成只有一列behavior_type
    '''
    columns = ['user_id', 'item_id','category_id', feature, 'time']
    datas = new_user_data[columns].copy()
    datas = datas.loc[datas[feature] == '1']
    datas.rename(columns = {feature:'behavior_type'}, inplace=True)
    datas['behavior_type'] = num
    return datas
