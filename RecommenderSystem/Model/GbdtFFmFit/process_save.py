# coding:gbk
'''
���ļ���Ŀ��
����׼������ѵ�����ݿ������
'''
import os

def del_files(path):
    for root,dirs,files in os.walk(path):
            for name in files:
                if '_log' in name:
                 os.remove(os.path.join(root,name))

def concat_data(new_user_data, feature, num):
    '''
    �κ�����Ŀ��
    �����û������ݱ��ѵ�����ݵĸ�ʽ
    ����xx_type���պϲ���ֻ��һ��behavior_type
    '''
    columns = ['user_id', 'item_id','category_id', feature, 'time']
    datas = new_user_data[columns].copy()
    datas = datas.loc[datas[feature] == '1']
    datas.rename(columns = {feature:'behavior_type'}, inplace=True)
    datas['behavior_type'] = num
    return datas
