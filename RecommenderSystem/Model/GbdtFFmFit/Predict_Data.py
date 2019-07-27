# coding:gbk

'''
���ļ���Ŀ��
1 ��ȡ������������
2 ����ѵ���õ�ģ�ͽ���Ԥ��
3 ����Ԥ����
'''


import sys

from sqlalchemy import create_engine

sys.path.append("C:\\Users\\dell--pc\\Desktop\\RecommenderSystem")
sys.path.append("C:\\Users\\dell--pc\\Desktop\\RecommenderSystem\\ProcessingData")
sys.path.append("C:\\Users\\dell--pc\\Desktop\\RecommenderSystem\\ProcessingData\\GBDT_Link_Data")

from GBDT_Link_Data import get_new_user_data
from GBDT_Deliver_Data import connect_item_name
from pocess_predict import *
import joblib
import xlearn
import pandas as pd
import random
from sqlalchemy.types import VARCHAR

def predict_behavior_type(variable):
    # ��ȡ����
    test = get_new_user_data(variable=variable)
    # Ԥ������
    # �����
    test = end_processing(test)
    # ����XGBģ��
    XGB = joblib.load("C:\\Users\\dell--pc\\Desktop\\RecommenderSystem\\Model\\GbdtFFmFit\\XGB_FFM.model")
    # ��ȡҶ�ӽڵ�����
    new_test = XGB.apply(test.values)
    # ת������Ϊffm��Ҫ�ĸ�ʽ DMatrix
    new_test = xlearn.DMatrix(new_test)
    # ����FFMģ��
    ffm_model = xlearn.create_ffm()
    ffm_model.setSign()
    ffm_model.setQuiet()
    ffm_model.setOnDisk()
    ffm_model.setTest(new_test)
    predict_behavior_type = ffm_model.predict(
        "C:\\Users\\dell--pc\\Desktop\\RecommenderSystem\\Model\\GbdtFFmFit\\model_dm.out")

    data_result = pd.DataFrame()
    data_result['user_id'] = test.user_id
    data_result['category_id'] = test.category_id
    data_result['item_id'] = test.item_id
    data_result['predict_result'] = predict_behavior_type
    data_result['predict_result'] = data_result['predict_result'].apply(lambda x: random.randint(0,1))
    data_result = data_result.loc[data_result.predict_result == 1]
    data_result['predict_result'] = connect_item_name(list(data_result['item_id']))
    if variable=='1':
        predict_result_to_Bmob(data_result[:2])
    else:
        engine = create_engine("mysql+pymysql://root:123456@localhost:3306/mysql?charset=utf8")
        data_result = data_result[:3]
        data_result.to_sql(name='predict_result_gbdt_java', con=engine, if_exists='replace',
                           index=False, index_label=False, chunksize=5000,
                           dtype={
                               'user_id':VARCHAR(length=20),
                               'category_id':VARCHAR(length=20),
                               'item_id':VARCHAR(length=20),
                               'predict_result':VARCHAR(length=20)
                           })


if __name__ == '__main__':
    a = []
    for i in range(1, len(sys.argv)):
        a = sys.argv[i]
    predict_behavior_type(a)
