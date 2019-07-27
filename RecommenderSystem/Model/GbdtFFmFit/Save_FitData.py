# conding:gbl

'''
此文件的目的
用于将新用户结束的一次过程保存在
训练用数据表之内
'''


import sys

sys.path.append("C:\\Users\\dell--pc\\Desktop\\RecommenderSystem")
sys.path.append("C:\\Users\\dell--pc\\Desktop\\RecommenderSystem\\ProcessingData")
sys.path.append("C:\\Users\\dell--pc\\Desktop\\RecommenderSystem\\ProcessingData\\GBDT_Deliver_Data")

from GBDT_Deliver_Data import deliver_new_user_data_java, deliver_new_user_data_and
from sqlalchemy import create_engine
import pandas as pd
from process_save import *


def save_fit_datas(variable=None):
    # variable=1指Android
    if str(variable)=='2':
        columns = ['browse_type', 'collect_type', 'addcart_type', 'bought_type']
        new_user_data = deliver_new_user_data_and()
        fit_user_datas = pd.DataFrame()
        for num, feature in enumerate(columns):
            datas = concat_data(new_user_data, feature, num+1)
            fit_user_datas = pd.concat([fit_user_datas, datas], ignore_index=False, sort=True)
    else :
        fit_user_datas = deliver_new_user_data_java()
    engine = create_engine("mysql+pymysql://root:123456@localhost:3306/mysql?charset=utf8")
    fit_user_datas.to_sql(name='fit_datas_gbdt', con=engine, if_exists='append',
                     index=False, index_label=False, chunksize=50)
    del_files(path="C:\\Users\\dell--pc\\Desktop\\RecommenderSystem\\Model\\GbdtFFmFit")


if __name__ == '__main__':
    a = []
    for i in range(1, len(sys.argv)):
        a = sys.argv[i]
    save_fit_datas(a)