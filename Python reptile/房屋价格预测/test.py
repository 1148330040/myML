#coding:gbk

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
#���þ�����
# ������(Decision Tree)������֪��������������ʵĻ����ϣ�
# ͨ�����ɾ���������ȡ����ֵ������ֵ���ڵ�����ĸ��ʣ�������Ŀ���գ�
# �ж�������Եľ��߷�����������ֱ�����ø��ʷ�����һ��ͼ�ⷨ
from sklearn.metrics import mean_absolute_error
#ƽ����������ʧ
#ƽ����������ܸ��õط�ӳԤ��ֵ����ʵ�����
from sklearn.model_selection import train_test_split

data=pd.read_csv("E:\melb_data.csv")
desc1=data.describe()
#columns_of_interest = ['Landsize', 'BuildingArea']
column=data.columns
columns_of_interest = ['Landsize', 'BuildingArea']
desc2=data[columns_of_interest].describe()
y=data.Price
melbourne_predictors = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X=data[melbourne_predictors]


# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
print (melbourne_model.fit(X, y))

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))
predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)



# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
'''
train_test_split(X,y, test_size=0.3,random_state=0)
�������ͣ�
X�������ֵ�����������
y�������ֵ�������ǩ
random_state���������������(Ҳ����ȷ��ÿ����������ı�����Ӷ��������仯)
'''
#���ػ��ֺõ�ѵ�������Լ�������ѵ�������Լ���ǩ
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
#print ("\n1:\n",train_X.head(),"\n2:\n", val_X.head(),"\n3:\n", train_y.head(),"\n4:\n", val_y.head())
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)
#����ͨ��train_X,train_yѵ�����ģ��
# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
#�˴���val_predictions����ͨ��melbourne_modelԤ��val_x��Priceֵ�б�
print(mean_absolute_error(val_y, val_predictions))
#mean_absolute_error��y_true��y_pred��sample_weight = None��multioutput ='uniform_average' ��