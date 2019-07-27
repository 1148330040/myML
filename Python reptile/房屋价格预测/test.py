#coding:gbk

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
#调用决策树
# 决策树(Decision Tree)是在已知各种情况发生概率的基础上，
# 通过构成决策树来求取净现值的期望值大于等于零的概率，评价项目风险，
# 判断其可行性的决策分析方法，是直观运用概率分析的一种图解法
from sklearn.metrics import mean_absolute_error
#平均误差绝对损失
#平均绝对误差能更好地反映预测值误差的实际情况
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
参数解释：
X：被划分的样本特征集
y：被划分的样本标签
random_state：是随机数的种子(也就是确保每次运行随机的编号种子都不发生变化)
'''
#返回划分好的训练集测试集样本和训练集测试集标签
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
#print ("\n1:\n",train_X.head(),"\n2:\n", val_X.head(),"\n3:\n", train_y.head(),"\n4:\n", val_y.head())
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)
#这里通过train_X,train_y训练拟合模型
# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
#此处的val_predictions就是通过melbourne_model预测val_x的Price值列表
print(mean_absolute_error(val_y, val_predictions))
#mean_absolute_error（y_true，y_pred，sample_weight = None，multioutput ='uniform_average' ）