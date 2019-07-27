# coding:utf-8
from __future__ import division

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import re

pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings(action = 'ignore') #忽略警告

import seaborn as sns
import scipy.stats as sts
from sklearn.metrics import classification_report,f1_score
from sklearn.metrics import confusion_matrix
"""
#不规则性可能是很重要的,这种不规则性是特征构造的良好候选者。
datas = pd.read_csv("kaggle\\HousingData\\train.csv")
tests = pd.read_csv("kaggle\\HousingData\\test.csv")
tests_id = tests['Id']
#print(tests.shape,datas.shape)
#查看价格与地理位置以及OverQual与SalePrice的图像后删除那些明显偏离正常情况的数据点(GLA过大但是价格过低)
datas.drop((datas[(datas['OverallQual']<5) & (datas['SalePrice']>3*10**5)].index),inplace=True)
datas.drop((datas[(datas['GrLivArea']>4000) & (datas['SalePrice']<3*10**5)].index),inplace=True)
datas.drop((datas[(datas['TotalBsmtSF']>3000) & (datas['SalePrice']<3*10**5)].index),inplace=True)
#对train数据的一些查看
'''


quantitative = [f for f in datas.columns if datas.dtypes[f] != 'object']
#得到了特征类型为int64的特征列
qualitative = [f for f in datas.columns if datas.dtypes[f] == 'object']
#得到了特征类型为object

for c in qualitative:
    #datas[c].isnull().any()有空为True没有为False
    if datas[c].isnull().any():
        datas[c] = datas[c].fillna('MISSING')

def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x=plt.xticks(size=7,rotation=90)#xlim倾斜度

quat_1 = pd.melt(datas,id_vars=["SalePrice"],value_vars=qualitative[:10])
quat_2 = sns.FacetGrid(quat_1,col="variable",col_wrap=5,sharey=False,sharex=False)
quat_3 = quat_2.map(boxplot,'value',"SalePrice")
#注意object和int的不同在于id_vars和map里的这两个"SalePrice"
#plt.show()

#----------------------
#对特征类型为Int的对其使用普通bar图来比较与结果的关系
#对特征类型为object的对其使用箱型图来比较与结果的关系
#----------------------

quant_1 = pd.melt(datas, value_vars=quantitative[:10])
quant_2 = sns.FacetGrid(quant_1, col="variable",  col_wrap=5, sharex=False, sharey=False)
quant_3 = quant_2.map(sns.distplot, "value")
#plt.show()



y = datas['SalePrice']
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=sts.johnsonsu)
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=sts.norm)
plt.figure(3); plt.title('Log Normal')
sns.distplot(y, kde=False, fit=sts.lognorm)
#plt.show()

#p-value是在原假设成立的前提下，出现与样本相同或者更极端的情况的概率(原假设出现错误的概率与p值成正比)
#方差分析的目的是通过数据分析找出对该事物有显著影响的因素，
#各因素之间的交互作用，以及显著影响因素的最佳水平

def anova(frame):
    f_oneway = pd.DataFrame()
    f_oneway['feature'] = quantitative
    f_oneway_values = []
    for colu in quantitative:
        colu_unix = datas[colu].unique()
        samples = []
        for colu_u in colu_unix:
            colu_unix_valuess = datas[datas[colu]==colu_u]['SalePrice'].values
            samples.append(colu_unix_valuess)
        samples = sts.f_oneway(*samples)[1]
        f_oneway_values.append(samples)
    f_oneway['values'] = f_oneway_values
    return f_oneway.sort_values('values')
object_anova = anova(datas)
print(object_anova.values)
values_log = np.log(1/object_anova['values'].values) #对数
object_anova['values_log'] = values_log
sns.barplot(data=object_anova,x='feature',y='values_log',color='skyblue')
plt.xticks(size=7,rotation=50)
plt.show()
#可以得到value值越小相关性越高（对比下面的通过热力图的得到的结果）

print(datas.shape)
#查看房子价格的核密度图，多集中于10**5~2.5*10**5
sns.kdeplot(y_train)
#plt.show()

#检查大致的数据结构
datas_index = pd.Series(datas[:1].values[0],index=datas.columns)

#plt.scatter(datas['GrLivArea'],datas['SalePrice'],color='skyblue')
plt.title("The price with area")
plt.xlim((0,8000))
plt.ylabel('Saleprice')
plt.xlabel('TotalBsmtSF')
#地下室总面积的绘图方式相同
#plt.show()
'''

#链接train数据和test数据进行数据清理
ntrain = datas.shape[0] #记录datas的数据序数
datas["SalePrice"] = np.log1p(datas["SalePrice"]) #这段代码是关键
y_train = datas.SalePrice.values
#print(y_train)
datas.drop('SalePrice',axis=1,inplace=True)
datas = pd.concat((datas,tests),sort=False).reset_index(drop=True)
datas.drop(['Id'],axis=1,inplace=True)

'''
#pd.corr()计算列的成对相关性，不包括NA / null值
corrmat = datas.corr()
corr_sale_index = corrmat.nlargest(10,'SalePrice')['SalePrice'].index
#print(corr_sale_index)
#获取与SalePrice相关性最强的10个特征,千万注意查看shape并转置
corr_sale =  np.corrcoef(datas[corr_sale_index].values.T)
#np.corrcoef 相关系数计算

print(corr_sale.shape)
plt.figure(figsize=(12, 9),dpi=60)
sns.heatmap(corr_sale, annot=True, #annot参数确保了显示数值
            fmt='.2f',vmax=.8,square=True,
            yticklabels=corr_sale_index.values,
            xticklabels=corr_sale_index.values)
#plt.show()
#由图可知大部分参数都与自身呈现强相关性，但是有几组变量他们之间的的相关性也很高
#因此我们可以依据他们与SalePrice的相关性(或者含有的空值数目)决定保留他们中间的一个
#相关性最强的10个特征

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars',
       '1stFlrSF', 'GarageArea', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
datas_GLA = datas['GrLivArea']
#保留GLA
datas_TRAG = datas['TotRmsAbvGrd']
datas_GC = datas['GarageCars']
#由于与SalePrice相关性都差不多所以互相弥补空值后删减掉一个
datas_GA = datas['GarageArea']

datas_TBS = datas['TotalBsmtSF']
#互相弥补后保留TBS
datas_FS = datas['1stFlrSF']

print (datas[cols].isnull().sum())
#均不含空值因此我们不作处理(测试后不作处理的得分要好一点)
'''

#进行缺失值处理

#缺失值查看
total = datas.isnull().sum().sort_values(ascending=False)
percent = (datas.isnull().sum()/datas.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
#print(missing_data.head(34))
'''
#用于检测指定object类型的特征下SalePrice的分布情况和
#不同值所占的数目
print(datas['BsmtExposure'].isnull().sum())
print(datas.groupby(['BsmtExposure']).count())
sns.boxplot(datas['BsmtExposure'],y_train)
plt.show()
'''
#泳池是决定一个房子价格的因素之一我觉得不能删除(鬼都知道带泳池的房子价格很高)
None_columns = ['PoolQC','FireplaceQu',
                'MiscFeature']
for colu in None_columns:
    datas[colu].fillna("None",inplace=True)

# 篱笆和胡同在乡村很普及在城市却很少相对于总体来说它并不能与房价产生很强的相关性
datas.drop(['Alley', 'Fence'],axis=1,inplace=True)
# 测试后删了的话确实有利于得分
'''
#找到NaN的位置
print (datas['GarageYrBlt'].isnull().sum())
print(datas.loc[datas.GarageYrBlt.isnull()].index)
a = np.array([39,48,78,88,89,99,107,124,126,139,147])
b = ['GarageFinish','GarageYrBlt','GarageQual']
print(datas.loc[a,b])
'''
#关于车库的特征之间有强烈的关联性，GarageYrBlt如果NAN的话意味着车库本身就没有
#所以我保留车库位置、车库状况以及是否有车库，而删除其他关于车库的信息


Garage_ = ['GarageFinish','GarageYrBlt','GarageQual',
           'GarageCond','GarageType']
for i in Garage_:
    datas[i].fillna("None",inplace=True)

datas['GarageCars'].fillna(0,inplace=True)
datas['GarageArea'].fillna(0,inplace=True)


#地下室删除这两个特征保留其他特征
datas.drop(['BsmtFinType1'],axis=1,inplace=True)
datas['BsmtQual'].fillna(datas['BsmtCond'].mode()[0],inplace=True)
datas['BsmtCond'].fillna(datas['BsmtQual'].mode()[0],inplace=True)
datas['BsmtFinType2'].fillna("None",inplace=True)
datas['BsmtExposure'].fillna("No",inplace=True)

bsm_feat = ['BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF',
            'BsmtFullBath', 'BsmtHalfBath','TotalBsmtSF']
for col in bsm_feat:
    datas[col].fillna(0,inplace=True)

#最后三个缺失值MasVnrArea和'MasVnrType'由缺失结果可基本确定两者缺失的都为同一行
datas["MasVnrType"].fillna("None",inplace=True)
datas["MasVnrArea"].fillna(0,inplace=True)
#LotFrontage 同样没有,就是0所以我也将其填充为0
datas['MasVnrArea'].fillna(0,inplace=True)
#按照Neighborhood的分组形式对'LotFrontage'的空值填补median()函数 产生的中间值
datas['LotFrontage'] = datas.groupby('Neighborhood')['LotFrontage'].apply(
    lambda x:x.fillna(x.median())
)
'''
#test.csv内的缺失列
MSZoning         4 一般分区分类
Functional       2 功能性
Utilities        2 公用事业
KitchenQual      1 厨房数量
Exterior1st      1 房屋外墙（普通）
Exterior2nd      1 房屋外墙（多种材料）
SaleType         1 销售类型
'''
cols_test = ['Exterior1st','Exterior2nd','Electrical',
             'KitchenQual','MSZoning','SaleType','Functional']
datas.drop(['Utilities'],axis=1,inplace=True)
for i in cols_test:
    datas[i].fillna(datas[i].mode()[0],inplace=True)


#清洗后对数据进行的操作
'''
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
#数据降维 tsne性能高但是速度慢,pca与之相反
from sklearn.cluster import KMeans
#我不怎么明白要用KMeans，因为他是无监督的算法
from sklearn.preprocessing import StandardScaler
#转化数据为均值为0方差为1


quantitative = [f for f in datas.columns if datas.dtypes[f] != 'object']
#得到了特征类型为int64的特征列
qualitative = [f for f in datas.columns if datas.dtypes[f] == 'object']
#得到了特征类型为object
#聚类
datas_Int = datas[quantitative]
tsne = TSNE(n_components=2,random_state=0,perplexity=50)
#perplexity 此参数非常不敏感
tsne_feature = tsne.fit_transform(datas_Int.values)

std = StandardScaler()
std_feature = std.fit_transform(datas_Int.values)
pca = PCA(n_components=30)
pca_feature = pca.fit_transform(std_feature)

kmeans = KMeans(n_clusters=2)
kmeans.fit(pca_feature)
# explained_variance_ratio_
# 每个所选组件解释的差异百分比

#print("TSNE:\n",tsne_feature)
#print("STD:\n",std_feature)
#print("PCA:\n",pca_feature)
#print(np.sum(pca.explained_variance_ratio_))

#查看现在得数据特征
#print(pca_feature.shape)
print(kmeans.labels_)
dats_show = pd.DataFrame({"tsne_1":tsne_feature[:,0],
                          "tsne_2":tsne_feature[:,1],
                          "cluster":kmeans.labels_})
sns.lmplot(data=dats_show,x='tsne_1',y='tsne_2',hue='cluster',fit_reg=False)
plt.show()
'''

datas['MSSubClass'] = datas['MSSubClass'].apply(str)
#Changing OverallCond into a categorical variable
datas['OverallCond'] = datas['OverallCond'].astype(str)
#Year and month sold are transformed into categorical features.
datas['YrSold'] = datas['YrSold'].astype(str)
datas['MoSold'] = datas['MoSold'].astype(str)

# 顺序变量之间存在固有的顺序 比如 (低, 中, 高) 、
# 病人疼痛指数 ( 1 到 10 - 但是他们之间的差是没有意义的, 因为1 到 10 仅仅表现了顺序)

featur_ = ['GarageQual','PoolQC','ExterCond',
           'GarageCond','BsmtCond']
for f in featur_:
    datas[f] = datas[f].apply(lambda x:0 if x=="Ex" else 1
    if x=="Gd" else 2 if x=="TA" else 3 if x=='Fa' else 4 if x=='None' else 5)
featur1_ = ['BsmtQual','KitchenQual','ExterQual']
for f in featur1_:
    datas[f] = datas[f].apply(lambda x:0 if x=="Ex" else 1 if x=="Gd" else 2 if x=="TA" else 3)
datas['HeatingQC'] = datas['HeatingQC'].map({'Ex':0,'Gd':1,'TA':2,'Fa':3,'Po':4} )
datas['FireplaceQu'] = datas['FireplaceQu'].map({'Ex':0,'Gd':1,'TA':2,'Fa':3,'Po':4,'None':5} )
datas['GarageFinish']=datas['GarageFinish'].apply(lambda x:0 if x=='Fin'
else 1 if x=='RFn' else 2 if x=='Unf' else 3)
datas['LotShape'] = datas['LotShape'].apply(lambda x:0 if x=='Reg'else 1 if x=='IR1'else 2 if x=='IR2'
else 3)
datas['LandSlope'] = datas['LandSlope'].apply(lambda x:0 if x=='Gtl'else 1 if x=='Mod'
else 2)
datas['PavedDrive'] = datas['PavedDrive'].apply(lambda x:0 if x=='Y'else 1 if x=='P' else 2)
def SaleCondition(x):
    if x in ['Normal','AdjLand','Alloca','Partial']:
        r = 0
    else:
        r = 1
    return r
datas['SaleCondition'] = datas['SaleCondition'].apply(SaleCondition)
datas['CentralAir'] = datas['CentralAir'].apply( lambda x: 0 if x == 'N' else 1)
datas['Street'] = datas['Street'].apply( lambda x: 0 if x == 'Pave' else 1)
datas['BldgType'] = datas['BldgType'].apply(lambda x:0 if x=='1Fam'else 1 if x=='TwnhsE'
else 2 if x=='Twnhs'else 3)
datas['MasVnrType'] = datas['MasVnrType'].apply(lambda x:0 if x=='Stone'else 1 if x=='BrkFace'
else 2 if x=='None'else 3 )
datas['Foundation'] = datas['Foundation'].apply(lambda x:0 if x=='PConc'else 1 if x=='CBlock'
else 2 if x=='BrkTil' else 3)
datas['BsmtExposure'] = datas['BsmtExposure'].apply(lambda x:0 if x=='Gd'else 1 if x=='Av'
else 2 if x=='Mn' else 3)
#特征结合生成更重要的特征
datas['HouserArea'] = datas['1stFlrSF']+datas['2ndFlrSF']+datas['TotalBsmtSF']
datas['KitchenWeight'] = datas['KitchenAbvGr']*datas['KitchenQual']

from sklearn.preprocessing import LabelEncoder
cols = ['BsmtFinType2', 'Functional',
         'CentralAir', 'MSSubClass',
        'OverallCond', 'MoSold','ExterCond']
labeE = LabelEncoder()
#对不连续的数字或者文本进行编号
for c in cols:
    labeE.fit(list(datas[c].values)) #list()将数值转换为列表
    datas[c] = labeE.transform(list(datas[c].values))

from scipy.stats import skew
#skew 用于计算偏度值 norm 用于获取平均值和方差
#偏度衡量随机变量概率分布的不对称性,若偏度为负,则 xx 均值左侧的离散度比右侧强;若偏度为正,则 xx均值左侧的离散度比右侧弱

numeric_feats = datas.dtypes[datas.dtypes != "object"].index
print(datas[numeric_feats].head(3))
print(numeric_feats)
#skew 偏度
skewed_feats = datas[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew' :skewed_feats})
#print(skewness)
skewness = skewness[abs(skewness) > 0.75]#abs函数取绝对值
skewed_features = skewness.index
from scipy.special import boxcox1p
#用来转换数组
lam = 0.15
#boxcox1p变换的功率参数

for index in skewed_features:
    datas[index] = boxcox1p(datas[index],lam)
#实话实说转换后的数据看不太懂


datas = pd.get_dummies(datas) #哑变量

#print(datas.isnull().sum().sort_values(ascending=False))

#转成Series方便查看每一项
#print(pd.Series(datas.values[1],index=datas.columns))
#print(datas.shape)
#print(datas.isnull().sum().sort_values(ascending=False))

X_train = datas[:ntrain]
X_test  = datas[ntrain:]

from sklearn.model_selection import KFold, cross_val_score
#验证某个模型在某个训练集上的稳定性，输出k个预测精度(k次)

n_folds = 5
def get_rmse(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train.values)
    rmse = np.sqrt(-cross_val_score(model, X_train.values, y_train,cv=kf,scoring="neg_mean_squared_error"))
    return (rmse)

from sklearn.preprocessing import RobustScaler
#利用中位数和四分位数进行数据转换将其转换到一定范围内
#print (RobustScaler().fit_transform(X_train))
from sklearn.model_selection import cross_val_score
#验证某个模型在某个训练集上的稳定性，输出k个预测精度(k次)
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge,Lasso,ElasticNet,Ridge
#三个模型分别是贝叶斯岭回归：用L1范数 套索回归：用L2范数 弹性网络：都用
from sklearn.pipeline import make_pipeline
#制作管道

BAY = make_pipeline(RobustScaler(),BayesianRidge())

RIG = Ridge(alpha=60)
SVr = make_pipeline(RobustScaler(),SVR(gamma= 0.0004,kernel='rbf',C=13,epsilon=0.009))

LAS = make_pipeline(RobustScaler(),Lasso(alpha =0.0003, random_state=1))
#0.0005的系数使其对数据的影响很小

ENT = make_pipeline(RobustScaler(),ElasticNet(alpha=4.0, l1_ratio=0.005, random_state=3))
#0<l1_ratio<1 是l1和l2范数都用 等于0是l2等于1是l1

from sklearn.kernel_ridge import KernelRidge
#内核岭回归简单来说就是岭回归和核函数(也就是SVR的什么多项式核函数啊，高斯核函数啊)结合l2正则
KRR = make_pipeline(RobustScaler(),KernelRidge(alpha=0.2,kernel='polynomial',degree=2,coef0=2.5))

from sklearn.ensemble import GradientBoostingRegressor
#随机数森和梯度提升(参数类似于决策树)
GBR = GradientBoostingRegressor(n_estimators=3000, #弱学习器的数量(人多力量大) Boosting思想
                                learning_rate=0.02,#学习率
                                max_depth=4, max_features='sqrt',
                                min_samples_leaf=15, min_samples_split=50,
                                loss='huber', random_state =5)

import xgboost as xgb
'''
XGBoost是一个优化的分布式梯度增强库(有监督)
它在GradientBoosting(梯度提升/渐变增强)框架下实现机器学习算法
XGBoost提供了并行树提升
XGBoost有非零数据的指针
'''


XGB = xgb.XGBRegressor(colsample_bytree=0.4603, #构造每个树列的子采样率。子采样将在每次增强迭代中发生一次。
                       gamma=0.0468, #在树的叶节点上进行进一步分区所需的最小损耗减少。越大，算法越保守(0~∞)
                       learning_rate=0.05, #更新使用中的步长缩小以防止过度拟合 (0,1)
                       max_depth=3,
                       min_child_weight=1.7817, #子树所需的权重之和，如果在分树的过程小于该值则停止分树(0~∞)
                       n_estimators=2200, #构建的子树的数量
                       reg_alpha=0.4640, #L1正则化项
                       reg_lambda=0.8571, #L2正则化项
                       subsample=0.5213, #训练时抽取的样本比率
                       silent=1, #1表示不打印训练的信息，0表示打印
                       random_state =7, nthread = -1 #线程数
                       )

import lightgbm as lgb
'''
梯度提升树
'''
LGB = lgb.LGBMRegressor(objective='regression',
                        num_leaves=5, #构建的树中叶子的最大节点
                        learning_rate=0.05, #更新使用中的步长缩小以防止过度拟合 (0,1)
                        n_estimators=650, #构建的子树数量
                        max_bin = 55, #lgb自动调整的内存默认为255
                        bagging_fraction = 0.6, #加速训练和过拟合处理
                        bagging_freq = 5, #bagging_fraction的频率(每k次执行一次)
                        feature_fraction = 0.24, #同bagging_fraction作用类似
                        feature_fraction_seed=9, bagging_seed=9, #两个随机数种子
                        min_data_in_leaf =6, #叶子上的最小数量
                        min_sum_hessian_in_leaf = 11 #叶子上的最小hessian
                        )

#进行验证rmse
model_list = []
model_list.append(("Lasso",LAS))
model_list.append(("ElasticNet",ENT))
model_list.append(("KernelRidge",KRR))
model_list.append(("SVR",SVr))
#model_list.append(("GradientBoostingRegressor",GBR))
#model_list.append(("xgboost",XGB))
#model_list.append(("lightgbm",LGB))

for name,model in model_list:
    score = get_rmse(model)
    print("this name :",name)
    print("this score : {:.4f},{:.4f}".format(score.mean(),score.std()))
#就本次计算结果来说，线性模型要好于树


'''
数据清洗后常规模型得分
                12.10               12.12
name            score
Lasso           0.1108,0.0053       0.1103
ElasticNet      0.1108,0.0053       0.1104
KernelRidge     0.1109,0.0059       0.1101
SVR             0.1104,0.0074       0.1106
GRegressor      0.1168,0.0076       0.1168
xgboost         0.1165,0.0051       0.1152
lightgbm        0.1151,0.0074       0.1142
'''


from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
#clone:克隆在估算器中执行模型的深层复制,产生新的估算器(模型)且不与数据进行拟合fit确保他们之前没有看到任何数据
#RegressionMixin:回归估计 TransformerMixin:作为基类,自动实现fit_transform()函数
#BaseEstimator:所有估算器的基类


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    #类似于构建一个pipeline
    #清理封装和代码重用（继承）
    def __init__(self,models):
        self.models = models
    def fit(self,X,y):
        self.models_ = [clone(x) for x in self.models]
        for model in self.models_:
            model.fit(X,y)
        return self
    def predict(self,X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)

models_Averag = AveragingModels(models=(KRR,ENT,LAS,GBR))
#print("this is AverageModel:\n",models_Averag,"this is AverageModel's vars:\n",vars(models_Averag))
#Averag_score = get_rmse(models_Averag)
#print("this Average score :{:.4f} , {:.4f}".format(Averag_score.mean(),Averag_score.std()))
# KRR,SVr,ENT,LAS,GBR the score :0.1081 , 0.0072


from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
class StackAveragingModels(BaseEstimator,RegressorMixin,TransformerMixin):
    def __init__(self,base_models,meta_model,n_folds):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
    def fit(self,X,y):
        self.base_models_ = [list() for x in self.base_models]
        #训练模型组
        self.meta_model_ = clone(self.meta_model)
        #元模型
        kfold = KFold(n_splits=self.n_folds,shuffle=True,random_state=123)
        get_predict_data = np.zeros((X.shape[0],len(self.base_models)))
        for i,model in enumerate(self.base_models):
            for train_index,test_index in kfold.split(X,y):
                #注意kfold.split()在此时有效
                #train_index:拆分后的训练数据的index, test_index:拆分后的测试数据集index
                model_base = clone(model)
                self.base_models_[i].append(model_base)
                model_base.fit(X[train_index],y[train_index])
                y_pread = model_base.predict(X[test_index])

                get_predict_data[test_index,i] = y_pread
                #print("----------------------")
                #查看后知道每n_folds填满一列数据，几个模型占几列数据
                #print(get_predict_data)
        #print(get_predict_data.shape)
        self.meta_model_.fit(get_predict_data,y)
        return self
    def predict(self,X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        #将训练模型组predict的数据堆叠在一起
        #print(meta_features.shape)
        return self.meta_model_.predict(meta_features)

model_stack = StackAveragingModels(base_models=(ENT,KRR,GBR),meta_model=LAS,n_folds=5)
#print ("this is StackModel:\n",model_stack,"this is StackModel's vars:\n",vars(model_stack))
#Stack_score = get_rmse(model_stack)
#print ("this stackscore : {:.4f},{:.4f}".format(Stack_score.mean(),Stack_score.std()))

from sklearn.metrics import mean_squared_error
def rmsle(y,y_predict):
    return np.sqrt(mean_squared_error(y, y_predict))


model_stack.fit(X_train.values, y_train)
stacked_predict = model_stack.predict(X_train.values)

stacked_predict_expm1 = np.expm1(model_stack.predict(X_test.values))
#expm1与log1p对应将数据还原到初始的格式

models_Averag.fit(X_train.values,y_train)
average_predict = models_Averag.predict(X_train.values)

average_predict_expm1 = np.expm1(models_Averag.predict(X_test.values))

XGB.fit(X_train.values,y_train)
XGB_predict = XGB.predict(X_train.values)

XGB_predict_expm1 = np.expm1(XGB.predict(X_test.values))
#XGB_score = rmsle(y_train,XGB_predict)
#print("XGB socres : {:.4f}".format(XGB_score))


LGB.fit(X_train.values,y_train)
LGB_predict = LGB.predict(X_train.values)

LGB_predict_expm1 = np.expm1(LGB.predict(X_test.values))

#LGB_score = rmsle(y_train,LGB_predict)
#print("LGB socres : {:.4f}".format(LGB_score))

print('RMSLE score on train data:')

print(rmsle(y_train,stacked_predict*0.9 + XGB_predict_expm1*0.05 + LGB_predict_expm1*0.05 ))


y_test = (stacked_predict_expm1*0.9 +LGB_predict_expm1*0.05+
         XGB_predict_expm1*0.05)

end_datas = pd.DataFrame()
end_datas['Id'] = tests_id
end_datas['SalePrice'] = y_test
end_datas.to_csv('kaggle\\HousingData\\y_tests_x.csv',index=False)
"""