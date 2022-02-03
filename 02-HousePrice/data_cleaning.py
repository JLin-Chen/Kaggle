import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from scipy.special import boxcox1p,inv_boxcox1p
from sklearn.preprocessing import RobustScaler
"""
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
"""
"""
# 异常值处理
# 观察结果’OverallQual’,‘GrLivArea’,‘TotalBsmtSF’,'YearBuilt’四个特征中存在异常值
# 将异常值去除
train_data.drop(train_data[(train_data.GrLivArea>4000)&(train_data.SalePrice<200000)].index,inplace=True)
train_data.drop(train_data[(train_data.OverallQual<5)&(train_data.SalePrice>200000)].index,inplace=True)
train_data.drop(train_data[(train_data.YearBuilt<1900)&(train_data.SalePrice>400000)].index,inplace=True)
train_data.drop(train_data[(train_data.YearBuilt>1980)&(train_data.SalePrice>700000)].index,inplace=True)
train_data.drop(train_data[(train_data.TotalBsmtSF>6000)&(train_data.SalePrice<200000)].index,inplace=True)
# 同时需要重置索引值，使得索引值连续
train_data.reset_index(drop=True, inplace=True)
# 查看训练集大小
print(train_data.shape)

# 目标值SalePrice处理
# plt.figure(1)  # 正态分布拟合
# (mu, sigma) = norm.fit(train_data.SalePrice)
# fig = plt.figure(figsize=(8, 8))
# sns.distplot(train_data.SalePrice, fit=norm)
# plt.ylabel('Frequency')
# plt.legend(['u={:.2f} and sigma={:.2f}'.format(mu, sigma)], loc='best')
#  plt.figure(2)  # 化为标准正态分布
train_data.SalePrice = np.log1p(train_data.SalePrice)  # 用log转换，压缩尺寸
(mu, sigma) = norm.fit(train_data.SalePrice)
fig = plt.figure(figsize=(8, 8))
sns.distplot(train_data.SalePrice, fit=norm)
plt.ylabel('Frequency')
plt.legend(['u={:.2f} and sigma={:.2f}'.format(mu, sigma)], loc='best')
# plt.show()

# 数据清洗
all_data = pd.concat([train_data, test_data], axis=0, ignore_index=True, sort=False)
# print(all_data.shape)
# all_data.info()
# all_data.describe()
# 查看缺失值
miss_data = all_data.isnull().sum().sort_values(ascending =False)
ratio = (miss_data/len(all_data)).sort_values(ascending=False)
missing_df = pd.concat([miss_data,ratio],axis=1,keys=['miss_data','ratio'],sort=False)
missing_df[missing_df>0].count()
# print(missing_df[:35])

# 根据data_description的描述进行数据填充
# 属性为空
all_data.PoolQC=all_data.PoolQC.fillna('None')
all_data.MiscFeature=all_data.MiscFeature.fillna('None')
all_data.Alley=all_data.Alley.fillna('None')
all_data.Fence=all_data.Fence.fillna('None')
all_data.FireplaceQu=all_data.FireplaceQu.fillna('None')
all_data.GarageFinish=all_data.GarageFinish.fillna('None')
all_data.GarageCond=all_data.GarageCond.fillna('None')
all_data.GarageQual=all_data.GarageQual.fillna('None')
all_data.GarageType=all_data.GarageType.fillna('None')
all_data.BsmtExposure=all_data.BsmtExposure.fillna('None')
all_data.BsmtCond=all_data.BsmtCond.fillna('None')
all_data.BsmtQual=all_data.BsmtQual.fillna('None')
all_data.BsmtFinType1=all_data.BsmtFinType1.fillna('None')
all_data.BsmtFinType2=all_data.BsmtFinType2.fillna('None')
all_data.MasVnrType =all_data.MasVnrType.fillna('None')
# 数值型的缺失
all_data.GarageCars=all_data.GarageCars.fillna(0)
all_data.GarageYrBlt=all_data.GarageYrBlt.fillna(0)
all_data.GarageArea=all_data.GarageArea.fillna(0)
all_data.BsmtFullBath=all_data.BsmtFullBath.fillna(0)
all_data.BsmtHalfBath=all_data.BsmtHalfBath.fillna(0)
all_data.BsmtFinSF1=all_data.BsmtFinSF1.fillna(0)
all_data.BsmtFinSF2=all_data.BsmtFinSF2.fillna(0)
all_data.BsmtUnfSF=all_data.BsmtUnfSF.fillna(0)
all_data.TotalBsmtSF=all_data.TotalBsmtSF.fillna(0)
all_data.MasVnrArea= all_data.MasVnrArea.fillna(0)
# 取众数填充
all_data.SaleType=all_data.SaleType.fillna(all_data.SaleType.mode()[0])
all_data.Exterior1st=all_data.Exterior1st.fillna(all_data.Exterior1st.mode()[0])
all_data.Electrical=all_data.Electrical.fillna(all_data.Electrical.mode()[0])
all_data.Exterior2nd=all_data.Exterior2nd.fillna(all_data.Exterior2nd.mode()[0])
all_data.KitchenQual=all_data.KitchenQual.fillna(all_data.KitchenQual.mode()[0])
all_data.MSZoning=all_data.MSZoning.fillna(all_data.MSZoning.mode()[0])
# functional缺失，即典型
all_data.Functional=all_data.Functional.fillna('Typ')
# 用均值填充
all_data['LotFrontage']=all_data.groupby('Neighborhood')['LotFrontage'].apply(lambda x:x.fillna(x.median()))

# 查看缺失值
miss_data= all_data.isnull().sum().sort_values(ascending =False)
ratio = (miss_data/len(all_data)).sort_values(ascending=False)
missing_df = pd.concat([miss_data,ratio],axis=1,keys=['miss_data','ratio'],sort=False)
# print(missing_df[:2])

all_data.drop(['Id','SalePrice'],axis=1,inplace=True) # 为什么将price都删除

# 特征化
# 数值型转换为字符型
column1 = ['MSSubClass', 'YrSold','MoSold', 'OverallCond', "BsmtFullBath", "BsmtHalfBath", "HalfBath",
            "YearBuilt","YearRemodAdd", "GarageYrBlt"]
for coln in column1:
    all_data[coln]=all_data[coln].astype(str)
del coln,column1
# 顺序变量数值化
def custom_coding(x):
    if(x=='Ex'):
        r = 0
    elif(x=='Gd'):
        r = 1
    elif(x=='TA'):
        r = 2
    elif(x=='Fa'):
        r = 3
    elif(x=='None'):
        r = 4
    else:
        r = 5
    return r

cols1 = ['BsmtCond','BsmtQual','ExterCond','ExterQual','FireplaceQu','GarageCond','GarageQual','HeatingQC','KitchenQual','PoolQC']
for col in cols1:
    all_data[col] = all_data[col].apply(custom_coding)
del col,cols1
# 字符型特征标签编码
from sklearn.preprocessing import LabelEncoder
Labels=['YearBuilt','YearRemodAdd','GarageYrBlt',"YrSold", 'MoSold']
label_encoder =LabelEncoder()
for i in Labels:
    label_encoder.fit(all_data[i].values)
    all_data[i]=label_encoder.transform(all_data[i].values)
del i,Labels
# 构建特征
all_data['TotalArea']=all_data['TotalBsmtSF']+all_data['1stFlrSF']+all_data['2ndFlrSF']
all_data['YearofRemodel']=all_data['YrSold'].astype(int)- all_data['YearRemodAdd'].astype(int)
# 数值型特征偏度处理
all_data_nums=all_data.select_dtypes(exclude='object')
all_data_skews= all_data_nums.skew().sort_values()
skews = pd.DataFrame({"skew":all_data_skews})
part_skews=skews[abs(skews)>0.75].dropna()
# print("The numbers of skews need to transform is :{}".format(part_skews.shape))
# print(part_skews)
for i in part_skews.index:
    all_data[i]=boxcox1p(all_data[i],0)
del i
# 字符型变量one-hot编码
# print(all_data.shape)
all_data = pd.get_dummies(all_data,drop_first=True)
all_data.head()
all_data.info()

# 归一化
train_data_new=all_data[:len(train_data)]  # 数据清洗后长度？
test_data_new=all_data[len(train_data):]
X=train_data_new
y=train_data.SalePrice
rs =RobustScaler()
rs.fit(X)
X_rs =rs.transform(X)
X_rs_prd =rs.transform(test_data_new)
print(np.isnan(train_data_new).any())
"""
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

train.drop(train[(train.GrLivArea>4000)& (train.SalePrice<300000)].index,inplace=True)
full=pd.concat([train,test],ignore_index=True)
full.drop(['Id'],axis=1,inplace=True)
print(full.shape)

aa=full.isnull().sum()
print(aa[aa>0].sort_values(ascending=True))
print(full.groupby(['Neighborhood'])[['LotFrontage']].agg(['mean','median','count']))
full['LotAreaCut']=pd.qcut(full.LotArea,10)#输出LotArea值所属qcut后的类别；
print(full.groupby(['LotAreaCut'])[['LotFrontage']].agg(['mean','median','count']))

full['LotFrontage']=full.groupby(['LotAreaCut','Neighborhood'])['LotFrontage'].transform(lambda x:x.fillna(x.median()))
full['LotFrontage']=full.groupby(['LotAreaCut'])['LotFrontage'].transform(lambda x:x.fillna(x.median()))

#Then we filling in other missing values according to data_description.
cols=["MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "BsmtFinSF2", "BsmtFinSF1", "GarageArea"]
for col in cols:
    full[col].fillna(0,inplace=True)
cols1 = ["PoolQC" , "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish", "GarageYrBlt", "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1", "MasVnrType"]
for col in cols1:
    full[col].fillna("None",inplace=True)

cols2 = ["MSZoning", "BsmtFullBath", "BsmtHalfBath", "Utilities", "Functional", "Electrical", "KitchenQual", "SaleType","Exterior1st", "Exterior2nd"]
for col in cols2:
    full[col].fillna(full[col].mode()[0],inplace=True)

print(full.isnull().sum()[full.isnull().sum()>0])