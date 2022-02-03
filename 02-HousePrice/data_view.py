import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 查看数据结构
train_data=pd.read_csv("train.csv")
train_data.info()
train_data.describe()

test_data=pd.read_csv("test.csv")
test_data.info()
test_data.describe()

# 查看数据间的相关关系
fig = plt.figure(figsize=(8,8))
corr=train_data.corr()
ax=sns.heatmap(corr,vmax=1,square=True)
bottom,top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)

# 取前10个最相关的因素
corrprice= corr.nlargest(10,'SalePrice').index
new_data=train_data[corrprice].corr()
fig=plt.figure(figsize=(8,8))
ax=sns.heatmap(new_data,annot=True,square=True,fmt='.2f')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)

# 查看各组数据特征
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train_data[cols], height = 2.5)
plt.show()
