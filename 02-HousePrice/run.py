import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from scipy.special import boxcox1p,inv_boxcox1p
from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from sklearn.base import clone, BaseEstimator, RegressorMixin
from sklearn.preprocessing import LabelEncoder

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

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
# print(train_data.shape)

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
column1 = ['MSSubClass', 'YrSold','MoSold', 'OverallCond',
           "BsmtFullBath", "BsmtHalfBath", "HalfBath",
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
del col, cols1
# 字符型特征标签编码
Labels = ['YearBuilt','YearRemodAdd','GarageYrBlt',"YrSold", 'MoSold']
label_encoder = LabelEncoder()
for i in Labels:
    label_encoder.fit(all_data[i].values)
    all_data[i] = label_encoder.transform(all_data[i].values)
del i, Labels
# 构建特征
all_data['TotalArea'] = all_data['TotalBsmtSF']+all_data['1stFlrSF']+all_data['2ndFlrSF']
all_data['YearofRemodel'] = all_data['YrSold'].astype(int) - all_data['YearRemodAdd'].astype(int)
# 数值型特征偏度处理
all_data_nums = all_data.select_dtypes(exclude='object')
all_data_skews = all_data_nums.skew().sort_values()
skews = pd.DataFrame({"skew": all_data_skews})
part_skews = skews[abs(skews) > 0.75].dropna()
print("The numbers of skews need to transform is :{}".format(part_skews.shape))
print(part_skews)
for i in part_skews.index:
    all_data[i] = boxcox1p(all_data[i], 0)
del i
# 字符型变量one-hot编码
print(all_data.shape)
all_data = pd.get_dummies(all_data, drop_first=True)
all_data.head()
all_data.info()

# 归一化
train_data_new = all_data[:len(train_data)]  # 数据清洗后长度？确实缩短了并且报错了
test_data_new = all_data[len(train_data):]
X = train_data_new
y = train_data.SalePrice
rs = RobustScaler()
rs.fit(X)
X_rs = rs.transform(X)
X_rs_prd = rs.transform(test_data_new)


# 建模
# 定义评价标准
def rmsle(model, X, y):
    return np.sqrt(-cross_val_score(model,X_rs,y,scoring='neg_mean_squared_error'))

# Lasso模型
lasso=Lasso(random_state=0)
las_param={'alpha':[0.0001,0.0002,0.0003,0.0004,0.0006,0.0007],
          'max_iter':[10000]}
# las_gcv=GridSearchCV(lasso,las_param,cv= KFold,scoring='neg_mean_squared_error',verbose=1,n_jobs=3) #不输入cv值，自动调整为5fold
las_gcv=GridSearchCV(lasso,las_param,scoring='neg_mean_squared_error',verbose=1,n_jobs=3)
las_gcv.fit(X_rs,y)
best_lasso=las_gcv.best_estimator_
print('lasso_score:%.4f'%(np.sqrt(-las_gcv.best_score_)))  # 网格搜索alpha最佳参数

# Ridge回归
ridge = Ridge(random_state=0)
rid_param ={'alpha':np.arange(1,100,2),
           'max_iter':[10000]}
rid_grid=GridSearchCV(ridge,rid_param,scoring='neg_mean_squared_error',verbose=1,n_jobs=3)
rid_grid.fit(X_rs,y)
best_ridge =rid_grid.best_estimator_
print('lasso_score:%.4f'%np.sqrt(-rid_grid.best_score_))

# ElesticNet
ela_net = ElasticNet(random_state=0)
ela_param={'alpha':[0.002,0.003,0.004,0.006,0.007,0.008],
       'l1_ratio':[0.01,0.02,0.03,0.04,0.05],
       'max_iter':[10000]}
ela_grid=GridSearchCV(ela_net,ela_param,scoring='neg_mean_squared_error',n_jobs=3,verbose=1)
ela_grid.fit(X_rs,y)
best_ela=ela_grid.best_estimator_
print('ela_score:%.4f'%np.sqrt(-ela_grid.best_score_))

# Svr
svr=SVR()
svr_param={'gamma':[0.0004,0.0005,0.0006,0.0007],
        'kernel':['rbf'],
        'C':[12,13,14,15],
        'epsilon':[0.006,0.007,0.008,0.009,0.01,0.02]
        }
svr_grid=GridSearchCV(svr,svr_param,n_jobs=3,verbose=1,scoring='neg_mean_squared_error')
svr_grid.fit(X_rs,y)
best_svr=svr_grid.best_estimator_
print('svr_score:%.4f'%np.sqrt(-svr_grid.best_score_))


# 权重
class AverageWeight(BaseEstimator,RegressorMixin):
    def __init__(self,mod,weight):
        self.mod=mod
        self.weight=weight

    def fit(self,X,y):
        self.models_=[clone(x) for x in self.mod]
        for model in self.models_:
            model.fit(X,y)
        return self

    def predict(self,X):
        w=list()
        pred=np.array([model.predict(X) for model in self.models_])
        for data in range(pred.shape[1]):
            single=[pred[model,data]*weight for model,weight in zip(range(pred.shape[0]),self.weight)]
            w.append(np.sum(single))
        return
# 集成模型
class StackingAveragedModels(BaseEstimator):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=0)
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                instance.fit(X[train_index], y[train_index])
                self.base_models_[i].append(instance)
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
        self.meta_model.fit(out_of_fold_predictions, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack(
            [np.column_stack([model.predict(X) for model in single_model]).mean(axis=1) for single_model in
             self.base_models_])
        return self.meta_model.predict(meta_features)


weight_avg = AverageWeight(mod=[best_lasso, best_ela, best_ridge, best_svr], weight=[0.25, 0.25, 0.25, 0.25])
weight_avg.fit(X_rs, y)
print('weight_score:%.4f' % np.mean(rmsle(weight_avg, X_rs, y)))

stack = StackingAveragedModels([best_ela,best_ridge,best_svr],best_lasso)
stack.fit(X_rs,y.values)
print('stack_score:%.4f'%np.mean(rmsle(stack,X_rs,y.values)))

y_predict2=[]
# 还原数据
# for i in part_skews.index:
#     y_predict2[i] = inv_boxcox1p(y.values[i], 0)
# del i

result2 = pd.DataFrame({'Id':test_data['Id'],'SalePrice':y.values})
result2.to_csv('result.csv',index=False)
