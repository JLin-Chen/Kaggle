from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.preprocessing import RobustScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.base import clone, BaseEstimator

# 建模
# 定义评价标准
def rmsle(model,X,y):
    return np.sqrt(-cross_val_score(model,X_rs,y,scoring='neg_mean_squared_error',cv=kfold))
# Lasso模型
lasso=Lasso(random_state=0)
las_param={'alpha':[0.0001,0.0002,0.0003,0.0004,0.0006,0.0007],
          'max_iter':[10000]}
las_gcv=GridSearchCV(lasso,las_param,cv=kfold,scoring='neg_mean_squared_error',verbose=1,n_jobs=3)
las_gcv.fit(X_rs,y)
best_lasso=las_gcv.best_estimator_
print('lasso_score:%.4f'%(np.sqrt(-las_gcv.best_score_)))  # 网格搜索alpha最佳参数

# Ridge回归
ridge = Ridge(random_state=0)
rid_param ={'alpha':np.arange(1,100,2),
           'max_iter':[10000]}
rid_grid=GridSearchCV(ridge,rid_param,scoring='neg_mean_squared_error',cv=kfold,verbose=1,n_jobs=3)
rid_grid.fit(X_rs,y)
best_ridge =rid_grid.best_estimator_
print('lasso_score:%.4f'%np.sqrt(-rid_grid.best_score_))

# ElesticNet
ela_net = ElasticNet(random_state=0)
ela_param={'alpha':[0.002,0.003,0.004,0.006,0.007,0.008],
       'l1_ratio':[0.01,0.02,0.03,0.04,0.05],
       'max_iter':[10000]}
ela_grid=GridSearchCV(ela_net,ela_param,cv=kfold,scoring='neg_mean_squared_error',n_jobs=3,verbose=1)
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
svr_grid=GridSearchCV(svr,svr_param,cv=kfold,n_jobs=3,verbose=1,scoring='neg_mean_squared_error')
svr_grid.fit(X_rs,y)
best_svr=svr_grid.best_estimator_
print('svr_score:%.4f'%np.sqrt(-svr_grid.best_score_))

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

result2 = pd.DataFrame({'Id':test_data['Id'],'SalePrice':y_predict2})
result2.to_csv('E:/house price/result2.csv',index=False)