import pandas as pd
import csv
import os
import numpy as np
import pickle
from xgboost import *
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import *
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from utils import *
CLASS1 = 'viral'
CLASS2 = 'quality'
CLASS3 = 'memoryless'
CLASS4 = 'junk'
source1 = 'top'
source2 = 'random'


xgb_param = {'colsample_bytree': 1,
                 'gamma': 0, 
                 'learning_rate': 0.03, 
                 'max_depth': 8, 
                 'min_child_weight': 7, 
                 'n_estimators': 117, 
                 'reg_alpha': 0.7, 
                 'reg_lambda': 0, 
                 'subsample': 0.6}

            # {
            # 'max_depth': 8,
            # 'n_estimators': 100,
            # 'gamma': 0,
            # 'subsample': 0.5,
            # 'colsample_bytree': 1,
            # 'colsample_bylevel':1, 
            # 'colsample_bynode':1,
            # 'reg_alpha': 1,
            # 'reg_lambda': 1,
            # 'learning_rate': 0.03,
            # 'objective': 'reg:squarederror',  # 回归任务的损失函数
            # 'eval_metric': 'rmse',            # 评价指标为均方根误差
            # 'seed': 42,
            # 'max_delta_step': 15,
            # 'min_child_weight': 0.8,
            # 'grow_policy' : 'lossguide'
            # }


temp_best = {
    'max_depth': 12,
    'n_estimators': 115,
    'gamma': 0,
    'subsample': 0.6,
    'colsample_bytree': 1,
    'colsample_bylevel':1, 
    'colsample_bynode':1,
    'reg_alpha': 0,
    'reg_lambda': 0.5,
    'learning_rate': 0.03,
    'objective': 'reg:squarederror',  # 回归任务的损失函数
    'eval_metric': 'rmse',            # 评价指标为均方根误差
    'seed': 42,
    'max_delta_step': 15,
    'min_child_weight': 7,
    'grow_policy' : 'lossguide'
    }


param_dist = {
    'max_depth': [5, 6, 7, 9, 10, 12, 13, 15, 17],            # 保持原始值
    'n_estimators': [109],       # 保持原始值
    'learning_rate': [0.03],     # 保持原始值
    'gamma': [0, 0.01, 0.05, 0.1, 0.3, 0.5, 1],                # 保持原始值
    'subsample': [0.6, 0.7, 0.8, 0.9, 1],          # 保持原始值
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1],     # 保持原始值
    'colsample_bylevel': [1],    # 保持原始值
    'colsample_bynode': [1],     # 保持原始值
    'reg_alpha': [0],            # 保持原始值
    'reg_lambda': [0.5],         # 保持原始值
    'objective': ['reg:squarederror'],  # 保持原始值
    'eval_metric': ['rmse'],             # 保持原始值
    'seed': [42],                # 保持原始值
    'max_delta_step': [15],      # 保持原始值
    'min_child_weight': [1, 3, 5, 7],   # 保持原始值
    'grow_policy': ['lossguide'] # 保持原始值
}

temp = {
    'n_estimators': [100],
    'learning_rate': [0.05],
    'max_depth':  [4],  
    'min_child_weight': [9],  
    'gamma': [0.3],   
    'subsample': [0.9,1,3,5,7], 
    'colsample_bytree': [1],   
    'reg_alpha': [0],           
    'reg_lambda': [0],        
    
}






def main(CLASS,source,regressor='XGB', classfied=True, include_rbf= None,threshold=0.3,centroids=10,sigma=50,tr=7,tt=29,thres=10,len=20):
    current_dir = get_parrent()
    

    rbf_data = read_token_file(source, threshold, CLASS, len)
    file2lst = read_file(f'{current_dir}/{source}_dao.csv',rbf_data,include_rbf ,tr, tt, thres)
    if classfied:
        regression_lst = classfy(CLASS, file2lst, threshold, source)
        # GridS_XGB(regression_lst,xgb_param,temp)
        (mse, y_test), (mse_train, y_train),x_test = baseline_XGB(regression_lst, param= xgb_param)
        residual_plot(mse, y_test,x_test)

        remaining_indices = np.argsort(mse)[:-20]

        # Extract corresponding y-values and residuals
        remaining_y = y_test[remaining_indices]
        remaining_mse = mse[remaining_indices]
        print(np.mean(remaining_mse ** 2))

    else:
        GridS_XGB(file2lst,xgb_param,temp)
 

main(CLASS1,source1,'XGB',classfied=True, include_rbf=None, threshold=0.3,thres=500)


