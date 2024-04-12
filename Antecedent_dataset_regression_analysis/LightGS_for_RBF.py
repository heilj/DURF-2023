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

quality_param_rbf = {'colsample_bytree': 0.5,
                 'gamma': 0, 
                 'learning_rate': 0.03, 
                 'max_depth': 8, 
                 'min_child_weight': 6, 
                 'n_estimators': 130, 
                 'reg_alpha': 0.1, 
                 'reg_lambda': 0, 
                 'subsample': 0.7}

viral_param_rbf = {'colsample_bytree': 0.5,
                 'gamma': 0, 
                 'learning_rate': 0.03, 
                 'max_depth': 7, 
                 'min_child_weight': 6, 
                 'n_estimators': 135, 
                 'reg_alpha': 0.1, 
                 'reg_lambda': 0, 
                 'subsample': 0.7}

temp_best_rbf = {
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


param_dist_rbf = {
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

temp_rbf = {
    'n_estimators': [140,150,170,200,250],
    'learning_rate': [0.01,0.03,0.05,0.1],
    'max_depth':  [3,4,5,6],  
    'min_child_weight': [1,3,5,6],  
    'gamma': [0],   
    'subsample': [0.7], 
    'colsample_bytree': [1],   
    'reg_alpha': [0],           
    'reg_lambda': [0],            
}

def main(CLASS,source,regressor='XGB', classfied=True, include_rbf= True,threshold=0.3,centroids=10,sigma=50,tr=7,tt=29,thres=10,len=20):
    current_dir = get_parrent()
    
    if include_rbf == True:
        rbf_data = read_token_file(source, threshold, CLASS, len)
        file2lst = read_file(f'{current_dir}/{source}_dao.csv',rbf_data,True ,tr, tt, thres)
        regression_lst = classfy(CLASS, file2lst, threshold, source)
        X_train_scaled, y_train, params = GridS_XGB(regression_lst,viral_param_rbf,temp_rbf)
        # find_best_n_estimators(X_train_scaled, y_train, params)
 
main(CLASS1,source1,'XGB',classfied=True, include_rbf=True, threshold=0.3,thres=150)


