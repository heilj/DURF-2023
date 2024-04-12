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
from utils import *
CLASS1 = 'viral'
CLASS2 = 'quality'
CLASS3 = 'memoryless'
CLASS4 = 'junk'
source1 = 'top'
source2 = 'random'


SGD_viral_param = {
    'loss': 'huber',
    'penalty': 'l2',
    'l1_ratio': 0.15,
    'fit_intercept' : True,
    'alpha': 0.01,
    'learning_rate': 'adaptive',
    'eta0': 0.8,
    'max_iter': 50000,
    'random_state': 42,
    'early_stopping' : True,
    'power_t':0.1,
    'validation_fraction':0.08,
    'epsilon' : 0.15,
    'average': False
    }

ridge_alpha = 0.1   




svr_params = {
    'kernel': 'rbf',        # 核函数类型，例如 'linear', 'poly', 'rbf', 等等
    'C': 1.0,               # 正则化参数
    'epsilon': 0.1,         # SVR的epsilon参数
    'gamma': 'scale'        # gamma参数，可以是 'scale' 或 'auto'，或一个具体的数值
}

'''主回归模型
regressor： 使用的回归方法
classified：是否对视频分类
include_rbf： 是否添加其他特征
    ***将include_rbf设置为None将会使用与包含特征时同样的视频便于比较 
       设置为False 则不考虑视频数量之间的差异
threshold： 视频分类依据的threshold
centroid： rbf使用的centroids数量（非rbf特征不考虑）
sigma： rbf使用的sigma大小（非rbf特征不考虑）
tr： 播放量特征截止的天数（第一天值为0）
tt： y值对应的播放量的天数-1
thres： 取用数据的特征天数最后一天的播放量下限，避免无效数据干扰
len： 使用token作为特征时的token长度'''

'''根据要用的特征从 read_token_file(直接输入token) read_knn_file(使用knn获取相似度) read_rbf_file(rbf获取相似度) 中选一个来获取rbf_data
'''
'''根据使用的特征，calss，regressor的不同替换params
'''

def main(CLASS,source,regressor='ridge', classfied=True, include_rbf= True,threshold=0.3,centroids=10,sigma=50,tr=7,tt=29,thres=1000,len=20):
    current_dir = get_parrent()
    if include_rbf == True:
        rbf_data = read_token_file(source, threshold, CLASS, len)
        file2lst = read_file(f'{current_dir}/{source}_dao.csv',rbf_data,True ,tr, tt, thres)

    else:
        file2lst = read_file(f'{current_dir}/{source}_dao.csv',None,include_rbf ,tr, tt, thres)
    if regressor == 'SGD':
        if classfied:
            regression_lst = classfy(CLASS, file2lst, threshold, source)
            SGD_MLR(regression_lst,SGD_viral_param)
        else:
            SGD_MLR(file2lst,SGD_viral_param)
    elif regressor == 'ridge':
        if classfied:
            regression_lst = classfy(CLASS, file2lst, threshold, source)
            Ridge_MLR(regression_lst,ridge_alpha)
        else:
            Ridge_MLR(file2lst,ridge_alpha)
    elif regressor == 'SVR':
        if classfied:
            regression_lst = classfy(CLASS, file2lst, threshold, source)
            SVR_MLR(regression_lst,svr_params)
        else:
            SVR_MLR(file2lst,svr_params)


main(CLASS1,source1,'ridge',False, False,0.3,thres=500)


