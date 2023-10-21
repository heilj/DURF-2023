import pandas as pd
import random
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold
from numpy import arange
'''
无效方法
'''

# 打开并且读取 CSV 文件
data_frame = pd.read_csv('random_dao.csv')

result_list = []
for index, row in data_frame.iterrows():
    small_list = [row[0], row[5], row[17].lstrip('[').rstrip(']').split(',')]
    result_list.append(small_list)

#去除无效数据，并整理成7天格式
result = []
for i in range(len(result_list)):
    if len(result_list[i][2]) >= 7:
        small_lst = result_list[i][:]
        small_lst[2] = small_lst[2][:7]
        result.append(small_lst)
print(result)
#Multiple Linear Ridge Regression
X = []
y = []
for video in result:
    y.append(video[1])
    X.append([int(i) for i in video[2]])
cv = RepeatedKFold(n_splits=4, n_repeats=3, random_state=1)
model = RidgeCV(alphas=arange(0, 1, 0.01), cv=cv, scoring='neg_mean_absolute_error')
model.fit(X, y)
print(model.best_score_)