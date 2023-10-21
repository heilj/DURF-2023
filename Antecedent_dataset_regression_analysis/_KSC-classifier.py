import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

'''用ksc对视频分类'''

# 打开并且读取 CSV 文件
data_frame = pd.read_csv('random_dao.csv')

result_list = []
for index, row in data_frame.iterrows():
    small_list = [row[0], row[5], row[17].lstrip('[').rstrip(']').split(',')]
    result_list.append(small_list)

# 转换成一个可以用的tensor形式的dataset
new = []
for video in result_list:
    if len(video[2]) == 10:
        new_video = [int(i) for i in video[2]]
        new.append(new_video)

# 进行K-Means算法的clustering
data_array = np.array(new)
num_clusters = 2
kmeans_model = KMeans(n_clusters=num_clusters, random_state=0)
cluster_labels = kmeans_model.fit_predict(data_array)

# 将簇标签添加到原始数据列表中的每个子列表末尾
for i, label in enumerate(cluster_labels):
    new[i].append(label)

# 打印带有类别标签的数据列表
for i in new:
    print(i)





