import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist 
import pickle
FIELD = 'top_0.3_viral_centroids='
input_file = 'top_0.3_viral_tokens.pkl'
k = 3

'''使用knn计算距离作为特征时用找centroids'''

def main(centroids_num,input_file, k):
    output = {}
    try:
        centroids = prepare_centroids(f'{FIELD}{centroids_num}.pkl')
    except FileNotFoundError:
        print('invalid centroids num')
    try:
        data_dict = prepare_data(input_file)
    except FileNotFoundError:
        print('invalid file address')
    for id, token in data_dict.items():
        features = kmeans(centroids, token, k)
        output[id] = np.array(features)
    with open(f'knn_{FIELD}{centroids_num}_k={k}.pkl','wb') as file:
        pickle.dump(output,file)
    file.close()
    


def prepare_centroids(path):
    with open(path,'rb') as f:
        dict = pickle.load(f)
        centroids = np.array(list(dict.values()))
    f.close()
    return centroids

def prepare_data(input_file):
    with open(input_file,'rb') as file:
        dict = pickle.load(file)
    file.close()
    return dict

def kmeans(centroids, token ,k):
    # 准备完整数据集（示例）

    X = token  # 输入数据，可以有多个样本
    # 计算 RBF 核矩阵
    # km = KMeans(n_clusters=k)

    # # 准备特征矩阵，将 RBF 核矩阵作为特征
    # KMeans.fit(centroids)
    # # 找到每个样本到最近的K个质心的距离
    # closest_centroids, distances = pairwise_distances_argmin_min(X, km.cluster_centers_)
    # knn = NearestNeighbors(n_neighbors=k)
    # knn.fit(X)
    # 找到样本最近的K个质心的距离和索引
    # distances, indices = knn.kneighbors(token.reshape(1, -1))
    # X_features = np.mean(distances)

    # 示例数据，centroids为质心，sample为待预测的样本


    # 计算样本与质心之间的距离
    distances = cdist(token.reshape(1, -1), centroids)

    # 找到距离最近的K个质心的索引
    closest_centroid_indices = np.argsort(distances)[0][:k]

    # 打印距离最近的K个质心的索引和距离
    X_features = 0
    for centroid_idx in closest_centroid_indices:
        distance = distances[0][centroid_idx]
        X_features += distance
    X_features /= k

    return X_features

main(200,input_file,k)
