import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LinearRegression
import pickle
FIELD = 'title_top_0.3_viral_centroids='
input_file = 'title_top_0.3_viral_tokens.pkl'
sigma = 50

'''rbf计算title相似度'''

def main(centroids_num,input_file, sigma):
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
        features = rbf(centroids, token, sigma)
        output[id] = np.array(features[0])
    with open(f'rbf_{FIELD}{centroids_num}_sigma={sigma}.pkl','wb') as file:
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

def rbf(centroids, token ,sigma):
    # 准备完整数据集（示例）

    X = token  # 输入数据，可以有多个样本


    # 指定 sigma（通过 gamma 控制）
    sigma = sigma
    gamma = 1.0 / (2.0 * sigma ** 2)

    # 计算 RBF 核矩阵
    X_rbf = rbf_kernel(X, centroids, gamma=gamma)

    # 准备特征矩阵，将 RBF 核矩阵作为特征
    X_features = X_rbf
    return X_features

main(100,input_file,sigma)
