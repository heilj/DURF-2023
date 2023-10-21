import pickle
import random
import numpy as np
from sklearn.cluster import KMeans
path = ['top_0.3_viral_tokens_Alan.pkl', 'top_0.3_viral_tokens_Alex.pkl', 'top_0.3_viral_tokens_Ken.pkl', 'top_0.3_viral_tokens_Qiaosong.pkl']
'''
为rbf选择centroids
自定义k数量
'''
def select_centroids(k,path):
    with open(path[0],'rb') as file:
        dict = pickle.load(file)
        values1 = list(dict.values())
        del dict
    file.close()
    with open(path[1],'rb') as file:
        dict = pickle.load(file)
        values2 = list(dict.values())
        del dict
    file.close()
    values1 += values2
    del values2
    print('done1')

    with open(path[2],'rb') as file:
        dict = pickle.load(file)
        values3 = list(dict.values())
        del dict
    file.close()
    values1 += values3
    del values3
    print('done2')
    with open(path[3],'rb') as file:
        dict = pickle.load(file)
        values4 = list(dict.values())
        del dict
    file.close()
    values1 += values4
    del values4
    print('done3')
    data = np.array(values1)
    print('done4')
    del values1

    
    


    data = data.reshape(-1, data.shape[-1])
    print('done5')
    kmeans = KMeans(n_clusters=k)
    # print(data)
    kmeans.fit(data)
    del data
    print('done6')
    centroids = kmeans.cluster_centers_
    # print(centroids)

    centroids_dict = {}
    for i in range(len(centroids)):
        centroids_dict[i] = centroids[i]

        
    with open(f'top_0.3_viral_centroids={k}.pkl','wb') as f:
        pickle.dump(centroids_dict, f)
    f.close()

select_centroids(200,path)