import pickle
import random
import numpy as np
from sklearn.cluster import KMeans

'''使用title作为特征时获取centroids'''

path = 'title_top_0.3_viral_tokens.pkl'
def select_centroids(k,path):
    with open(path,'rb') as file:
        dict = pickle.load(file)
        values = list(dict.values())

    file.close()
    
    data = np.array(values)


    
    


    data = data.reshape(-1, data.shape[-1])

    kmeans = KMeans(n_clusters=k)
    # print(data)
    kmeans.fit(data)


    centroids = kmeans.cluster_centers_
    # print(centroids)

    centroids_dict = {}
    for i in range(len(centroids)):
        centroids_dict[i] = centroids[i]

        
    with open(f'title_top_0.3_viral_centroids={k}.pkl','wb') as f:
        pickle.dump(centroids_dict, f)
    f.close()

select_centroids(100,path)