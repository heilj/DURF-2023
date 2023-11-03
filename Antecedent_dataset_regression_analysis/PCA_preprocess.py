import pickle
import numpy as np
from sklearn.decomposition import PCA

path = 'top_0.3_quality_tokens.pkl'

'''对要进行pca对token预处理，整理为list，减小处理压力'''

# 步骤 1: 加载 pkl 文件
with open(path, 'rb') as f:
    data_dict = pickle.load(f)
f.close()
t = True
data = []
for token in data_dict.values():
    if t == True:
        data = token
        t = False
    else:
        data = np.append(data,values=token,axis=0)
np.save('list_top_0.3_qualtiy_tokens',data)
