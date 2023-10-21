import pickle
import numpy as np
from sklearn.decomposition import PCA

path = 'top_0.3_viral_tokens.pkl'
path2 = 'top_0.3_viral_tokens_len=20.p.npy'
pathout = 'dict_top_0.3_viral_tokens_len=20.pkl'

'''将pca的结果与视频id配对'''

# 步骤 1: 加载 pkl 文件
with open(path, 'rb') as f:
    data_dict = pickle.load(f)
f.close()

token_list = np.load(path2)

result_dict = {}

for i, key in enumerate(data_dict.keys()):
    result_dict[key] = token_list[i]
del data_dict
del token_list
with open(pathout, "wb") as file:
    pickle.dump(result_dict, file)
