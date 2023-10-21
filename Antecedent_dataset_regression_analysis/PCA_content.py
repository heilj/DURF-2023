import pickle
import numpy as np
from sklearn.decomposition import PCA

path = 'list_top_0.3_viral_tokens.npy'

'''pca压缩高维的token到低维
指定目标维度n'''

# 步骤 1: 加载 pkl 文件
all_arrays = np.load(path)


# 步骤 3: 应用 PCA
n_components = 20  # 你可以根据需要调整主成分的数量
pca = PCA(n_components=n_components)
transformed_data = pca.fit_transform(all_arrays)
del all_arrays
# 步骤 4: 将结果与 ID 关联
# result_dict = {}
# for i, key in enumerate(data_dict.keys()):
#     result_dict[key] = transformed_data[i]
# del data_dict
pathout = f'top_0.3_viral_tokens_len={n_components}.p'
np.save(pathout,transformed_data)

# result_dict 就是包含每个 ID 对应 PCA 结果的新字典
