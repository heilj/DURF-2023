# line = '0.15 GIWAqu2fuMw\n'
# line.rstrip("\n")
# print(line)
# samll = line.split()
# print(samll)
# samll[0] = float(samll[0])
# data_lst.append(samll)
# metadata = [1,2,3,4,5,6,7]
# # data = ['a',[1,2,3]]
# # print(data[:])
# # # data.append([])
# # data.append(metadata[6:7])
# # print(data[2])
# # list = ' '.join(map(str,data))
# # print(list)
# print(metadata.mean())
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# # 创建示例的训练用2D数组，每列是一个样本
# training_data = np.array([[1, 4, 7],
#                           [2, 5, 8],
#                           [3, 6, 9]])

# # 创建MinMaxScaler对象
# scaler = MinMaxScaler()

# # 对每个特征（列）进行MinMax归一化
# normalized_training_data = scaler.fit_transform(training_data)

# print(normalized_training_data)
# def custom_mrse(y_true, y_pred):
#         return np.mean(((y_pred / y_true) - 1) ** 2)
# y_pred = np.array([[1000], [20]])
# y_true = np.array([[100], [170]])
# relative_errors = (y_pred / y_true) - 1
# print(relative_errors)
# squared_relative_errors = relative_errors ** 2
# print(squared_relative_errors)
# mean_squared_relative_error = np.mean(squared_relative_errors)
# print(mean_squared_relative_error)
# list = [1,2,3,4,5]
# array = np.array(list)
# print(array)
# 假设有一个输入向量和一组中心点
input_vector = np.array([1.0, 2.0, 3.0])  # 输入向量
centroids = np.array([
    [2.0, 3.0, 4.0],  # 中心点1
    [0.0, 1.0, 2.0],  # 中心点2
    [3.0, 4.0, 5.0]   # 中心点3
])

# 确保输入向量是一维数组
if input_vector.ndim > 1:
    input_vector = input_vector.flatten()

# 计算输入与每个中心点之间的RBF距离
distances = np.linalg.norm(centroids - input_vector, axis=1)

print("RBF距离列表:", distances)
