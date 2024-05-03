from datasets import load_dataset, load_from_disk
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

dataset = load_from_disk("top_quality_onehot")
x_scalar = StandardScaler()
y_scalar = StandardScaler()
X_train = np.array(dataset['train']['x'])
y_train = np.array(dataset['train']['y'])
# print(y_train)
# print(type(X_train))
# print(y_train.shape)
# X_mean = np.mean(X_train,axis=0)
# X_std = np.std(X_train,axis=0)
# y_mean = np.mean(y_train,axis=0)
# y_std = np.std(y_train,axis=0)
x_scalar.fit_transform(X_train)
X_mean = x_scalar.mean_
X_std = x_scalar.scale_
y_scalar.fit_transform(y_train.reshape(-1,1))
y_mean = y_scalar.mean_
y_std = y_scalar.scale_
# print(mean_per_feature.shape)
# print(std_per_feature.shape)
X_train_standardized = (X_train - X_mean) / X_std

y_train_standardized = (y_train - y_mean) / y_std
print(X_train_standardized.max())
y_recovered = y_train_standardized * y_std
y_recovered += y_mean
# print(y_recovered) 