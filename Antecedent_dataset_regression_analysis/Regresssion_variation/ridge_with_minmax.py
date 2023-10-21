import pandas as pd
import csv
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def main():
    classfied = True
    # classfied = False
    file2lst = read_file('/Users/gqs/Downloads/Antecedent_dataset_regression_analysis/top_dao.csv')
    if classfied:
        regression_lst = classfy("quality", file2lst, 0.3)
        MLR(regression_lst)
    else:
        MLR(file2lst)

def convert_to_list(s, float):
    
    '''
    将str类的数据形式转换为list，用于输入的格式化，这里每个元素转化为float type
    Converts a string in form format "[data0, data1, data2, ... ]" to a real array.

    Arguments:
    s -- the string

    Keywork arguments:
    dtype=str -- A basic python type (str, int flaot) to convert each value of the array to.
    '''
    data_list = [float(d.strip()) for d in s[1:-1].split(',')] if len(s) > 2 else []
    return data_list
    
def read_file(file_pass):
    result = []
    result_list = []
    with open(file_pass) as f:
        '''读取dao文件'''
        dictr = csv.DictReader(f)
        for data in dictr:
            key = data['#ID']
            data_list = convert_to_list(data['VIEW_DATA'], float)
            small_list = [key,data_list]
            result_list.append(small_list)
            # filepath = '/Users/gqs/Documents/DURF/youtube_data/test_small.txt'
            # with open(filepath,'w') as file:
            #     list = ','.join(map(str,small_list))
            #     file.write(list)
    #去除无效数据（样本小于30天），并整理前10天的数据
    for i in range(len(result_list)):
        if len(result_list[i][1]) > 30 and result_list[i][1][29] != 0:
            # print(result_list[i][1])
            k = result_list[i][1][29]
            
            # with open('top_thres0.3_viral_test.txt', 'w') as file:
            
                    
            #     file.write(result_list[i][1] + '\n')
            # print(result_list[i][0])
            # print(k)
            # if k != 0:
            small_lst = result_list[i]
            small_lst[1] = small_lst[1][0:7]
            small_lst.append([k])
            # print(small_lst[2])
            result.append(small_lst)

    return result

# def read_file(file_pass):
#     """
#     打开一个csv文件，并整理出一个[id, total_views, [d1_views, d2_views, d3_views, ..... , d10_views]]
#     """
#     data_frame = pd.read_csv(file_pass)

#     result_list = []
#     for index, row in data_frame.iterrows():
#         small_list = [row[0], row[5], row[10].lstrip('[').rstrip(']').split(',')]
#         result_list.append(small_list)
    
#     #去除无效数据，并整理成10天的格式
#     result = []
#     for i in range(len(result_list)):
#         if len(result_list[i][2]) == 10:
#             small_lst = result_list[i][:]
#             small_lst[2] = small_lst[2][:]
#             result.append(small_lst)
#     return result

def classfy(CLASS, videos, threshold):
    """
    把dataset中的data根据一个threshold归到一个类中
    """
    if CLASS == "viral":
        path = '/Users/gqs/Downloads/Antecedent_dataset_regression_analysis/classfied_top/viral_classes.txt'
    elif CLASS == "quality":
        path = "/Users/gqs/Downloads/Antecedent_dataset_regression_analysis/classfied_top/quality_classes.txt"
    elif CLASS == "junk":
        path = "/Users/gqs/Downloads/Antecedent_dataset_regression_analysis/classfied_top/junk_classes.txt"
    elif CLASS == "memoryless":
        path = "/Users/gqs/Downloads/Antecedent_dataset_regression_analysis/classfied_top/memoryless_classes.txt"
    
    #从txt文件中读取所有该列别的视频id
    file = open(path,"r")
    temp = file.readlines()
    
    if CLASS != "memoryless":
        data_lst = []
        for line in temp:
            line.rstrip("\n")
            samll = line.split()
            samll[0] = float(samll[0])
            data_lst.append(samll)
        
        all_video = set()
        for data in data_lst:
            if data[0] == threshold:
                all_video.add(data[1])
    else:
        data_lst = []
        for line in temp:
            line.rstrip("\n")
            line.rstrip()
            samll = line.rstrip("\n")
            data_lst.append(samll)

        all_video = set()
        for data in data_lst:
            
            all_video.add(data)


    all_video = list(all_video)
    # print(all_video)

    #把所有该类别视频的id和相关数据存放进一个list
    result = []
    count = 0 
    with open('top_thres0.3_viral.txt', 'w') as file:
        for video in videos:
            if video[0] in all_video:
                result.append(video)
                count += 1
                # file.write(video[0] + '\n')
    print('number_of_videos :')
    print(count)
    return result
    


def MLR(regression_lst):
    #Multiple Linear Ridge Regression
    data = regression_lst[:]
    # print(data)
    independent_variables = np.array([item[1] for item in data])
    dependent_values = np.array([item[2][0] for item in data])
    print(independent_variables.shape)
    print(dependent_values.shape)
    # independent_variables.reshape(-1,1)
    # print(independent_variables)
    y_max = max([item[2][0] for item in data])
    y_min = min([item[2][0] for item in data])
    sum = 0 
    for item in data:
        sum += item[2][0]
    y_mean = sum / len(data)
    print(y_max)
    print(y_min)
    print(f'y mean: {y_mean}')
    X_train, X_test, y_train, y_test = train_test_split(independent_variables, dependent_values, test_size=0.5, random_state=42)
    scalar = MinMaxScaler()
    X_train_scaled = scalar.fit_transform(X_train)
    # print(X_train_scaled)
    X_test_scaled = scalar.transform(X_test)
    # y_train = y_train.reshape(-1,1)
    # y_test = y_test.reshape(-1,1)
    # y_train_scaled = scalar.fit_transform(y_train)
    # y_test_scaled = scalar.transform(y_test)
    
    print('-------------------------------')
    # print(X_train)
    print('-------------------------------')
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_scaled, y_train)
    #在test数据集上测试模型
    y_pred = ridge_model.predict(X_test_scaled)
    # y_pred = scalar.inverse_transform(y_pred)
    y_pred_train = ridge_model.predict(X_train_scaled)
    # y_pred_train = scalar.inverse_transform(y_pred_train)
    y_pred_mean = 0
    for i in range(len(y_pred)):
        s = y_pred[i]
        y_pred_mean += s
    y_pred_mean = y_pred_mean / len(y_pred)
    y_test_mean = 0
    for i in range(len(y_test)):
        s = y_test[i]
        y_test_mean += s
    y_test_mean = y_test_mean / len(y_test)

    print(f'mean of predict: {y_pred_mean}')
    print(f'mean of test: {y_test_mean}')
    mse = mean_squared_error(y_test, y_pred)
    mrse = 0 
    # for i in y_test:
    #     print(i)
    for i in range(len(y_test)):
        # print(y_pred[i])
        # print(y_test[i])
        if y_test[i] != 0:
            mrse += ((float(y_pred[i])/float(y_test[i])) - 1)**2
    mrse = mrse/len(y_test)

    mrse_train = 0
    for i in range(len(y_train)):
        
        if y_train[i] != 0 :
            print(y_pred_train[i])
            print(y_train[i])
            mrse_train += ((float(y_pred_train[i])/float(y_train[i])) - 1)**2
    mrse_train = mrse_train/len(y_train)
    

    normalized_mrse1 = mrse / (y_max - y_min)
    normalized_mrse2 = mrse / y_mean

    mae = mean_squared_error(y_test,y_pred)
    print(f"均方误差（Mean Squared Error）：{mse}")
    print(f"均方根误差（Mean Root Squared Error）：{mrse}")
    print(f"train 均方根误差（Mean Root Squared Error -- train）：{mrse_train}")
    print(f"平均绝对误差（Mean Absolute Error）：{mae}")
    print(f'标准化mrse1:{normalized_mrse1}')
    print(f'标准化mrse2:{normalized_mrse2}')


    # 打印学习参数
    coefficients = ridge_model.coef_
    for feature_index, coefficient in enumerate(coefficients):
        print(f"特征 {feature_index+1}: {coefficient}")

    # 在测试集上计算决定系数
    r2_score = ridge_model.score(X_test_scaled, y_test)
    print(f"决定系数（R²）：{r2_score}")

    y_pred = ridge_model.predict(X_test_scaled)
    # y_pred = scalar.inverse_transform(y_pred)
    # 绘制散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('True Values vs. Predicted Values (Log Scale)')
    plt.xlabel('True Values (Log Scale)')
    plt.ylabel('Predicted Values (Log Scale)')
    plt.show()

    # 绘制残差图
    plt.figure(figsize=(10, 6))
    residuals = y_test - y_pred
    plt.scatter(y_test, residuals, color='red', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    plt.title('Residuals vs. True Values')
    plt.xlabel('True Values')
    plt.ylabel('Residuals')
    plt.show()

main()