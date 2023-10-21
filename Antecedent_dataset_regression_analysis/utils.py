import numpy as np
import csv
import os
import pickle
from xgboost import *
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import *
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

'''key utils for model'''

def get_parrent():
    # 获取当前工作目录
    current_working_directory = os.getcwd()

    # 获取文件的母文件夹（假设文件名为 'myfile.txt'）
    file_name = 'myfile.txt'
    file_path = os.path.join(current_working_directory, file_name)
    file_folder = os.path.dirname(file_path)
    return file_folder

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

def convert_to_daily(data_list):
    '''进行分类时使用的是每天的新增播放量而非总播放量
    对数据进行预处理'''
    daily_views = [data_list[0]]  # 创建一个新列表，初始值为第一天的播放量

    for i in range(1, len(data_list)):
        daily_change = data_list[i] - data_list[i - 1]  # 计算每天的播放量变化
        daily_views.append(daily_change)  # 将每天的播放量变化添加到新列表中
    return daily_views

def logt(input):
    return np.log1p(input)

def inverselogt(input):
    return np.expm1(input)

def read_rbf_file(type='top',thres=0.3,CLASS='viral',centroids=10,sigma=50):
    file_path = f'rbf_{type}_{thres}_{CLASS}_centroids={centroids}_sigma={sigma}.pkl'      
    with open(file_path,'rb') as file:
        rbf_data = pickle.load(file)
    file.close()
    return rbf_data

def read_knn_file(type='top',thres=0.3,CLASS='viral',centroids=10,k=3):
    file_path = f'knn_{type}_{thres}_{CLASS}_centroids={centroids}_k={k}.pkl'      
    with open(file_path,'rb') as file:
        knn_data = pickle.load(file)
    file.close()

    return knn_data

def read_token_file(type='top',thres=0.3,CLASS='viral',len = 20):
    file_path = f'dict_{type}_{thres}_{CLASS}_tokens_len={len}.pkl'      
    with open(file_path,'rb') as file:
        token_data = pickle.load(file)
    file.close()

    return token_data

def read_file(file_pass,rbf_data ,rbf=None, tr=7, tt=29,thres=10):
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
    f.close()
    #去除无效数据（样本小于30天），并整理前7天的数据
    
    all_id = []
    
    if rbf == True:
        print('include rbf')
        for id, rbf in rbf_data.items():
            all_id.append(id)

        for i in range(len(result_list)):
            if  result_list[i][0] in all_id:
                if len(result_list[i][1]) > 30 and result_list[i][1][tt] != 0 and result_list[i][1][1] > 0 and result_list[i][1][tt] > thres:
                    k = result_list[i][1][tt]
                    small_lst = result_list[i]
                    id = small_lst[0]
                    small_lst[1] = small_lst[1][1:tr]
                    small_lst[1] = convert_to_daily(small_lst[1])
                    small_lst.append([k])
                    small_lst.append(rbf_data[id])
                    # print(small_lst[2])
                    result.append(small_lst)
    elif rbf == None:
        for id, rbf in rbf_data.items():
            all_id.append(id)
        for i in range(len(result_list)):
            if  result_list[i][0] in all_id:
                if len(result_list[i][1]) > 30 and result_list[i][1][tt] != 0 and result_list[i][1][1] > 0 and result_list[i][1][tt] > thres:
                    k = result_list[i][1][tt]
                    small_lst = result_list[i]
                    small_lst[1] = small_lst[1][1:tr]
                    small_lst[1] = convert_to_daily(small_lst[1])
                    small_lst.append([k])
                    
                    result.append(small_lst)
    else:
         for i in range(len(result_list)):
            if len(result_list[i][1]) > 30 and result_list[i][1][tt] != 0 and result_list[i][1][1] > 0 and result_list[i][1][tt] > thres:
                k = result_list[i][1][tt]
                small_lst = result_list[i]
                small_lst[1] = small_lst[1][1:tr]
                small_lst[1] = convert_to_daily(small_lst[1])
                small_lst.append([k])    
                result.append(small_lst)     

    return result

def classfy(CLASS, videos, threshold=0.3,type='top'):
    """
    把dataset中的data根据一个threshold归到一个类中
    """
    dir = get_parrent()
    path = f'{dir}/classfied_{type}/{CLASS}_classes.txt'

    
    #从txt文件中读取所有该列别的视频id
    file = open(path,"r")
    temp = file.readlines()
    
    if CLASS != "memoryless":
        data_lst = []
        for line in temp:
            line.rstrip("\n")
            samll = line.split()
            
            data_lst.append(samll)
        
        all_video = set()
        for data in data_lst:
            if data[0] == str(threshold):
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

    for video in videos:
        if video[0] in all_video:
            result.append(video)
            count += 1
            
            

    
    print('number_of_videos :')
    print(count)
    return result

def XGB_MLR(regression_lst,params,test_size=0.3,):
    #Multiple Linear Ridge Regression
    data = regression_lst[:]
    # print(data)
    independent_variables = np.array([item[1] for item in data])
    dependent_values = np.array([item[2][0] for item in data])

    # print(independent_variables.shape)
    # print(dependent_values.shape)
    # independent_variables.reshape(-1,1)
    # print(independent_variables)
    y_max = max([item[2][0] for item in data])
    y_min = min([item[2][0] for item in data])
    sum = 0 
    for item in data:
        sum += item[2][0]
    y_mean = sum / len(data)
    # print(f'y_max: {y_max}')
    # print(f'y_min: {y_min}')
    # print(f'y mean: {y_mean}')
    scalar = RobustScaler()
    scalar2 = MinMaxScaler()
    independent_variables = logt(independent_variables)
    dependent_values = logt(dependent_values)
    independent_variables = scalar.fit_transform(independent_variables)
    if len(data[0]) == 4:
        rbf_variables = np.array([item[3] for item in data])
        # rbf_variables = rbf_variables.reshape(-1,1)
        # rbf_variables = scalar2.fit_transform(rbf_variables)
        independent_variables = np.concatenate((independent_variables,rbf_variables),axis=1)
    X_train, X_test, y_train, y_test = train_test_split(independent_variables, dependent_values, test_size=test_size, random_state=42)

    X_train_scaled = X_train
    X_test_scaled = X_test
    # X_train_scaled = scalar.transform(X_train)
    # print(X_train_scaled)
    # X_test_scaled = scalar.transform(X_test)
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    dependent_values = dependent_values.reshape(-1,1)
    # y_train_scaled = scalar.fit_transform(y_train)
    # y_test_scaled = scalar.fit_transform(y_test)
    
    print('-------------------------------')
    # print(X_train)
    print('-------------------------------')
        # 定义 XGBoost 参数
    # xgb_params = {
    # 'max_depth': 3,
    # 'gamma': 0,
    # 'subsample': 1,
    # 'colsample_bytree': 0.8,
    # 'reg_alpha': 0.5,
    # 'reg_lambda': 0,
    # 'learning_rate': 0.04,
    # 'objective': 'reg:squarederror',  # 回归任务的损失函数
    # 'eval_metric': 'rmse',            # 评价指标为均方根误差
    # 'seed': 42
    # }
    xgb_params = params
    xgb_model = XGBRegressor(**xgb_params)
    xgb_model.fit(X_train_scaled, y_train) #fit模型
    #在test数据集上测试模型
    y_pred = xgb_model.predict(X_test_scaled) #第一次predict y 使用xtest
    y_pred = inverselogt(y_pred)
    r2_score = xgb_model.score(X_test_scaled, y_test)
    print(f"决定系数（R²）：{r2_score}")
    y_test = inverselogt(y_test)
    y_train = inverselogt(y_train)
    # y_pred = scalar.inverse_transform(y_pred)
    y_pred_train = xgb_model.predict(X_train_scaled) #第二次predict y 使用xtrain
    y_pred_train = inverselogt(y_pred_train)
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
    # print(len(y_test))
    count = 0 
    count2 = 0
    for i in range(len(y_test)):
        # print(y_pred[i])
        # print(y_test[i])
        # k = ((y_pred[i][0])/(y_test[i][0])) 
        # if k > 2:
        #     # print(k)
        #     # print(y_pred[i][0])
        #     # print(y_test[i][0])
        #     count += 1
        # else:
        mrse += (((y_pred[i])/(y_test[i])) - 1)**2
        count2 += 1
    print('----------------------------------------')
    # print(count)
    print(count2)
    mrse = mrse/count2

    count = 0 
    count2 = 0
    mrse_train = 0
    for i in range(len(y_train)):
        # print(y_pred_train[i])
        # print(y_train[i])
        # k = ((y_pred_train[i][0])/(y_train[i][0])) 
        # if k > 2:
        #     count += 1
        # else:
            mrse_train += (((y_pred_train[i])/(y_train[i])) - 1)**2
            count2 += 1
    print('----------------------------------------')
    # print(count)
    print(count2)
    mrse_train = mrse_train/count2

    normalized_mrse1 = mrse / (y_max - y_min)
    normalized_mrse2 = mrse / y_mean

    mae = mean_squared_error(y_test,y_pred)
    print(f"均方误差（Mean Squared Error）：{mse}")
    print(f"均相对平方误差（Mean Relative Squared Error）：{mrse}")
    print(f"train 均相对平方误差（Mean Relative Squared Error -- train）：{mrse_train}")
    print(f"平均绝对误差（Mean Absolute Error）：{mae}")
    # print(f'标准化mrse1:{normalized_mrse1}')
    # print(f'标准化mrse2:{normalized_mrse2}')



    # 打印学习参数
    coefficients = xgb_model.feature_importances_
    # for feature_index, coefficient in enumerate(coefficients):
    #     print(f"特征 {feature_index+1}: {coefficient}")
    # 在测试集上计算决定系数
    y_test = logt(y_test)
    r2_score2 = xgb_model.score(X_test_scaled, y_test)
    # print(f"两次log转换后的决定系数（R²）：{r2_score2}")

    # Define the MRSE custom scoring function
    def custom_mrse(y_true, y_pred):
        sum = 0
        # print(y_pred / y_true)
        for i in range(len(y_true)):
            a = inverselogt(y_pred[i])
            b = inverselogt(y_true[i])
            sum += (((a / b) - 1) ** 2)

        return sum/ len(y_true)
    # Create the custom scorer
    mrse_scorer = make_scorer(custom_mrse, greater_is_better=True)

    #cross validation
    cv_r2 = cross_val_score(xgb_model, independent_variables, dependent_values, cv=4, scoring='r2')
    cv_mrse = cross_val_score(xgb_model, independent_variables, dependent_values, cv=4, scoring = mrse_scorer)
    
    # print(f"cross validation 均相对平方误差（Mean Relative Squared Error）：{cv_mrse}")
    # print(f"cross validation r2：{cv_r2}")
    print(f"cross validation 平均 均相对平方误差（Mean Relative Squared Error）：{cv_mrse.mean()}")
    print(f"cross validation 平均 r2：{cv_r2.mean()}")

    y_test = inverselogt(y_test)
    # # 绘制散点图
    # plt.figure(figsize=(10, 6))
    # plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.title('True Values vs. Predicted Values (Log Scale)')
    # plt.xlabel('True Values (Log Scale)')
    # plt.ylabel('Predicted Values (Log Scale)')
    # plt.show()

    # # 绘制残差图
    # plt.figure(figsize=(10, 6))
    # residuals = y_test - y_pred
    # plt.scatter(y_test, residuals, color='red', alpha=0.7)
    # plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    # plt.title('Residuals vs. True Values')
    # plt.xlabel('True Values')
    # plt.ylabel('Residuals')
    # plt.show()

def SGD_MLR(regression_lst,params,test_size=0.3,):
    #Multiple Linear Ridge Regression
    data = regression_lst[:]
    # print(data)
    independent_variables = np.array([item[1] for item in data])
    dependent_values = np.array([item[2][0] for item in data])

    print(independent_variables.shape)
    # print(dependent_values.shape)
    # independent_variables.reshape(-1,1)
    # print(independent_variables)
    y_max = max([item[2][0] for item in data])
    y_min = min([item[2][0] for item in data])
    sum = 0 
    for item in data:
        sum += item[2][0]
    y_mean = sum / len(data)
    print(f'y_max: {y_max}')
    print(f'y_min: {y_min}')
    print(f'y mean: {y_mean}')
    scalar = RobustScaler()
    scalar2 = MinMaxScaler()
    scalar3 = RobustScaler()
    scalar4 = MinMaxScaler()
    independent_variables = logt(independent_variables)
    dependent_values = logt(dependent_values)
    independent_variables = scalar.fit_transform(independent_variables)
    independent_variables = scalar2.fit_transform(independent_variables)
    if len(data[0]) == 4:
        rbf_variables = np.array([item[3] for item in data])
        rbf_variables = rbf_variables.reshape(-1,1)
        # rbf_variables = scalar3.fit_transform(rbf_variables)
        rbf_variables = scalar4.fit_transform(rbf_variables)
        independent_variables = np.concatenate((independent_variables,rbf_variables),axis=1)
    X_train, X_test, y_train, y_test = train_test_split(independent_variables, dependent_values, test_size=test_size, random_state=42)

    X_train_scaled = X_train
    X_test_scaled = X_test
    # X_train_scaled = scalar.transform(X_train)
    # print(X_train_scaled)
    # X_test_scaled = scalar.transform(X_test)
    # y_train = y_train.reshape(-1,1)
    # y_test = y_test.reshape(-1,1)
    # dependent_values = dependent_values.reshape(-1,1)
    # y_train_scaled = scalar.fit_transform(y_train)
    # y_test_scaled = scalar.fit_transform(y_test)
    
    print('-------------------------------')
    # print(X_train)
    print('-------------------------------')

    xgb_params = params
    xgb_model = SGDRegressor(**xgb_params)
    xgb_model.fit(X_train_scaled, y_train) #fit模型
    #在test数据集上测试模型
    y_pred = xgb_model.predict(X_test_scaled) #第一次predict y 使用xtest
    y_pred = inverselogt(y_pred)
    r2_score = xgb_model.score(X_test_scaled, y_test)
    print(f"决定系数（R²）：{r2_score}")
    y_test = inverselogt(y_test)
    y_train = inverselogt(y_train)
    # y_pred = scalar.inverse_transform(y_pred)
    y_pred_train = xgb_model.predict(X_train_scaled) #第二次predict y 使用xtrain
    y_pred_train = inverselogt(y_pred_train)
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
    # print(len(y_test))
    count = 0 
    count2 = 0
    for i in range(len(y_test)):
        # print(y_pred[i])
        # print(y_test[i])
        # k = ((y_pred[i][0])/(y_test[i][0])) 
        # if k > 2:
        #     # print(k)
        #     # print(y_pred[i][0])
        #     # print(y_test[i][0])
        #     count += 1
        # else:
        mrse += (((y_pred[i])/(y_test[i])) - 1)**2
        count2 += 1
    print('----------------------------------------')
    # print(count)
    print(count2)
    mrse = mrse/count2

    count = 0 
    count2 = 0
    mrse_train = 0
    for i in range(len(y_train)):
        # print(y_pred_train[i])
        # print(y_train[i])
        # k = ((y_pred_train[i][0])/(y_train[i][0])) 
        # if k > 2:
        #     count += 1
        # else:
            mrse_train += (((y_pred_train[i])/(y_train[i])) - 1)**2
            count2 += 1
    print('----------------------------------------')
    # print(count)
    print(count2)
    mrse_train = mrse_train/count2

    normalized_mrse1 = mrse / (y_max - y_min)
    normalized_mrse2 = mrse / y_mean

    mae = mean_squared_error(y_test,y_pred)
    print(f"均方误差（Mean Squared Error）：{mse}")
    print(f"均相对平方误差（Mean Relative Squared Error）：{mrse}")
    print(f"train 均相对平方误差（Mean Relative Squared Error -- train）：{mrse_train}")
    print(f"平均绝对误差（Mean Absolute Error）：{mae}")
    # print(f'标准化mrse1:{normalized_mrse1}')
    # print(f'标准化mrse2:{normalized_mrse2}')



    # 打印学习参数
    coefficients = xgb_model.coef_
    for feature_index, coefficient in enumerate(coefficients):
        print(f"特征 {feature_index+1}: {coefficient}")
    # 在测试集上计算决定系数
    y_test = logt(y_test)
    r2_score2 = xgb_model.score(X_test_scaled, y_test)
    print(f"两次log转换后的决定系数（R²）：{r2_score2}")

    # Define the MRSE custom scoring function
    def custom_mrse(y_true, y_pred):
        sum = 0
        # print(y_pred / y_true)
        for i in range(len(y_true)):
            a = inverselogt(y_pred[i])
            b = inverselogt(y_true[i])
            sum += (((a / b) - 1) ** 2)

        return sum/ len(y_true)
    # Create the custom scorer
    mrse_scorer = make_scorer(custom_mrse, greater_is_better=True)

    #cross validation
    cv_r2 = cross_val_score(xgb_model, independent_variables, dependent_values, cv=4, scoring='r2')
    cv_mrse = cross_val_score(xgb_model, independent_variables, dependent_values, cv=4, scoring = mrse_scorer)
    
    print(f"cross validation 均相对平方误差（Mean Relative Squared Error）：{cv_mrse}")
    print(f"cross validation r2：{cv_r2}")
    print(f"cross validation 平均 均相对平方误差（Mean Relative Squared Error）：{cv_mrse.mean()}")
    print(f"cross validation 平均 r2：{cv_r2.mean()}")

    y_test = inverselogt(y_test)
    # 绘制散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('True Values vs. Predicted Values (Log Scale)')
    plt.xlabel('True Values (Log Scale)')
    plt.ylabel('Predicted Values (Log Scale)')
    plt.show()

    # # 绘制残差图
    # plt.figure(figsize=(10, 6))
    # residuals = y_test - y_pred
    # plt.scatter(y_test, residuals, color='red', alpha=0.7)
    # plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    # plt.title('Residuals vs. True Values')
    # plt.xlabel('True Values')
    # plt.ylabel('Residuals')
    # plt.show()

def Ridge_MLR(regression_lst,params,test_size=0.3,):
    #Multiple Linear Ridge Regression
    data = regression_lst[:]
    # print(data)
    independent_variables = np.array([item[1] for item in data])
    dependent_values = np.array([item[2][0] for item in data])

    print(independent_variables.shape)
    # print(dependent_values.shape)
    # independent_variables.reshape(-1,1)
    # print(independent_variables)
    y_max = max([item[2][0] for item in data])
    y_min = min([item[2][0] for item in data])
    sum = 0 
    for item in data:
        sum += item[2][0]
    y_mean = sum / len(data)
    print(f'y_max: {y_max}')
    print(f'y_min: {y_min}')
    print(f'y mean: {y_mean}')
    scalar = RobustScaler()
    # independent_variables = scalar.fit_transform(independent_variables)
    independent_variables = logt(independent_variables)
    dependent_values = logt(dependent_values)
    independent_variables = scalar.fit_transform(independent_variables)
    if len(data[0]) == 4:
        rbf_variables = np.array([item[3] for item in data])
        independent_variables = np.concatenate((independent_variables,rbf_variables),axis=1)
    X_train, X_test, y_train, y_test = train_test_split(independent_variables, dependent_values, test_size=test_size, random_state=42)

    X_train_scaled = X_train
    X_test_scaled = X_test
    # X_train_scaled = scalar.transform(X_train)
    # print(X_train_scaled)
    # X_test_scaled = scalar.transform(X_test)
    # y_train = y_train.reshape(-1,1)
    # y_test = y_test.reshape(-1,1)
    # dependent_values = dependent_values.reshape(-1,1)
    # y_train_scaled = scalar.fit_transform(y_train)
    # y_test_scaled = scalar.fit_transform(y_test)
    
    print('-------------------------------')
    # print(X_train)
    print('-------------------------------')


    xgb_model = Ridge(alpha= params)
    xgb_model.fit(X_train_scaled, y_train) #fit模型
    #在test数据集上测试模型
    y_pred = xgb_model.predict(X_test_scaled) #第一次predict y 使用xtest
    y_pred = inverselogt(y_pred)
    r2_score = xgb_model.score(X_test_scaled, y_test)
    print(f"决定系数（R²）：{r2_score}")
    y_test = inverselogt(y_test)
    y_train = inverselogt(y_train)
    # y_pred = scalar.inverse_transform(y_pred)
    y_pred_train = xgb_model.predict(X_train_scaled) #第二次predict y 使用xtrain
    y_pred_train = inverselogt(y_pred_train)
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
    # print(len(y_test))
    count = 0 
    count2 = 0
    for i in range(len(y_test)):
        # print(y_pred[i])
        # print(y_test[i])
        # k = ((y_pred[i][0])/(y_test[i][0])) 
        # if k > 2:
        #     # print(k)
        #     # print(y_pred[i][0])
        #     # print(y_test[i][0])
        #     count += 1
        # else:
        mrse += (((y_pred[i])/(y_test[i])) - 1)**2
        count2 += 1
    print('----------------------------------------')
    # print(count)
    print(count2)
    mrse = mrse/count2

    count = 0 
    count2 = 0
    mrse_train = 0
    for i in range(len(y_train)):
        # print(y_pred_train[i])
        # print(y_train[i])
        # k = ((y_pred_train[i][0])/(y_train[i][0])) 
        # if k > 2:
        #     count += 1
        # else:
            mrse_train += (((y_pred_train[i])/(y_train[i])) - 1)**2
            count2 += 1
    print('----------------------------------------')
    # print(count)
    print(count2)
    mrse_train = mrse_train/count2

    normalized_mrse1 = mrse / (y_max - y_min)
    normalized_mrse2 = mrse / y_mean

    mae = mean_squared_error(y_test,y_pred)
    print(f"均方误差（Mean Squared Error）：{mse}")
    print(f"均相对平方误差（Mean Relative Squared Error）：{mrse}")
    print(f"train 均相对平方误差（Mean Relative Squared Error -- train）：{mrse_train}")
    print(f"平均绝对误差（Mean Absolute Error）：{mae}")
    # print(f'标准化mrse1:{normalized_mrse1}')
    # print(f'标准化mrse2:{normalized_mrse2}')



    # 打印学习参数
    coefficients = xgb_model.coef_
    for feature_index, coefficient in enumerate(coefficients):
        print(f"特征 {feature_index+1}: {coefficient}")
    # 在测试集上计算决定系数
    y_test = logt(y_test)
    r2_score2 = xgb_model.score(X_test_scaled, y_test)
    print(f"两次log转换后的决定系数（R²）：{r2_score2}")

    # Define the MRSE custom scoring function
    def custom_mrse(y_true, y_pred):
        sum = 0
        # print(y_pred / y_true)
        for i in range(len(y_true)):
            a = inverselogt(y_pred[i])
            b = inverselogt(y_true[i])
            sum += (((a / b) - 1) ** 2)

        return sum/ len(y_true)
    # Create the custom scorer
    mrse_scorer = make_scorer(custom_mrse, greater_is_better=True)

    #cross validation
    cv_r2 = cross_val_score(xgb_model, independent_variables, dependent_values, cv=4, scoring='r2')
    cv_mrse = cross_val_score(xgb_model, independent_variables, dependent_values, cv=4, scoring = mrse_scorer)
    
    print(f"cross validation 均相对平方误差（Mean Relative Squared Error）：{cv_mrse}")
    print(f"cross validation r2：{cv_r2}")
    print(f"cross validation 平均 均相对平方误差（Mean Relative Squared Error）：{cv_mrse.mean()}")
    print(f"cross validation 平均 r2：{cv_r2.mean()}")

    y_test = inverselogt(y_test)
    # 绘制散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('True Values vs. Predicted Values (Log Scale)')
    plt.xlabel('True Values (Log Scale)')
    plt.ylabel('Predicted Values (Log Scale)')
    plt.show()

    # # 绘制残差图
    # plt.figure(figsize=(10, 6))
    # residuals = y_test - y_pred
    # plt.scatter(y_test, residuals, color='red', alpha=0.7)
    # plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    # plt.title('Residuals vs. True Values')
    # plt.xlabel('True Values')
    # plt.ylabel('Residuals')
    # plt.show()

def SVR_MLR(regression_lst,params,test_size=0.3,):
    #Multiple Linear Ridge Regression
    data = regression_lst[:]
    # print(data)
    independent_variables = np.array([item[1] for item in data])
    dependent_values = np.array([item[2][0] for item in data])

    print(independent_variables.shape)
    # print(dependent_values.shape)
    # independent_variables.reshape(-1,1)
    # print(independent_variables)
    y_max = max([item[2][0] for item in data])
    y_min = min([item[2][0] for item in data])
    sum = 0 
    for item in data:
        sum += item[2][0]
    y_mean = sum / len(data)
    print(f'y_max: {y_max}')
    print(f'y_min: {y_min}')
    print(f'y mean: {y_mean}')
    scalar = RobustScaler()
    scalar2 = MinMaxScaler()
    scalar3 = MinMaxScaler()
    independent_variables = logt(independent_variables)
    dependent_values = logt(dependent_values)
    independent_variables = scalar.fit_transform(independent_variables)
    independent_variables = scalar2.fit_transform(independent_variables)
    if len(data[0]) == 4:
        rbf_variables = np.array([item[3] for item in data])
        rbf_variables = rbf_variables.reshape(-1,1)
        rbf_variables = scalar3.fit_transform(rbf_variables)
        independent_variables = np.concatenate((independent_variables,rbf_variables),axis=1)
    X_train, X_test, y_train, y_test = train_test_split(independent_variables, dependent_values, test_size=test_size, random_state=42)

    X_train_scaled = X_train
    X_test_scaled = X_test
    # X_train_scaled = scalar.transform(X_train)
    # print(X_train_scaled)
    # X_test_scaled = scalar.transform(X_test)
    # y_train = y_train.reshape(-1,1)
    # y_test = y_test.reshape(-1,1)
    # dependent_values = dependent_values.reshape(-1,1)
    # y_train_scaled = scalar.fit_transform(y_train)
    # y_test_scaled = scalar.fit_transform(y_test)
    
    print('-------------------------------')
    # print(X_train)
    print('-------------------------------')

    xgb_params = params
    xgb_model = SVR(**xgb_params)
    xgb_model.fit(X_train_scaled, y_train) #fit模型
    #在test数据集上测试模型
    y_pred = xgb_model.predict(X_test_scaled) #第一次predict y 使用xtest
    y_pred = inverselogt(y_pred)
    r2_score = xgb_model.score(X_test_scaled, y_test)
    print(f"决定系数（R²）：{r2_score}")
    y_test = inverselogt(y_test)
    y_train = inverselogt(y_train)
    # y_pred = scalar.inverse_transform(y_pred)
    y_pred_train = xgb_model.predict(X_train_scaled) #第二次predict y 使用xtrain
    y_pred_train = inverselogt(y_pred_train)
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
    # print(len(y_test))
    count = 0 
    count2 = 0
    for i in range(len(y_test)):
        # print(y_pred[i])
        # print(y_test[i])
        # k = ((y_pred[i][0])/(y_test[i][0])) 
        # if k > 2:
        #     # print(k)
        #     # print(y_pred[i][0])
        #     # print(y_test[i][0])
        #     count += 1
        # else:
        mrse += (((y_pred[i])/(y_test[i])) - 1)**2
        count2 += 1
    print('----------------------------------------')
    # print(count)
    print(count2)
    mrse = mrse/count2

    count = 0 
    count2 = 0
    mrse_train = 0
    for i in range(len(y_train)):
        # print(y_pred_train[i])
        # print(y_train[i])
        # k = ((y_pred_train[i][0])/(y_train[i][0])) 
        # if k > 2:
        #     count += 1
        # else:
            mrse_train += (((y_pred_train[i])/(y_train[i])) - 1)**2
            count2 += 1
    print('----------------------------------------')
    # print(count)
    print(count2)
    mrse_train = mrse_train/count2

    normalized_mrse1 = mrse / (y_max - y_min)
    normalized_mrse2 = mrse / y_mean

    mae = mean_squared_error(y_test,y_pred)
    print(f"均方误差（Mean Squared Error）：{mse}")
    print(f"均相对平方误差（Mean Relative Squared Error）：{mrse}")
    print(f"train 均相对平方误差（Mean Relative Squared Error -- train）：{mrse_train}")
    print(f"平均绝对误差（Mean Absolute Error）：{mae}")
    # print(f'标准化mrse1:{normalized_mrse1}')
    # print(f'标准化mrse2:{normalized_mrse2}')



    # # 打印学习参数
    # coefficients = xgb_model.dual_coef_
    # for feature_index, coefficient in enumerate(coefficients):
    #     print(f"特征 {feature_index+1}: {coefficient}")
    # 在测试集上计算决定系数
    y_test = logt(y_test)
    r2_score2 = xgb_model.score(X_test_scaled, y_test)
    print(f"两次log转换后的决定系数（R²）：{r2_score2}")

    # Define the MRSE custom scoring function
    def custom_mrse(y_true, y_pred):
        sum = 0
        # print(y_pred / y_true)
        for i in range(len(y_true)):
            a = inverselogt(y_pred[i])
            b = inverselogt(y_true[i])
            sum += (((a / b) - 1) ** 2)

        return sum/ len(y_true)
    # Create the custom scorer
    mrse_scorer = make_scorer(custom_mrse, greater_is_better=True)

    #cross validation
    cv_r2 = cross_val_score(xgb_model, independent_variables, dependent_values, cv=4, scoring='r2')
    cv_mrse = cross_val_score(xgb_model, independent_variables, dependent_values, cv=4, scoring = mrse_scorer)
    
    print(f"cross validation 均相对平方误差（Mean Relative Squared Error）：{cv_mrse}")
    print(f"cross validation r2：{cv_r2}")
    print(f"cross validation 平均 均相对平方误差（Mean Relative Squared Error）：{cv_mrse.mean()}")
    print(f"cross validation 平均 r2：{cv_r2.mean()}")

    y_test = inverselogt(y_test)
    # 绘制散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('True Values vs. Predicted Values (Log Scale)')
    plt.xlabel('True Values (Log Scale)')
    plt.ylabel('Predicted Values (Log Scale)')
    plt.show()

    # # 绘制残差图
    # plt.figure(figsize=(10, 6))
    # residuals = y_test - y_pred
    # plt.scatter(y_test, residuals, color='red', alpha=0.7)
    # plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    # plt.title('Residuals vs. True Values')
    # plt.xlabel('True Values')
    # plt.ylabel('Residuals')
    # plt.show()

def GridS_XGB(regression_lst,params,param_dist,test_size=0.3):
    #Multiple Linear Ridge Regression
    data = regression_lst[:]
    independent_variables = np.array([item[1] for item in data])
    dependent_values = np.array([item[2][0] for item in data])


    scalar = RobustScaler()
    scalar2 = MinMaxScaler()
    independent_variables = logt(independent_variables)
    dependent_values = logt(dependent_values)
    independent_variables = scalar.fit_transform(independent_variables)
    if len(data[0]) == 4:
        rbf_variables = np.array([item[3] for item in data])
        # rbf_variables = rbf_variables.reshape(-1,1)
        # rbf_variables = scalar2.fit_transform(rbf_variables)
        independent_variables = np.concatenate((independent_variables,rbf_variables),axis=1)
    X_train, X_test, y_train, y_test = train_test_split(independent_variables, dependent_values, test_size=test_size, random_state=42)

    X_train_scaled = X_train
    X_test_scaled = X_test

    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    dependent_values = dependent_values.reshape(-1,1)

    xgb_params = params
    xgb_model = XGBRegressor(**xgb_params)
    xgb_model.fit(X_train_scaled, y_train) #fit模型
    #在test数据集上测试模型
    y_pred = xgb_model.predict(X_test_scaled) #第一次predict y 使用xtest
    y_pred = inverselogt(y_pred)
    r2_score = xgb_model.score(X_test_scaled, y_test)
    print(f"决定系数（R²）：{r2_score}")
    y_test = inverselogt(y_test)
    y_train = inverselogt(y_train)

    y_pred_train = xgb_model.predict(X_train_scaled) #第二次predict y 使用xtrain
    y_pred_train = inverselogt(y_pred_train)

    mse = mean_squared_error(y_test, y_pred)
    mrse = 0 
    count2 = 0
    for i in range(len(y_test)):
        mrse += (((y_pred[i])/(y_test[i])) - 1)**2
        count2 += 1
    mrse = mrse/count2

    count2 = 0
    mrse_train = 0
    for i in range(len(y_train)):
        mrse_train += (((y_pred_train[i])/(y_train[i])) - 1)**2
        count2 += 1
    mrse_train = mrse_train/count2


    mae = mean_squared_error(y_test,y_pred)
    print(f"均方误差（Mean Squared Error）：{mse}")
    print(f"均相对平方误差（Mean Relative Squared Error）：{mrse}")
    print(f"train 均相对平方误差（Mean Relative Squared Error -- train）：{mrse_train}")
    print(f"平均绝对误差（Mean Absolute Error）：{mae}")

    # Define the MRSE custom scoring function
    def custom_mrse(y_true, y_pred):
        sum = 0
        # print(y_pred / y_true)
        for i in range(len(y_true)):
            a = inverselogt(y_pred[i])
            b = inverselogt(y_true[i])
            sum += (((a / b) - 1) ** 2)

        return sum/ len(y_true)
    # Create the custom scorer
    mrse_scorer = make_scorer(custom_mrse, greater_is_better=False)

    #cross validation
    cv_r2 = cross_val_score(xgb_model, independent_variables, dependent_values, cv=4, scoring='r2')
    cv_mrse = cross_val_score(xgb_model, independent_variables, dependent_values, cv=4, scoring = mrse_scorer)
    
    print(f"cross validation 平均 均相对平方误差（Mean Relative Squared Error）：{cv_mrse.mean()}")
    print(f"cross validation 平均 r2：{cv_r2.mean()}")

    random_search = GridSearchCV(
        xgb_model, param_grid=param_dist, 
         scoring=mrse_scorer, cv=5,n_jobs=-1
    )
    # 运行随机搜索
    random_search.fit(independent_variables, dependent_values)
    # 获取最优参数
    best_params = random_search.best_params_
    print("Best parameters:", best_params)

    # 使用最优参数训练模型
    best_xgb_model = XGBRegressor(**best_params)

    #cross validation round2
    cv2_r2 = cross_val_score(best_xgb_model, independent_variables, dependent_values, cv=4, scoring='r2')
    cv2_mrse = cross_val_score(best_xgb_model, independent_variables, dependent_values, cv=4, scoring = mrse_scorer)
    

    print(f"cross validation2 平均 均方根误差（Mean Root Squared Error）：{cv2_mrse.mean()}")
    print(f"cross validation2 平均 r2：{cv2_r2.mean()}")

    y_test = inverselogt(y_test)
    # # 绘制散点图
    # plt.figure(figsize=(10, 6))
    # plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.title('True Values vs. Predicted Values (Log Scale)')
    # plt.xlabel('True Values (Log Scale)')
    # plt.ylabel('Predicted Values (Log Scale)')
    # plt.show()

