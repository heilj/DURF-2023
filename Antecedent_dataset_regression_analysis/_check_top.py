import pandas as pd
import csv
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import *
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

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
    #去除无效数据（样本小于30天），并整理前7天的数据
    with open('top_avaliable.txt','r') as file:
        all_id = []
        temp = file.readlines()
        for line in temp:
            line.rstrip("\n")
            line.rstrip()
            line = line.rstrip('\n')
            # print(line)
            all_id.append(line)

        for i in range(len(result_list)):
            if result_list[i][0] in  all_id:
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

def main():
    file2lst = read_file('top_dao.csv')
    with open('top_to_check.txt','a') as file:
        for i in file2lst:
            file.write(str(i) + '\n')

main()