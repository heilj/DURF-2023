from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
import numpy as np

import subprocess
import pickle
import os

pathin = "top_thres0.3_viral_checked_Alan.txt"
pathout = "top_0.3_memoryless_tokens.pkl"

'''基础tokenize方法，使用longformer处理长语句'''

def main(pathin, pathout):
    tokens = {}
    get_feature(pathin)

    



def get_feature(path):

    string_lst = []
    idlst = []
    outputdict = {}
    max_length = 4000
    count_long = 0
    count = 0
    count_short = 0
    f = open(path, "r")  # a file that consist of path to the video, likes, and total views

    for line in f: # 
        content = line.strip('\n')
        video_id = content  # getting the video id
        for foldername, subfolders, filenames in os.walk('Visual_content/video_pkl_Alan'):
            for filename in filenames:
                if video_id in filename:
                    break
                
        try:
            
            with open(f'Visual_content/video_pkl_Alan/{video_id}.mp4.pkl', 'rb') as video_file:
                loaded_data = pickle.load(video_file)
                action_labels = loaded_data["actions"]
                # processing the action labels into a string
                for i in range(len(action_labels)):
                    action_labels[i] = action_labels[i].strip('"')
                data = " ".join(action_labels)[:]
                string_lst.append(data)
                if len(data) >= max_length:
                    count_long += 1
                else:
                    count_short += 1
                count += 1
                video_file.close()
        except FileNotFoundError:
            print('video not avaliable')

            



    f.close()
    print(f'thers:{max_length}of words')
    print(f'valid:{count_short}')
    print(f'invalid:{count_long}')
    print(f'total:{count}')



main(pathin, pathout)