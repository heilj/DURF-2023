from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
import numpy as np

import subprocess
import pickle
import os

pathin = "top_thres0.3_viral_checked_Ken.txt"
def main(pathin):
    tokens = {}
    get_feature(pathin)

def get_feature(path):
    f = open(path, "r")  # a file that consist of path to the video, likes, and total views

    for line in f: # 
        content = line.strip('\n')
        video_id = content  # getting the video id
        print(video_id)
        for foldername, subfolders, filenames in os.walk('Visual_content/video_pkl_Ken'):
            for filename in filenames:
                if video_id in filename:
                    print('change')
                    print(video_id)
                    new_name = video_id + '.mp4.pkl'
                    os.rename('Visual_content/video_pkl_Ken/' + filename,'Visual_content/video_pkl_Ken/'+new_name)

    f.close()

main(pathin)