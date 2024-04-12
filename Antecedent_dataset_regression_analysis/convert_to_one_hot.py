from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertModel, LongformerTokenizerFast, LongformerModel, BertTokenizer
import subprocess
import pickle
import os
import torch
from sklearn.feature_extraction.text import CountVectorizer
import json

pathin = "top_thres0.3_quality_checked.txt2"
pathout = "top_quality_onehot_vector.pkl"
with open('kinetics_classnames.json', 'r') as json_file:
    classes = json.load(json_file)
    lexicon = classes.keys()
    lexicon = list(lexicon)
    for i in range(len(lexicon)):
                    lexicon[i] = lexicon[i].strip('"')
                    lexicon[i] = '_'.join(lexicon[i].split())

    
vectorizer = CountVectorizer(vocabulary=lexicon)

'''基础tokenize方法，使用longformer处理长语句'''
def main(pathin, pathout):
    tokens = {}
    string_lst, idlst = get_feature(pathin)

    with open(pathout, "wb") as file:
        pickle.dump(string_lst, file)

def get_feature(path):
    string_lst = []
    idlst = []
    outputdict = {}
    max_length = 0
    f = open(path, "r")  # a file that consist of path to the video, likes, and total views
    unique = set()
    for line in f: # 
        content = line.strip('\n')
        video_id = content  # getting the video id
        
        for foldername, subfolders, filenames in os.walk('Visual_content/video_pkl_quality'):
            for filename in filenames:
                if video_id in filename:
                    break    
        try:
            
            with open(f'Visual_content/video_pkl_quality/{video_id}.mp4.pkl', 'rb') as video_file:
                loaded_data = pickle.load(video_file)
                action_labels = loaded_data["actions"]
                # processing the action labels into a string
                for i in range(len(action_labels)):
                    action_labels[i] = action_labels[i].strip('"')
                    action_labels[i] = '_'.join(action_labels[i].split())
                data = " ".join(action_labels)
                data = data.split()
                data = data[:501]
                unique = unique.union(set(data))
                
                BOW_data = " ".join(data)
                vector = vectorizer.fit_transform([BOW_data])
                idlst.append(video_id)
                outputdict[video_id] = vector
                video_file.close()
        except FileNotFoundError:
            print('video not avaliable')
    print(len(unique))
    f.close()
    return outputdict, idlst

# def tokenize(action):
#     print(action)  # check the format

#     tokenizer = tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     # only process the fist 10 action labels of each sequence (to get output matrices of the same size)
#     # 也可以其中最长的长度为基准进行处理
#     # max_length = max(len(tokenizer.encode(text)) for text in string_lst)
#     tokens = tokenizer(action, return_tensors='pt',  truncation=True, padding= 'max_length', max_length = 500 )
#     # we only get an int (input_ids) and attention masks for each word
#     tokens = tokens["input_ids"]
#     print(type(tokens))
#     print(tokens.shape)
#     return tokens

main(pathin, pathout)