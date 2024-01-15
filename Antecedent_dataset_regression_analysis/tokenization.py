from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertModel, LongformerTokenizerFast, LongformerModel, BertTokenizer, BertModel
import subprocess
import pickle
import os

pathin = "top_thres0.3_viral_checked.txt"
pathout = "top_viral_embedding_len=500.pkl"

'''基础tokenize方法，使用longformer处理长语句'''

def main(pathin, pathout):
    tokens = {}
    string_lst, idlst = get_feature(pathin)
    # print(string_lst)
    for id, action in string_lst.items():
        token = tokenize(action)
        tokens[id] = token
    # print(token)
    #     tokens[id] = token
    with open(pathout, "wb") as file:
        pickle.dump(tokens, file)

    



def get_feature(path):

    string_lst = []
    idlst = []
    outputdict = {}
    max_length = 0

    f = open(path, "r")  # a file that consist of path to the video, likes, and total views

    for line in f: # 
        content = line.strip('\n')
        video_id = content  # getting the video id
        
        for foldername, subfolders, filenames in os.walk('Antecedent_dataset_regression_analysis/Visual_content/video_pkl_viral'):
            for filename in filenames:
                if video_id in filename:
                    
                    break
                
        try:
            
            with open(f'Visual_content/video_pkl_viral/{video_id}.mp4.pkl', 'rb') as video_file:
                loaded_data = pickle.load(video_file)
                action_labels = loaded_data["actions"]
                # processing the action labels into a string
                for i in range(len(action_labels)):
                    action_labels[i] = action_labels[i].strip('"')
                data = " ".join(action_labels)
                data = data.split()
                data = data[:501]
                data = " ".join(data)
                idlst.append(video_id)
                outputdict[video_id] = data
                video_file.close()
        except FileNotFoundError:
            print('video not avaliable')

            



    f.close()
    return outputdict, idlst

def tokenize(action):
    print(action)  # check the format

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # only process the fist 10 action labels of each sequence (to get output matrices of the same size)
    # 也可以其中最长的长度为基准进行处理
    # max_length = max(len(tokenizer.encode(text)) for text in string_lst)
    tokens = tokenizer(action, return_tensors='pt',  truncation=True, padding= 'max_length', max_length = 500 )
    # we only get an int (input_ids) and attention masks for each word

    model = BertModel.from_pretrained('bert-base-uncased')
    embedded = model(tokens["input_ids"]).last_hidden_state
    # we put the ids(mathematical representations of the words) into a net work and get the returned values(a matrix of the same size for each words)


    # decode = tokenizer.batch_decode(sequences = tokens["input_ids"], skip_special_tokens = True)
    X = embedded  # X is a numpy array consists of feature matrices (each action label is transform into the corresponding row of matrix)
    X = X.detach().numpy()
     #reshape the feature matrix of each video to a feature vector so that it can be put into the kernel
    print(X.shape)
    # y = np.array(y) # y is an array consists of the "popularity" of videos
    # kernel = 50.0**2 * RBF(length_scale=50.0)
    # krr = KernelRidge(alpha=1.0, kernel=kernel)
    # krr.fit(X, y)
    return X

main(pathin, pathout)