from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertModel, LongformerTokenizerFast, LongformerModel
import subprocess
import pickle
import os

'''对标题tokenize'''

pathin = "quality_title_dict.pkl"
pathout = "title_top_0.3_viral_tokens.pkl"

def main(pathin, pathout):
    tokens = {}
    string_lst = get_feature(pathin)
    max_length = max(len(title) for id, title in string_lst.items())
    print(max_length)

    # print(string_lst)
    # print(string_lst)
    for id, title in string_lst.items():
        token = tokenize(title, max_length)
        tokens[id] = token
    # print(token)
    #     tokens[id] = token
    with open(pathout, "wb") as file:
        pickle.dump(tokens, file)

    



def get_feature(path):

    outputdict = {}
    max_length = 0

    f = open(path, "rb")  # a file that consist of path to the video, likes, and total views

    title_dict = pickle.load(f)
    # print(title_dict)
    for id, title in title_dict.items():
        outputdict[id] = title           

    f.close()
    return outputdict

def tokenize(title, max_length):
    print(title)  # check the format

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    # only process the fist 10 action labels of each sequence (to get output matrices of the same size)
    # 也可以其中最长的长度为基准进行处理
    # max_length = max(len(tokenizer.encode(text)) for text in string_lst)
    tokens = tokenizer(title, return_tensors='pt', truncation=True, padding='max_length', max_length=max_length)

    # we only get an int (input_ids) and attention masks for each word

    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    embedded = model(tokens["input_ids"]).last_hidden_state
    # we put the ids(mathematical representations of the words) into a net work and get the returned values(a matrix of the same size for each words)


    # decode = tokenizer.batch_decode(sequences = tokens["input_ids"], skip_special_tokens = True)
    X = embedded  # X is a numpy array consists of feature matrices (each action label is transform into the corresponding row of matrix)
    X = X.detach().numpy()
    X = X.reshape(X.shape[0], -1) #reshape the feature matrix of each video to a feature vector so that it can be put into the kernel
    print(X.shape)
    # y = np.array(y) # y is an array consists of the "popularity" of videos
    # kernel = 50.0**2 * RBF(length_scale=50.0)
    # krr = KernelRidge(alpha=1.0, kernel=kernel)
    # krr.fit(X, y)
    return X

main(pathin, pathout)