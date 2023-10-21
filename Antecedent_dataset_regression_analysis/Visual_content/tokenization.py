from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertModel
import subprocess
import pickle

    
y = []
string_lst = []

f = open("videos_data.txt", "r")  # a file that consist of path to the video, likes, and total views
for i in range(3):  # 
    content = f.readline().split(",")
    video_id = content[0].split("/")[-1]  # getting the video id
    with open(f'Visual_content/video_pkl_Qiaosong/{video_id}.pkl', 'rb') as video_file:
        loaded_data = pickle.load(video_file)
        action_labels = loaded_data["actions"]
        # processing the action labels into a string
        for i in range(len(action_labels)):
            action_labels[i] = action_labels[i].strip('"')
        data = " ".join(action_labels)
        string_lst.append(data)
        video_file.close()

f.close()
print(string_lst)  # check the format

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
# only process the fist 10 action labels of each sequence (to get output matrices of the same size)
# 也可以其中最长的长度为基准进行处理
tokens = tokenizer(string_lst, return_tensors='pt',  truncation=True, padding= 'max_length', max_length=10)
# we only get an int (input_ids) and attention masks for each word

model = DistilBertModel.from_pretrained("distilbert-base-uncased")
embedded = model(tokens["input_ids"]).last_hidden_state 
# we put the ids(mathematical representations of the words) into a net work and get the returned values(a matrix of the same size for each words)


# decode = tokenizer.batch_decode(sequences = tokens["input_ids"], skip_special_tokens = True)
X = embedded  # X is a numpy array consists of feature matrices (each action label is transform into the corresponding row of matrix)
X = X.detach().numpy()
X = X.reshape(X.shape[0], -1) #reshape the feature matrix of each video to a feature vector so that it can be put into the kernel
y = np.array(y) # y is an array consists of the "popularity" of videos
kernel = 50.0**2 * RBF(length_scale=50.0)
krr = KernelRidge(alpha=1.0, kernel=kernel)
krr.fit(X, y)