from utils import *
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split



def load_visual_features(type):
    file_path = f'top_{type}_tokens_len=500.pkl'
    with open(file_path,'rb') as file:
        token_data = pickle.load(file)
    file.close()
    return token_data

def select_view_data(type,view_raw):
    view_and_content = classfy(type,view_raw)
    return view_and_content

def preprocess_data(view_and_tokens):
    data = view_and_tokens
    view_data = np.array([item[1] for item in data])
    dependent_values = np.array([item[2][0] for item in data])
    scalar = RobustScaler()
    view_data = logt(view_data)
    dependent_values = logt(dependent_values)
    view_data = scalar.fit_transform(view_data) 
    raw_content_data = np.array([item[3] for item in data]) 
    return view_data, dependent_values, raw_content_data

def generate(CLASS):
    tokenized_datasets = load_visual_features(CLASS)
    view_raw = read_file("top_dao.csv",tokenized_datasets,True,7,29,1000)
    view_and_tokens = select_view_data(CLASS, view_raw)
    view_data, y, raw_content_data = preprocess_data(view_and_tokens)
    print(view_data.shape)
    raw_content_data = np.squeeze(raw_content_data, axis=1)
    print(raw_content_data.shape)
    view_and_content = np.concatenate((view_data,raw_content_data),axis=1)

    dataset = Dataset.from_dict({'x': view_and_content, 'y': y})

    train_x, test_x, train_y, test_y = train_test_split(dataset['x'], dataset['y'], test_size=0.3, random_state=42)
    train_dataset = Dataset.from_dict({'x': train_x, 'y': train_y})
    test_dataset = Dataset.from_dict({'x': test_x, 'y': test_y})
    final_dataset_dict = DatasetDict({'train': train_dataset, 'test': test_dataset})
    # print((final_dataset_dict['test']['x'][2]))
    final_dataset_dict.save_to_disk(f'top_{CLASS}_dataset_500')

generate('viral')


