import os
import pickle

output_file = 'top_0.3_viral_tokens.pkl'

def merge_data_from_file(file_path):
    with open(file_path, "rb") as file:
        data_dict = pickle.load(file)
    return data_dict

def process_and_merge(paths, output_file, batch_size=100):
    output = {}
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i+batch_size]
        batch_data = [merge_data_from_file(path) for path in batch_paths]
        for data_dict in batch_data:
            output.update(data_dict)
            del data_dict
            

    with open(output_file, "wb") as output_file:
        pickle.dump(output, output_file)

if __name__ == "__main__":
    paths = ['top_0.3_viral_tokens_Alan.pkl', 'top_0.3_viral_tokens_Alex.pkl', 'top_0.3_viral_tokens_Ken.pkl', 'top_0.3_viral_tokens_Qiaosong.pkl']
    process_and_merge(paths, output_file, batch_size=2)
    print('done')
