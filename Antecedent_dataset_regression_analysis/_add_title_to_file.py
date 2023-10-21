import os
import pickle
import numpy
from googleapiclient.discovery import build

# 设置你的YouTube API 密钥
api_key = "AIzaSyDej5gwZOj9Kux1upr083BqZgjWcLvrYhk"

# 创建 YouTube 数据服务对象
youtube = build('youtube', 'v3', developerKey=api_key)

# 指定包含要重命名的文件的文件夹路径
folder_path = 'top_0.3_viral_tokens.pkl'

# 遍历文件夹中的文件
with open(folder_path,'rb') as file:
    oud_dict = pickle.load(file)
    for video_id in oud_dict:
        try:

            # 使用 videos().list 方法来获取视频信息
            video_info = youtube.videos().list(
                part='snippet',
                id=video_id
            ).execute()

            # 提取视频标题
            video_title = video_info['items'][0]['snippet']['title']
            oud_dict[video_id] = video_title
            #构建新的文件名，结合ID和标题
            # new_filename = f'{video_id}.mp4'

            # # 构建新的文件路径
            # new_filepath = os.path.join(folder_path, new_filename)

            # # 使用os.rename()重命名文件
            # os.rename(os.path.join(folder_path, filename), new_filepath)

            # print(f'已重命名文件：{filename} -> {new_filename}')

        except Exception as e:
            print(f'发生错误: {e}')
with open('viral_title_dict.pkl','wb') as file:
    pickle.dump(oud_dict,file)