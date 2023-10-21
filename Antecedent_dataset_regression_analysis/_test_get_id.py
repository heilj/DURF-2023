import os
import shutil
from pathlib import Path
# from googleapiclient.discovery import build

# # 设置你的 YouTube API 密钥
# api_key ="AIzaSyArL_ejUfhrigo8BB5Ec7SFf0UXJhC-MOo"

# # 创建 YouTube 数据服务对象
# youtube = build('youtube', 'v3', developerKey=api_key)

# # 指定要获取标题的视频的 YouTube ID
# video_id = 'QzjfZX-t1uY'

# try:
#     # 使用 videos().list 方法来获取视频信息
#     video_info = youtube.videos().list(
#         part='snippet',
#         id=video_id
#     ).execute()

#     # 提取视频标题
#     video_title = video_info['items'][0]['snippet']['title']
#     print(video_info)
#     print(f'视频标题: {video_title}')

# except Exception as e:
#     print(f'发生错误: {e}')


import os
from pathlib import Path

folder_path = '/Volumes/T7/youtube_data/youtime/top_0.3_quality'
filename = 'ykTBoSe01_g.mp4'
new_filename = 'ykTBoSe01_g|I TACKLE CHICKS! (6/2/09-90).mp4'

source_path = os.path.join(folder_path, filename)
destination_path = os.path.join(folder_path, new_filename)
# 或者使用 Path 对象
# source_path = Path(folder_path) / filename
# destination_path = Path(folder_path) / new_filename

try:
    os.rename(source_path, destination_path)
    print(f'文件重命名成功: {source_path} -> {destination_path}')
except FileNotFoundError:
    print(f'文件不存在: {source_path}')
except Exception as e:
    print(f'发生错误: {e}')

