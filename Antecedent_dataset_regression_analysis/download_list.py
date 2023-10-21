
import subprocess
import os
root_dir = "/Volumes/T7/youtube_data/youtime/top_0.3_memoryless"

def download(video_id):
    command = f"yt-dlp 'https://www.youtube.com/watch?v={video_id}' --output \"/Volumes/T7/youtube_data/youtime/top_0.3_memoryless/{video_id}.mp4\""
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()

    # 打印输出结果
    print("命令行输出：")
    print(output.decode("utf-8"))
    # 打印错误信息
    if error:
        print("错误信息：")
        print(error.decode("utf-8"))

def main():
    with open('top_thres.3_memoryless_checked.txt', 'r') as file:
        for line in file:
            video_id = line.strip()
            if not check_dup(root_dir,video_id):
                download(video_id)
            else: 
                continue
    print('done')

def check_dup(root_dir, search_str):
    found = False
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if (search_str in file) and ('.part' not in file):
                found = True
    return found

main()