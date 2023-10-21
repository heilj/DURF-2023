import requests

def request(videoid):
    API_KEY = "AIzaSyARSAdSqnfB84u9Qz5VkCZNDKaFDErYmNI"
    VIDEO_ID = videoid

    url = f'https://www.googleapis.com/youtube/v3/videos?id={VIDEO_ID}&key={API_KEY}&part=status'

    response = requests.get(url)
    data = response.json()

    if 'items' in data and len(data['items']) > 0:
        video_status = data['items'][0]['status']['uploadStatus']
        if video_status == 'deleted':
            return False
        else:
            return True
    else:
        return False

def main():
    count = 0
    with open('top_thres0.3_quality_checked.txt', 'r') as file:
        for line in file:
            line = line.split(' ')
            thres = line[0]
            video_id = line[0]
            video_id = video_id.strip('\n')  # Remove newline character
            
            k = request(video_id)
            if k:
                count += 1
                print(video_id)
                with open('top_thres0.3_quality_checked.txt2', 'a') as f:
                    f.write(video_id + '\n')
    print(count)

main()
