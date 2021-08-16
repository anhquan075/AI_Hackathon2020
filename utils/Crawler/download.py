from tqdm import tqdm
import json
from urllib.request import Request, urlopen  
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
import argparse



def download_image(data_path):  
    folder_name = data_path.split('/')[-1].replace('.json',"")
    os.makedirs(folder_name, exist_ok=True)
    with open(data_path) as json_file:
        data = json.load(json_file)
    print("Download data in {} folder:".format(folder_name))
    for i in tqdm(range(len(data))):
        f =  open('./{}/image_{}.jpg'.format(folder_name,i),'wb')
        url = data[i]['url']
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            response = urlopen(req, timeout=1000).read()
            f.write(response)
            f.close()
            # print("Image number {} just downloaded".format(i))
        except:
            # print("Cannot download image {}".format(i))
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download image through file json')
    parser.add_argument('--key_word',type=str)

    arg = parser.parse_args()

    download_image(f'json_file/{arg.key_word}.json')