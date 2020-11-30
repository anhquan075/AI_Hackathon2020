import requests
import json
from urllib.request import Request, urlopen  
import urllib     
import time
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
import argparse
from http.client import IncompleteRead


def download_image(data_path):  
    folder_name = str(data_path).replace('.json',"")
    if os.path.exists(folder_name):
        pass
    else:
        os.mkdir(folder_name)
    with open(data_path) as json_file:
        data = json.load(json_file)
    print("Download data in {} folder".format(folder_name))
    for i in range(len(data)):
        f =  open('./{}/image_{}.jpg'.format(folder_name,i),'wb')
        url = data[i]['url']
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            response = urlopen(req, timeout=1000).read()
            f.write(response)
            f.close()
            print("Image number {} just downloaded".format(i))
        except urllib.error.URLError:
            print("Cannot download image {}".format(i))
            continue
        except IncompleteRead:
            print("Cannot download image {}".format(i))
            continue
        except ConnectionResetError:
            print("Cannot download image {}".format(i))
            continue

download_image('cay_do.json')