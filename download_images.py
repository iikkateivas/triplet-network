# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 20:06:29 2019

@author: Iikka
"""

import pandas as pd
import requests
import shutil
import os
from io import BytesIO
from PIL import Image

path = 'faces'
if not os.path.exists(path):
    os.makedirs(path)

df=pd.read_csv('facescrub_actresses.txt', sep='\t')
count = 0

for index, row in df.iterrows():
    print(index)

    try:
        response = requests.get(row['url'], stream=True, timeout=2)
    except requests.exceptions.Timeout:
        print('The request timed out')
        continue
    except requests.exceptions.ConnectionError:
        print("Connection refused")
        continue
    except:
        continue
    else:
        print('OK')
    
    if response.status_code != 200:
        continue

    if(response.headers.get('Content-Length') and int(response.headers.get('Content-Length')) < 2000):
        continue
    
    bbox = row['bbox']
    split_bbox = bbox.split(',')

    try:
        img = Image.open(BytesIO(response.content))
    except:
        continue
    
    crop = img.crop((int(split_bbox[0]),int(split_bbox[1]),int(split_bbox[2]),int(split_bbox[3])))   
    
    name = row['name'].replace(" ", "_")
    if not os.path.exists(path + '\\' + name):
        os.makedirs(path + '\\' + name)
        
    fn = path + '\\' + name + '\\' + name + '_' + str(row['image_id']) + '.jpg'
    
    crop.convert('RGB').save(fn)
    

