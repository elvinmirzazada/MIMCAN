# Created by elvinmirzazada at 20:20 28/05/2023 using PyCharm
import pandas as pd
from tqdm import tqdm
import shutil
import os

data = pd.read_csv('similarities_politifact_real.csv', delimiter='~', index_col=False)
FILE_PATH = 'politifact_images/'
DEST_DIR = 'filtered_gossipcop_imgs_by_similarity/real/'

def filter_images(row, dest_dir, file_path):
    sorted_row = row.sort_values('similarity', ascending=False)
    count = 0
    for i, r in sorted_row.iterrows():
        if r['similarity'] <= 0.1:
            return
        if count >= 3:
            return
        id = r['id']
        url: str = r['url']
        url = url.replace('http://', '').replace('https://', '').replace('/', '!')
        img_name = r['img']
        src_path = file_path + id + '/' + img_name
        dest_path = dest_dir + url + '/' + img_name
        try:
            os.mkdir(dest_dir + url)
        except:
            pass
        shutil.copy(src_path, dest_path)
        count += 1


sorted_data = data.groupby('id').apply(lambda x: filter_images(x, DEST_DIR, FILE_PATH))


