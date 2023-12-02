# Created by elvinmirzazada at 20:05 17/06/2023 using PyCharm
import re
import math
import os
import numpy as np
import shutil
import pandas as  pd
from transformers import pipeline
from collections import Counter


WORD = re.compile(r"\w+")
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)


def calc_similarity(text: str, img_path: str):
    img_caption = captioner(img_path)[0]['generated_text']
    vector_text = text_to_vector(text)
    vector_img = text_to_vector(img_caption)
    similarity = get_cosine(vector_text, vector_img)
    return similarity


def get_n_images_by_similarity(id: str, text: str, img_dir: str, dest_dir: str, img_size: int = 3):
    file_dir = f"{img_dir}/{id}"
    if not os.path.exists(file_dir):
        return
    image_files = os.listdir(file_dir)
    similarities = []

    for img in image_files:
        try:
            img_path = f"{file_dir}/{img}"
            similarity = calc_similarity(text, img_path)
            similarities.append(similarity)
        except:
            pass

    sorted_indexes = np.argsort(similarities)[::-1][:img_size]
    os.makedirs(f"{dest_dir}/{id}", exist_ok=True)
    for idx in sorted_indexes:
        img_file = image_files[idx]
        shutil.copy(f"{file_dir}/{img_file}", f"{dest_dir}/{id}")


def clean_all_gossipcop_images():
    data = pd.read_csv('datasets/gossipcop_fake.csv')
    for row in data.values:
        id = row[0]
        title = row[2]
        get_n_images_by_similarity(id=id, text=title, img_dir='gossipcop_images', dest_dir='cleaned_gossipcop_imgs')

    data = pd.read_csv('datasets/gossipcop_real.csv')
    for row in data.values:
        id = row[0]
        title = row[2]
        get_n_images_by_similarity(id=id, text=title, img_dir='gossipcop_images', dest_dir='cleaned_gossipcop_imgs')


def clean_all_politifact_images():
    data = pd.read_csv('datasets/politifact_fake.csv')
    for row in data.values:
        id = row[0]
        title = row[2]
        get_n_images_by_similarity(id=id, text=title, img_dir='politifact_images', dest_dir='cleaned_politifact_imgs')

    data = pd.read_csv('datasets/politifact_real.csv')
    for row in data.values:
        id = row[0]
        title = row[2]
        get_n_images_by_similarity(id=id, text=title, img_dir='politifact_images', dest_dir='cleaned_politifact_imgs')


if __name__ == '__main__':
    clean_all_politifact_images()