# Created by elvinmirzazada at 12:51 21/05/2023 using PyCharm
from transformers import pipeline
import math
import re
import os
from collections import Counter
import pandas as pd
import csv
from tqdm import tqdm

#
# text1 = captioner("gossipcop_images/gossipcop-843759/ZooGiraffeBabyMother-901505.jpg")[0]['generated_text']
# text2 = "April the giraffe baby sex revealed: Giraffe calf is a BOY"
# print(text1, text2)
WORD = re.compile(r"\w+")


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


# text1 = "This is a foo bar sentence ."
# text2 = "This sentence is similar to a foo bar sentence ."

# vector1 = text_to_vector(text1)
# vector2 = text_to_vector(text2)
#
# cosine = get_cosine(vector1, vector2)
#
# print("Cosine:", cosine)
#
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")


def calc_sim(row, dir_p, name):
    title = row[2]
    url = row[1]
    if not os.path.exists(f"{dir_p}/{row[0]}"):
        return
    image_files = os.listdir(f"{dir_p}/{row[0]}")
    with open(f'similarities_{name}.csv', 'a', newline='') as file:
        writer = csv.writer(file, delimiter='~')

        for img in image_files:
            try:
                img_path = f"{dir_p}/{row[0]}/{img}"
                img_caption = captioner(img_path)[0]['generated_text']
                vector1 = text_to_vector(title)
                vector2 = text_to_vector(img_caption)
                cosine = get_cosine(vector1, vector2)
                writer.writerow([row[0], title, url, img, cosine])
            except:
                pass

def calculate_similarity_for_all_images(name='politifact_real'):
    with open(f'similarities_{name}.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter='~')
        writer.writerow(['id', 'title', 'url', 'img', 'similarity'])
    data = pd.read_csv(f'datasets/{name}.csv')
    data = data.sample(frac=1)[:250]
    for row in tqdm(data.values):
        calc_sim(row, "politifact_images", name)


if __name__ == "__main__":
    calculate_similarity_for_all_images()