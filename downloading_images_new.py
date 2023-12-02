import pandas as pd
import numpy as np
import transformers
import requests
from bs4 import BeautifulSoup
import os

# gossicop_fake = pd.read_csv('datasets/gossipcop_fake.csv')
# gossicop_real = pd.read_csv('datasets/gossipcop_real.csv')
politifact_fake = pd.read_csv('datasets/politifact_fake.csv')
# politifact_real = pd.read_csv('datasets/politifact_real.csv')

df = politifact_fake.copy()
df = df.sample(frac=1)
df = df[:5000].copy()
df.head()

df = df.dropna(subset=['news_url'])


def format_url(news_url):
    if news_url.startswith('http') or news_url.startswith('https'):
        return news_url
    else:
        return f'https://{news_url}'


def convert_datatype_str(data, col):
    data = data.copy()
    data[col] = data[col].astype(str)
    return data


df = convert_datatype_str(df, 'news_url')
df = convert_datatype_str(df, 'title')
df['news_url'] = df['news_url'].apply(format_url)


def cosine_similarity(text1, text2):
    from transformers import AutoTokenizer, AutoModel
    import torch
    from scipy.spatial.distance import cosine

    # Load pre-trained BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')

    # Tokenize the texts and convert to tensors
    inputs1 = tokenizer(text1, return_tensors='pt')
    inputs2 = tokenizer(text2, return_tensors='pt')

    # Get the embeddings for the texts
    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

    # Flatten the embeddings into 1-D arrchaays
    embedding1 = outputs1.last_hidden_state.mean(dim=1).flatten().numpy()
    embedding2 = outputs2.last_hidden_state.mean(dim=1).flatten().numpy()

    # Calculate the cosine similarity between the embeddings
    similarity = 1 - cosine(embedding1, embedding2)
    return similarity


def image_statistics(article_url):
    """
    going to the url for the news:
    - extracting image url from the articles
    """
    list_image_urls = set()
    try:
        resp = requests.get(article_url, timeout=10)
        html = resp.content
        soup = BeautifulSoup(html, 'html.parser')
        # Find the main article content
        article = soup.body  # soup.find('div', class_='article-text')
        min_w, min_h = 250, 250
        if article is not None:
            for i, img in enumerate(article.find_all('img')):
                if 'data-src' in img.attrs and img['data-src'].startswith('http') and img['data-src'].endswith(
                        ('.jpg', '.png', 'jpeg')):
                    try:
                        width = int(img.get('width', 0))
                        height = int(img.get('height', 0))
                        if (width < min_w) or (height < min_h):
                            continue
                        image_url = img['data-src']
                    except:
                        continue
                elif 'src' in img.attrs and img['src'].startswith('http') and img['src'].endswith(
                        ('.jpg', '.png', 'jpeg')):
                    try:
                        width = int(img.get('width', '0').replace('px', ''))
                        height = int(img.get('height', '0').replace('px', ''))
                        if (width < min_w) or (height < min_h):
                            continue
                        image_url = img['src']
                    except:
                        continue
                else:
                    continue

                list_image_urls.add(image_url)

    except requests.exceptions.RequestException as ex:
        print(str(ex))
    return list(list_image_urls)


def number_images(col):
    return len(col)


df['image_urls'] = df['news_url'].apply(image_statistics)
df['number_images'] = df['image_urls'].apply(number_images)

df = df[df['image_urls'].apply(lambda x: len(x) > 0)]
# Function to download the first few images from the dataframe

df = df.sample(frac=1)

count = 0

from PIL import ImageFile


def get_image_sizes(res):
    data = res.content
    p = ImageFile.Parser()
    p.feed(data)  ## feed the data to image parser to get photo info from data headers
    if p.image:
        return p.image.size
    return 0, 0


# Loop through the rows of the dataframe (Downloading the first 10 images from the urls)
for index, row in df.iterrows():
    # Get the list of image URLs and ID for this row
    urls = row['image_urls']
    ids = row['id']
    print(ids)
    if len(urls) > 0:
        urls = urls[:20]
        # Define the folder where you want to store the images
        dir_name = row["news_url"].replace(":", ".").replace("/", "_")
        folder_path = f'gossipcop_images/{ids}'

        # Make sure the folder exists, or create it if it doesn't
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for url in urls:
            try:
                response = requests.get(url, timeout=20)
                w, h = get_image_sizes(response)
                # Construct the filename for the image
                filename = os.path.join(folder_path, f"{w}_{h}_" + os.path.basename(url))
                print(filename)
                with open(filename, 'wb') as f:
                    f.write(response.content)
                count += 1
            except Exception as e:
                print(e)
                continue


import sys

sys.exit()
