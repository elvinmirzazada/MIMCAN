import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import os


gossipcop_fake = pd.read_csv('datasets/gossipcop_fake.csv')
gossicop_real = pd.read_csv('datasets/gossipcop_real.csv')
politifact_fake = pd.read_csv('datasets/politifact_fake.csv')
politifact_real = pd.read_csv('datasets/politifact_real.csv')



df = gossipcop_fake.copy()
# df.head()
# df = df.iloc[:10]

df = df.dropna(subset=['news_url'])


def format_url(url):
    if url.startswith('http'):
        return url
    else:
        return f'http://{url}'

def convert_datatype_str(df, col):
    df = df.copy()
    df[col] = df[col].astype(str)
    return df


df = convert_datatype_str(df, 'news_url')
# convert the titles url also to string
df = convert_datatype_str(df, 'title')



df['news_url'] = df['news_url'].apply(format_url)



def image_statistics(url):
    """
    going to the url for the news:
    - extracting from the articles the images
    """
    try:
        response = requests.get(url, timeout=10)
        html = response.content
        soup = BeautifulSoup(html, 'html.parser')
        # Find the main article content
        article = soup.find('div', class_='article-text')
        
        #print(images)
        list_image_urls = []
        if article != None:
            for i, img in enumerate(article.find_all('img')):
                if 'data-src' in img.attrs:  
                    image_url = img['data-src']
                else:
                    #print('Does not exist')
                    image_url = ""
        
                list_image_urls.append(image_url)
                # removing urls not starting with http since we might not be able to download the file and most of them are logos and svg image
            downloadeble_images = [x for x in list_image_urls if x.startswith('http')]
            image_list = [file for file in downloadeble_images if file.endswith(('.jpg', '.png', 'jpeg'))]  
        else:
            image_list = []
        
    except requests.exceptions.RequestException as e:
        image_list = []
        print(e)
    return image_list
    


def number_images(col):
    return len(col)



df['image_urls'] = df['news_url'].apply(image_statistics)
df['number_images'] = df['image_urls'].apply(number_images)
# function to download only a single image for each post id

# Loop through the rows of the dataframe
# for index, row in df.iterrows():
#     # Get the list of image URLs and ID for this row
#     urls = row['image_urls']
#     ids = row['id']
#     print(ids)
#     if len(urls) > 0:
#         print(len(urls))
#         try:
#             response = requests.get(urls[0])
#             # Define the folder where you want to store the images
#             folder_path = 'single_images/'
#
#             # Make sure the folder exists, or create it if it doesn't
#             if not os.path.exists(folder_path):
#                 os.makedirs(folder_path)
#             # Construct the filename for the image
#             filename = f"{folder_path}{ids}.jpg"
#             print(filename)
#             with open(filename, 'wb') as f:
#                 f.write(response.content)
#         except requests.exceptions.RequestException as e:
#             pass
#             print(e)
        

# Function to download the first few images from the dataframe




# Loop through the rows of the dataframe (Downloading the first 10 images from the urls)
for index, row in df.iterrows():
    # Get the list of image URLs and ID for this row
    urls = row['image_urls']
    ids = row['id']
    print(ids)
    if len(urls) > 0:
        urls = urls[:10]
        print(len(urls))
        # Define the folder where you want to store the images
        folder_path = f'images/{ids}'

        # Make sure the folder exists, or create it if it doesn't
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for url in urls:
            try:
                response = requests.get(url, timeout=10)
           
                # Construct the filename for the image
                filename = os.path.join(folder_path, os.path.basename(url))
                print(filename)
                with open(filename, 'wb') as f:
                    f.write(response.content)
            except requests.exceptions.RequestException as e:
                pass
                print(e)
        



