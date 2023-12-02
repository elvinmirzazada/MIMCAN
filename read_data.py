# Created by elvinmirzazada at 23:20 16/04/2023 using PyCharm
import pandas as pd
import os
import shutil

path_images = 'gossipcop_images'


def exclude_non_images_news(row):
    try:
        images = os.listdir(f"{path_images}/{row['id']}")
        return len(images) > 0
    except:
        return False


def prepare_dataset(fake, real):
    fake = fake.dropna(subset=['news_url'])
    real = real.dropna(subset=['news_url'])
    print('The dataset after dropping not news url containing rows')
    print('Fake dataset info: ')
    print(fake.info())
    print('Real dataset info: ')
    print(real.info())

    real['labels'] = [0] * len(real)
    fake['labels'] = [1] * len(fake)
    final_data = pd.concat([real, fake], ignore_index=True)
    final_data = final_data.sample(frac=1)
    print('Final dataset description:')
    print(final_data.info())

    final_data = final_data[final_data.apply(exclude_non_images_news, axis=1)]

    return final_data


def load_dataset_gossipcop():
    # gossipcop_fake = pd.read_csv('datasets/gossipcop_fake.csv')
    # gossipcop_real = pd.read_csv('datasets/gossipcop_real.csv')
    gossipcop_fake = pd.read_csv('datasets/gossipcop_fake.csv')
    gossipcop_real = pd.read_csv('datasets/gossipcop_real.csv')

    data = prepare_dataset(gossipcop_fake, gossipcop_real)

    return data