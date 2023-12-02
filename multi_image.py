#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# In[2]:


# # 1. Importing modules

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score

from PIL import Image

# Import nn module for building stacked layers and optimizers
import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, models, transforms
from torch.nn import functional as F
# Import modules for dataset configuration and loading
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from transformers import AdamW, DistilBertModel

import os
from tqdm.notebook import tqdm
from collections import defaultdict
from textwrap import wrap
from PIL import Image, ImageFile, UnidentifiedImageError

# In[ ]:


import warnings

warnings.filterwarnings("ignore")
import pdb

# # Reading data

# ### Paths to the datasets

# In[11]:


path_images = 'politifact/images'

gossipcop_fake = pd.read_csv('datasets/gossipcop_fake.csv')
gossipcop_real = pd.read_csv('datasets/gossipcop_real.csv')
politifact_fake = pd.read_csv('datasets/politifact_fake.csv')
politifact_real = pd.read_csv('datasets/politifact_real.csv')

# In[13]:


politifact_fake.info()

# In[14]:

politifact_real = politifact_real.dropna(subset=['news_url'])
politifact_fake = politifact_fake.dropna(subset=['news_url'])

politifact_fake.info()

# In[17]:

politifact_real['label'] = [0] * len(politifact_real)
politifact_fake['label'] = [1] * len(politifact_fake)

# In[18]:


# Import os & shutil module
import os
import shutil

# Path of source directory & destination directory
src_directory = '/home/elvin/elvin/UT Courses/THESIS/SOURCE CODES/MIMCAN/politifact/real'
dst_directory = '/home/elvin/elvin/UT Courses/THESIS/SOURCE CODES/MIMCAN/politifact/images'

# Extract file from Source directory and copy to Destination directory
for file in os.listdir(src_directory):
    src_file = os.path.join(src_directory, file)
    dest_file = os.path.join(dst_directory, file)
    try:
        shutil.copytree(src_file, dest_file)
    except:
        pass

# In[46]:


data = pd.concat([politifact_real, politifact_fake])
data = data.sample(frac=1)

# In[20]:


from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(data,
                                     random_state=104,
                                     test_size=0.30,
                                     shuffle=True)

df_val, df_test = train_test_split(df_test,
                                     random_state=104,
                                     test_size=0.40,
                                     shuffle=True)

# In[ ]:
# Checking size of training split dataframe
len(df_train), len(df_test), len(df_val)

# In[53]:


# Fake News subtypes in order of Fakeddit benchmark dataset labeling
CLASS_NAMES = ["Real", "Fake"]

# Importing needed modules for DistilBert model (using deep learning language models to perform tokenization)
from transformers import DistilBertTokenizer

title_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# ### Assigning device to train the model

# In[21]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if str(device) == "cpu":
    print("CPU is allocated.")
else:
    print("GPU is allocated.")


# # Dataset Classes

# In[22]:


class DatasetClass(Dataset):
    # Constructor initialized with relevant attributes plus tokenizer information
    def __init__(self, post_id, title, label, title_tokenizer, max_len):
        self.post_id = post_id
        self.title = title
        self.label = label
        self.title_tokenizer = title_tokenizer
        self.max_length = max_len

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        post_id = self.post_id[idx]
        title = self.title[idx]
        label = self.label[idx]

        sample = {
            "post_id": post_id,
            "clean_title": title,
            "label": label
        }

        # Return sample dictionary containing all needed attributes
        return sample


# # Data Augmentation

# In[23]:


# Transform function for image processing (training)
# Performing data augmentation by random resizing, cropping
# and flipping images in order to artificially create new
# image data per training epoch
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.255]
    )
])

# In[ ]:


# Just normalization for validation
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.255]
    )
])


# In[25]:


def collate_batch(batch):
    # List to save processed batch samples
    batch_processed = []
    # Iteration over input batch of size
    for i in range(len(batch)):
        post_id = batch[i]["post_id"]
        title = batch[i]["clean_title"]
        label = batch[i]["label"]

        encoding = title_tokenizer.encode_plus(
            title,
            max_length=80,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        images = []
        try:
            for ind in range(1, 3):
                image = None
                try:
                    f = os.path.join(f"{path_images}/{post_id}/{ind}.png")
                    if os.path.isfile(f) and f.endswith('png'):
                        image = Image.open(f)
                        image = image.convert("RGB")
                        image = train_transform(image)
                        image = torch.unsqueeze(image, 0)
                        image = image.flatten()
                    else:
                        raise Exception
                # Handling FileNotFoundError and randomly initializing pixels
                except:
                    image = torch.rand(3, 224, 224)
                    image = torch.unsqueeze(image, 0)
                finally:
                    images.append(image)
        except:
            pass

        sample = {
            "post_id": post_id,
            "title": title,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "image1": images[0],
            "image2": images[1],
            "label": torch.tensor(label, dtype=torch.long)
        }

        batch_processed.append(sample)

    post_id = []
    titles = []
    for i in range(len(batch_processed)):
        if i == 0:
            post_id.append(batch_processed[i]["post_id"])
            titles.append(batch_processed[i]["title"])
            input_ids_tensor = batch_processed[i]["input_ids"].reshape(-1, 80)
            attention_mask_tensor = batch_processed[i]["attention_mask"].reshape(-1, 80)
            if batch_processed[i]["image1"] is not None:
                image1_tensor = batch_processed[i]["image1"].reshape(-1, 3, 224, 224)

            if batch_processed[i]["image2"] is not None:
                image2_tensor = batch_processed[i]["image2"].reshape(-1, 3, 224, 224)

            label_tensor = batch_processed[i]["label"].reshape(-1, )
            continue

        # Stack attributes of sample dictionary keys to generate correct tensor shape
        post_id.append(batch_processed[i]["post_id"])
        titles.append(batch_processed[i]["title"])
        input_ids_tensor = torch.cat((input_ids_tensor, batch_processed[i]["input_ids"].reshape(-1, 80)))
        attention_mask_tensor = torch.cat((attention_mask_tensor, batch_processed[i]["attention_mask"].reshape(-1, 80)))
        if batch_processed[i]["image1"] is not None:
            image1_tensor = torch.cat((image1_tensor, batch_processed[i]["image1"].reshape(-1, 3, 224, 224)))

        if batch_processed[i]["image2"] is not None:
            image2_tensor = torch.cat((image2_tensor, batch_processed[i]["image2"].reshape(-1, 3, 224, 224)))

        label_tensor = torch.cat((label_tensor, batch_processed[i]["label"].reshape(-1, )))

    # Returning batch list of sample dictionaries containing 16 processed samples
    return {
        "post_id": post_id,
        "title": titles,
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask_tensor,
        "image1": image1_tensor,
        "image2": image2_tensor,
        "label": label_tensor
    }


# In[26]:


def collate_batch_val(batch):
    # List to save processed batch samples
    batch_processed = []
    for i in range(len(batch)):

        # Iteration over input batch of size 16
        post_id = batch[i]["post_id"]
        title = batch[i]["clean_title"]
        label = batch[i]["label"]

        encoding = title_tokenizer.encode_plus(
            title,
            max_length=80,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
        )

        images = []
        try:
            for ind in range(1, 3):
                image = None
                try:
                    f = os.path.join(f"{path_images}/{post_id}/{ind}.png")
                    if os.path.isfile(f) and f.endswith('png'):
                        image = Image.open(f)
                        image = image.convert("RGB")
                        image = train_transform(image)
                        image = torch.unsqueeze(image, 0)
                    else:
                        raise Exception
                # Handling FileNotFoundError and randomly initializing pixels

                except:
                    image = torch.rand(3, 224, 224)
                    image = torch.unsqueeze(image, 0)
                finally:
                    images.append(image.flatten())
        except:
            pass

        sample = {
            "post_id": post_id,
            "title": title,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "image1": images[0],
            "image2": images[1],
            "label": torch.tensor(label, dtype=torch.long)
        }

        batch_processed.append(sample)

    post_id = []
    titles = []
    for i in range(len(batch_processed)):
        if i == 0:
            post_id.append(batch_processed[i]["post_id"])
            titles.append(batch_processed[i]["title"])
            input_ids_tensor = batch_processed[i]["input_ids"].reshape(-1, 80)
            attention_mask_tensor = batch_processed[i]["attention_mask"].reshape(-1, 80)
            image1_tensor = batch_processed[i]["image1"].reshape(-1, 3, 224, 224)
            image2_tensor = batch_processed[i]["image2"].reshape(-1, 3, 224, 224)
            label_tensor = batch_processed[i]["label"].reshape(-1, )
            continue

        # Stack attributes of sample dictionary keys to generate correct tensor shape
        post_id.append(batch_processed[i]["post_id"])
        titles.append(batch_processed[i]["title"])
        input_ids_tensor = torch.cat((input_ids_tensor, batch_processed[i]["input_ids"].reshape(-1, 80)))
        attention_mask_tensor = torch.cat((attention_mask_tensor, batch_processed[i]["attention_mask"].reshape(-1, 80)))
        image1_tensor = torch.cat((image1_tensor, batch_processed[i]["image1"].reshape(-1, 3, 224, 224)))
        image2_tensor = torch.cat((image2_tensor, batch_processed[i]["image2"].reshape(-1, 3, 224, 224)))
        label_tensor = torch.cat((label_tensor, batch_processed[i]["label"].reshape(-1, )))

    # Returning batch list of sample dictionaries containing 16 processed samples
    return {
        "post_id": post_id,
        "title": titles,
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask_tensor,
        "image1": image1_tensor,
        "image2": image2_tensor,
        "label": label_tensor
    }


# # Dataloaders

# ## training dataloader

# In[27]:


def main_train_data_loader(df, title_tokenizer, max_len, batch_size):
    dataset = DatasetClass(
        post_id=df["id"].to_numpy(),
        title=df["title"].to_numpy(),
        label=df["label"].to_numpy(),
        title_tokenizer=title_tokenizer,
        max_len=max_len)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_batch, num_workers=2, pin_memory=True,
                      prefetch_factor=2)


# ## Validation dataloader

# In[28]:


def val_create_data_loader(df, title_tokenizer, max_len, batch_size):
    dataset = DatasetClass(
        post_id=df["id"].to_numpy(),
        title=df["title"].to_numpy(),
        label=df["label"].to_numpy(),
        title_tokenizer=title_tokenizer,
        max_len=max_len)

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_batch_val, num_workers=2, pin_memory=True,
                      prefetch_factor=2)


# In[29]:

BATCH_SIZE = 16
MAX_LEN = 80
train_data_loader = main_train_data_loader(df_train, title_tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = val_create_data_loader(df_val, title_tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = val_create_data_loader(df_test, title_tokenizer, MAX_LEN, BATCH_SIZE)

# In[ ]:


class MultimodalModel(nn.Module):
    def __init__(self, num_classes):
        super(MultimodalModel, self).__init__()
        self.title_module = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.image_module = models.resnet34(pretrained="imagenet")
        self.drop = nn.Dropout(p=0.3)
        self.fc_title = nn.Linear(in_features=self.title_module.config.hidden_size, out_features=num_classes, bias=True)
        # Reshaping image feature tensor (1, 1000) --> (1, 2)
        self.fc_image = nn.Linear(in_features=1000, out_features=num_classes, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, title_input_ids, title_attention_mask, image1, image2):
        # Returning title feature tensor of shape (768,)
        title_last_hidden_states = self.title_module(
            input_ids=title_input_ids,
            attention_mask=title_attention_mask,
            return_dict=False
        )
        # reconfigure model, decrease drop layer.
        title_pooled_output = title_last_hidden_states[0][:, 0, :]
        title_pooled_output = self.drop(title_pooled_output)
        title_output = self.fc_title(title_pooled_output)

        image1_output = self.image_module(image1)
        image1_output = self.drop(image1_output)
        image1_output = self.fc_image(image1_output)

        image2_output = self.image_module(image2)
        image2_output = self.drop(image2_output)
        image2_output = self.fc_image(image2_output)

        image_output = torch.maximum(image1_output, image2_output)
        # Fusion of textual and visual feature tensors to multi-modal feature tensor(Use max fusion strategy)
        fusion = torch.maximum(title_output, image_output)

        return self.softmax(fusion)


# ### Model Training Configuration

# In[30]:


def get_class_weights(dataframe):
    # Count labels per class / subtype of Fake News in training set split
    # in sorted order 0, 1 and put into label_count list
    label_count = [dataframe["label"].value_counts().sort_index(0)[0],
                   dataframe["label"].value_counts().sort_index(0)[1]]

    # Calculate weights per class by subtracting from 1 label_count per class divided
    # by sum of all label_counts
    class_weights = [1 - (x / sum(label_count)) for x in label_count]

    class_weights = torch.FloatTensor(class_weights)
    return class_weights


# In[31]:


# Calculate class weights on basis of training split dataframe and print weight tensor
class_weights = get_class_weights(df_train)
print(class_weights)

# ### Hyperparameters for training Title and Image for the models.

# In[ ]:


EPOCHS = 10

# Initializing Adam optimizer for trainable parameters with initial learning rate of 3e-5, RAdam
total_steps = len(train_data_loader) * EPOCHS
# Initializing weighted Cross Entropy Loss function and assignment to device
loss_function = nn.CrossEntropyLoss(weight=class_weights).to(device)


# # Training Routine

# In[ ]:


def train_model(model, data_loader, loss_function, optimizer, device, num_examples):
    print("Training model in progress..")
    print("-" * 15)
    model = model.train()
    train_losses = []
    correct_preds = 0
    for data in tqdm(data_loader):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        image1 = data["image1"].to(device)
        image2 = data['image2'].to(device)
        labels = data["label"].to(device)

        outputs = model(
            title_input_ids=input_ids,
            title_attention_mask=attention_mask,
            image1=image1,
            image2=image2
        )

        _, preds = torch.max(outputs, dim=1)
        train_loss = loss_function(outputs, labels)
        correct_preds += torch.sum(preds == labels)
        train_losses.append(train_loss.item())
        train_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    # Return train_acc and train_loss values
    return correct_preds.double() / num_examples, np.mean(train_losses)


# In[ ]:


def evaluate_model(model, data_loader, loss_function, device, num_examples):
    print("validation of the model in progress...")
    print("-" * 15)
    model = model.eval()
    val_losses = []
    correct_preds = 0
    with torch.no_grad():
        for data in tqdm(data_loader):
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            image1 = data["image1"].to(device)
            image2 = data['image2'].to(device)
            labels = data["label"].to(device)

            outputs = model(
                title_input_ids=input_ids,
                title_attention_mask=attention_mask,
                image1=image1,
                image2=image2
            )

            _, preds = torch.max(outputs, dim=1)

            val_loss = loss_function(outputs, labels)
            correct_preds += torch.sum(preds == labels)
            val_losses.append(val_loss.item())
    return correct_preds.double() / num_examples, np.mean(val_losses)


# In[ ]:


model = MultimodalModel(len(CLASS_NAMES))
optimizer = AdamW(model.parameters(), lr=1e-5, correct_bias=False)
model.to(device)

# ## Training the model

# In[ ]:


best_accuracy = 0

# Iteration times the total number of epochs
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("-" * 10)

    train_acc, train_loss = train_model(
        model,
        train_data_loader,
        loss_function,
        optimizer,
        device,
        len(df_train)
    )

    print(f"Train loss {train_loss} | Accuracy {train_acc}")
    print()
    val_acc, val_loss = evaluate_model(
        model,
        val_data_loader,
        loss_function,
        device,
        len(df_val)
    )

    print(f"Val   loss {val_loss} | Accuracy {val_acc}")
    print()

print()
print("Completed Training!")
print("-" * 20)

# ### Plotting the output results

# In[ ]:


# Plotting training and validation accuracy curves across the epochs
# plt.plot(train_acc.cpu(), color="green", label="Training Accuracy")
# plt.plot(val_acc.cpu(), color="red", label="Validation Accuracy")
#
# plt.title("Training History")
# #Defining x- and y-axis labels
# plt.ylabel("Accuracy")
# plt.xlabel("Epochs")
# plt.legend()
# plt.show()

# In[ ]:


# plt.plot(train_loss, color="blue", label="Training Loss")
# plt.plot(val_loss, color="orange", label="Validation Loss")
# plt.title("Training History")
# plt.ylabel("Cross Entropy Loss")
# plt.xlabel("Epochs")
# plt.legend()
# plt.show()


# ## Testing the models 

# In[ ]:


def test_model(model, data_loader, loss_function, device, num_examples):
    print("Testing model in progress...")
    print("-" * 15)
    model.eval()
    test_losses = []
    correct_preds = 0
    predictions = []
    prediction_probs = []
    real_labels = []
    wrong_preds = []
    with torch.no_grad():
        for data in tqdm(data_loader):
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            image1 = data["image1"].to(device)
            image2 = data['image2'].to(device)
            labels = data["label"].to(device)
            post_ids = data['post_id']

            outputs = model(
                title_input_ids=input_ids,
                title_attention_mask=attention_mask,
                image1=image1,
                image2=image2
            )
            _, preds = torch.max(outputs, dim=1)
            test_loss = loss_function(outputs, labels)
            correct_preds += torch.sum(preds == labels)
            test_losses.append(test_loss.item())
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_labels.extend(labels)

            for i in range(len(preds)):
                if preds[i] != labels[i]:
                    wrong_preds.append({post_ids[i]: {"true": labels[i].cpu(), "pred": preds[i].cpu()}})

    test_acc = correct_preds.double() / num_examples
    test_loss = np.mean(test_losses)
    predictions = torch.stack(predictions)
    prediction_probs = torch.stack(prediction_probs)
    real_labels = torch.stack(real_labels)

    # Return test_acc, test_loss, predictions, prediction_probs, real_labels
    return test_acc, test_loss, predictions, prediction_probs, real_labels, wrong_preds


# In[ ]:


# Testing model on test data split and initilaizing test values
test_acc, test_loss, y_preds, y_prediction_probs, y_test, wrong_preds = test_model(
    model,
    test_data_loader,
    loss_function,
    device,
    len(df_test)
)

# In[ ]:

print('---------------------------------------------------------------')
print(f'Wrong predictions:')
print(wrong_preds)
print('---------------------------------------------------------------')
# Printing model test accuracy
print(f"Model testing accuracy for classifier:  {test_acc * 100}%")

# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, \
    roc_auc_score, auc

# In[ ]:


# Plotting classification report
print(classification_report(y_test.cpu(), y_preds.cpu(), target_names=CLASS_NAMES))

# In[ ]:


# or individual evaluations of the results
print("f1 score is: ", f1_score(y_test.cpu(), y_preds.cpu(), average='macro'))
print("precision score is: ", precision_score(y_test.cpu(), y_preds.cpu(), average='macro'))
print("recall score is: ", recall_score(y_test.cpu(), y_preds.cpu(), average='macro'))

# In[ ]:


# print("auc score for the model: ",
#       roc_auc_score(y_test.cpu(), 1 - y_prediction_probs.cpu()[:, 1], multi_class="ovo", average="macro"))


# In[ ]:


def plot_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Purples")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha="right")
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha="right")
    # Set x- and y-axis labels
    plt.ylabel("Fakeddit Dataset Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix for the model")
    plt.tight_layout
    plt.show()


# Initialize confusion_matrix with y_test (ground truth labels) and predicted labels
cm = confusion_matrix(y_test.cpu(), y_preds.cpu())
df_cm = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
plot_confusion_matrix(df_cm)


# ## Plotting AUC and ROC score for the models
# 

# In[ ]:


# create ROC curve
def plot_auc(y_test, y_pred_proba):
    fpr, tpr, _ = metrics.roc_curve(y_test, 1 - y_pred_proba[:, 1])
    plt.plot(fpr, tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title("ROC curve of the model")
    plt.tight_layout
    plt.show()


# In[ ]:


# plot_auc(y_test.cpu(), y_prediction_probs.cpu())
