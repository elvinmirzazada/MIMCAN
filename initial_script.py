#!/usr/bin/env python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from bidirectional_cross_attention import BidirectionalCrossAttention

from PIL import Image

# Import nn module for building stacked layers and optimizers
import torch
import os
from torch import nn, optim
import torchvision
from torchvision import datasets, models, transforms
from torch.nn import functional as F
from custom_fusion import CrossAttention
# Import modules for dataset configuration and loading
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from transformers import AdamW, DistilBertModel, DistilBertTokenizer
from tqdm.notebook import tqdm
from collections import defaultdict
from textwrap import wrap
from PIL import Image, ImageFile, UnidentifiedImageError
import warnings
warnings.filterwarnings("ignore")


politifact_path_images = 'politifact_images'
gossipcop_path_images = 'gossipcop_images'
gossipcop_fake = pd.read_csv('datasets/gossipcop_fake.csv')
gossipcop_real = pd.read_csv('datasets/gossipcop_real.csv')
politifact_fake = pd.read_csv('datasets/politifact_fake.csv')
politifact_real = pd.read_csv('datasets/politifact_real.csv')

path_images = gossipcop_path_images

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

    real['label'] = [0] * len(real)
    fake['label'] = [1] * len(fake)
    final_data = pd.concat([real, fake])
    final_data = final_data.sample(frac=1)
    print('Final dataset description:')
    print(final_data.info())

    final_data = final_data[final_data.apply(exclude_non_images_news, axis=1)]

    return final_data


data = prepare_dataset(gossipcop_fake, gossipcop_real)

# Path of source directory & destination directory
# src_directory = '/home/elvin/elvin/UT Courses/THESIS/SOURCE CODES/MIMCAN/politifact/real'
# dst_directory = '/home/elvin/elvin/UT Courses/THESIS/SOURCE CODES/MIMCAN/politifact/images'
#
# # Extract file from Source directory and copy to Destination directory
# for file in os.listdir(src_directory):
#     src_file = os.path.join(src_directory, file)
#     dest_file = os.path.join(dst_directory, file)
#     try:
#         shutil.copytree(src_file, dest_file)
#     except:
#         pass
#
# # In[46]:
#
#
# data = pd.concat([politifact_real, politifact_fake])
# data = data.sample(frac=1)

df_train, df_test = train_test_split(data[:100],
                                     random_state=104,
                                     test_size=0.25,
                                     shuffle=True)

print(f'Train data size: {len(df_train)}, Test data size: {len(df_test)}')
CLASS_NAMES = ["Real", "Fake"]

# Importing needed modules for DistilBert model (using deep learning language models to perform tokenization)
TITLE_TOKENIZER = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
MAX_LEN = 80

device = "cpu"#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if str(device) == "cpu":
    print("CPU is allocated.")
else:
    print("GPU is allocated.")


# Dataset Classes
class DatasetClass(Dataset):
    # Constructor initialized with relevant attributes plus tokenizer information
    def __init__(self, post_id, title, label, tokenizer, max_len):
        self.post_id = post_id
        self.title = title
        self.label = label
        self.title_tokenizer = tokenizer
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
            "label": label,
            'tokenizer': self.title_tokenizer,
            'max_len': self.max_length
        }

        # Return sample dictionary containing all needed attributes
        return sample


# Data Augmentation
# Transform function for image processing (training)
# Performing data augmentation by random resizing, cropping and flipping images in order to artificially create new
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

IMG_SIZE = 5


def collate_batch(batch):
    # List to save processed batch samples
    batch_processed = []
    # Iteration over input batch of size
    for i in range(len(batch)):
        post_id = batch[i]["post_id"]
        title = batch[i]["clean_title"]
        label = batch[i]["label"]
        title_tokenizer = batch[i]['tokenizer']
        max_length = batch[i]['max_len']

        encoding = title_tokenizer.encode_plus(
            title,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
        )

        images = []
        try:
            for file_name in os.listdir(f"{path_images}/{post_id}"):
                if len(images) >= IMG_SIZE:
                    break
                image = None
                try:
                    f = os.path.join(f"{path_images}/{post_id}", file_name)
                    if os.path.isfile(f) and f.endswith(('.jpg', '.png', 'jpeg')):
                        image = Image.open(f)
                        image = image.convert("RGB")
                        image = train_transform(image)
                        image = torch.unsqueeze(image, 0)
                        images.append(image)
                except:
                    image = torch.rand(3, 224, 224)
                    image = torch.unsqueeze(image, 0)
                    images.append(image)
                finally:
                    if image is None:
                        image = torch.rand(3, 224, 224)
                        image = torch.unsqueeze(image, 0)
                        images.append(image)
        except:
            pass

        while len(images) < IMG_SIZE:
            image = torch.rand(3, 224, 224)
            image = torch.unsqueeze(image, 0)
            images.append(image)

        sample = {
            "post_id": post_id,
            "title": title,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "image": images,
            "label": torch.tensor(label, dtype=torch.long)
        }

        batch_processed.append(sample)

    post_id = []
    titles = []
    input_ids_tensor = None
    attention_mask_tensor = None
    images_tensors = None
    label_tensor = None
    for i in range(len(batch_processed)):
        post_id.append(batch_processed[i]["post_id"])
        titles.append(batch_processed[i]["title"])
        # Concatenate input_ids_tensor and attention_mask_tensor
        if input_ids_tensor is None:
            input_ids_tensor = batch_processed[i]["input_ids"].reshape(-1, MAX_LEN)
            attention_mask_tensor = batch_processed[i]["attention_mask"].reshape(-1, MAX_LEN)
        else:
            input_ids_tensor = torch.cat((input_ids_tensor, batch_processed[i]["input_ids"].reshape(-1, MAX_LEN)))
            attention_mask_tensor = torch.cat(
                (attention_mask_tensor, batch_processed[i]["attention_mask"].reshape(-1, MAX_LEN)))

        # Concatenate images_tensors
        if batch_processed[i]['image']:
            images = [im.reshape(-1, 3, 224,224) for im in batch_processed[i]['image']]
            images = torch.cat(images)
            if images_tensors is None:
                images_tensors = images
            else:
                images_tensors = torch.cat((images_tensors, images))

        # Concatenate label_tensor
        if label_tensor is None:
            label_tensor = batch_processed[i]["label"].reshape(-1, )
        else:
            label_tensor = torch.cat((label_tensor, batch_processed[i]["label"].reshape(-1, )))

    # Returning batch list of sample dictionaries containing 16 processed samples
    return {
        "post_id": post_id,
        "title": titles,
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask_tensor,
        "image": images_tensors,
        "label": label_tensor
    }


def collate_batch_val(batch):
    batch_processed = []
    for i in range(len(batch)):

        # Iteration over input batch of size 16
        post_id = batch[i]["post_id"]
        title = batch[i]["clean_title"]
        label = batch[i]["label"]
        title_tokenizer = batch[i]['tokenizer']
        max_length = batch[i]['max_len']

        encoding = title_tokenizer.encode_plus(
            title,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
        )

        images = []
        try:
            for file_name in os.listdir(f"{path_images}/{post_id}"):
                if len(images) >= IMG_SIZE:
                    break
                image = None
                try:
                    f = os.path.join(f"{path_images}/{post_id}", file_name)
                    if os.path.isfile(f) and f.endswith(('.jpg', '.png', 'jpeg')):
                        image = Image.open(f)
                        image = image.convert("RGB")
                        image = train_transform(image)
                        image = torch.unsqueeze(image, 0)
                        images.append(image)
                except:
                    image = torch.rand(3, 224, 224)
                    image = torch.unsqueeze(image, 0)
                    images.append(image)
                finally:
                    if image is None:
                        image = torch.rand(3, 224, 224)
                        image = torch.unsqueeze(image, 0)
                        images.append(image)
        except:
            pass

        while len(images) < IMG_SIZE:
            image = torch.rand(3, 224, 224)
            image = torch.unsqueeze(image, 0)
            images.append(image)

        sample = {
            "post_id": post_id,
            "title": title,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "image": images,
            "label": torch.tensor(label, dtype=torch.long)
        }

        batch_processed.append(sample)

    post_id = []
    titles = []
    input_ids_tensor = None
    attention_mask_tensor = None
    images_tensors = None
    label_tensor = None
    for i in range(len(batch_processed)):
        post_id.append(batch_processed[i]["post_id"])
        titles.append(batch_processed[i]["title"])
        # Concatenate input_ids_tensor and attention_mask_tensor
        if input_ids_tensor is None:
            input_ids_tensor = batch_processed[i]["input_ids"].reshape(-1, MAX_LEN)
            attention_mask_tensor = batch_processed[i]["attention_mask"].reshape(-1, MAX_LEN)
        else:
            input_ids_tensor = torch.cat((input_ids_tensor, batch_processed[i]["input_ids"].reshape(-1, MAX_LEN)))
            attention_mask_tensor = torch.cat(
                (attention_mask_tensor, batch_processed[i]["attention_mask"].reshape(-1, MAX_LEN)))

        # Concatenate images_tensors
        if batch_processed[i]['image']:
            images = [im.reshape(-1, 3, 224, 224) for im in batch_processed[i]['image']]
            images = torch.cat(images)
            if images_tensors is None:
                images_tensors = images
            else:
                images_tensors = torch.cat((images_tensors, images))

        # Concatenate label_tensor
        if label_tensor is None:
            label_tensor = batch_processed[i]["label"].reshape(-1, )
        else:
            label_tensor = torch.cat((label_tensor, batch_processed[i]["label"].reshape(-1, )))

    # Returning batch list of sample dictionaries containing 16 processed samples
    return {
        "post_id": post_id,
        "title": titles,
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask_tensor,
        "image": images_tensors,
        "label": label_tensor
    }


def main_train_data_loader(df, tokenizer, max_len, batch_size):
    dataset = DatasetClass(
        post_id=df["id"].to_numpy(),
        title=df["title"].to_numpy(),
        label=df["label"].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_batch, num_workers=2, pin_memory=True,
                      prefetch_factor=2)


def val_create_data_loader(df, tokenizer, max_len, batch_size):
    dataset = DatasetClass(
        post_id=df["id"].to_numpy(),
        title=df["title"].to_numpy(),
        label=df["label"].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len)

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_batch_val, num_workers=2, pin_memory=True,
                      prefetch_factor=2)


BATCH_SIZE = 32
train_data_loader = main_train_data_loader(df_train, TITLE_TOKENIZER, MAX_LEN, BATCH_SIZE)
test_data_loader = val_create_data_loader(df_test, TITLE_TOKENIZER, MAX_LEN, BATCH_SIZE)


class MultiModalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.resnet = models.resnet50(pretrained=True)
        self.drop = nn.Dropout(p=0.3)

        # Defining the attention mechanism for the model
        self.image_to_title_attention = nn.MultiheadAttention(self.bert.config.hidden_size,
                                                              num_heads=4)  # Increase num_heads

        self.linear = nn.Linear(1000, self.bert.config.hidden_size)
        self.norm = nn.BatchNorm1d(self.bert.config.hidden_size)
        self.relu = nn.ReLU()  # Add ReLU activation
        self.hidden = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)  # Add hidden layer
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)
        self.softmax = nn.Softmax()

    def forward(self,  title_input_ids, title_attention_mask, images):
        # Process text input
        # pdb.set_trace()
        text_features = self.bert(input_ids=title_input_ids, attention_mask=title_attention_mask)[0][:, 0, :]

        # Process image input
        img_embeddings = [self.resnet(torch.unsqueeze(img, 0)) for img in images]
        img_embeddings = torch.stack(img_embeddings)
        # print(img_embeddings.shape)
        img_embeddings = self.linear(img_embeddings)
        # img_embeddings = self.norm(img_embeddings)  # Apply batch normalization
        # img_embeddings = self.relu(img_embeddings)  # Apply ReLU activation

        # Calculate attention between text and each image
        attention_outputs = []
        for img_emb in img_embeddings:
            img_emb = img_emb.view(1, 1, 768)
            # text_output.unsqueeze(1).shape (1, batch_size, hidden_size) => (1, 2, 768)
            # img_emb.shape => (1, 1, hidden_size)
            att_out, _ = self.image_to_title_attention(text_features.unsqueeze(1), img_emb, img_emb)
            attention_outputs.append(att_out)

        # Average attention outputs
        attention_output = torch.stack(attention_outputs).mean(dim=0)

        # Classifier
        logits = self.hidden(attention_output.squeeze(1))  # Apply hidden layer
        logits = self.drop(logits)  # Apply dropout to the hidden layer
        logits = self.classifier(logits)
        return self.softmax(logits)

# class MultimodalModel(nn.Module):
#     def __init__(self, num_classes, hidden_dim):
#         super(MultimodalModel, self).__init__()
#         # Define image feature extractor (ResNet)
#         self.image_model = models.resnet50(pretrained=True)
#         self.image_feature_dim = 2048
#
#         # Define text feature extractor (DistilBERT)
#
#         self.text_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
#         self.text_feature_dim = self.text_model.config.hidden_size
#         self.image_linear = nn.Linear(1000, 768)
#         self.image_to_title_attention = nn.MultiheadAttention(self.text_model.config.hidden_size, num_heads=4)
#         # Define CrossAttention module
#         # self.cross_attention = BidirectionalCrossAttention(dim=768, heads=8, dim_head=64, context_dim=1000)
#         self.cross_attention = CrossAttention(768)
#         self.drop = nn.Dropout(p=0.3)
#         self.softmax = nn.Softmax(dim=1)
#         self.classifier = nn.Linear(self.text_model.config.hidden_size, 1)
#         self.hidden = nn.Linear(self.text_model.config.hidden_size, self.text_model.config.hidden_size)  # Add hidden layer
#
#     def forward(self, title_input_ids, title_attention_mask, images):
#         # Extract text features
#         text_features = self.text_model(input_ids=title_input_ids, attention_mask=title_attention_mask)[0][:, 0, :]
#
#         # Image module
#         images_outputs = []
#         for img in images:
#             img = torch.unsqueeze(img, 0)
#             img = self.image_model(img)
#             image_feature = self.image_linear(img)
#             images_outputs.append(image_feature)
#
#         attention_outs = []
#         for io in images_outputs:
#             text_output = text_features.to(device)
#             image_output = io.to(device)
#             image_output = image_output.view(1, 1, 768)
#             # output = self.cross_attention(text_output, image_output)
#             att_out, _ = self.image_to_title_attention(text_output.unsqueeze(1), image_output, image_output)
#             # output = torch.cat([O_i, O_t], dim=-1)
#             # output = output.permute(0, 2, 1)
#             attention_outs.append(att_out)
#
#         strong_im_signals = torch.stack(attention_outs).max(dim=0).values
#         # Fusion of textual and visual feature tensors to multi-modal feature tensor(Use max fusion strategy)
#         fusion = self.drop(strong_im_signals)
#         # result = torch.max(torch.softmax(fusion, dim=1))
#         # logits = self.hidden(strong_im_signals.squeeze(1))  # Apply hidden layer
#         # logits = self.drop(logits)  # Apply dropout to the hidden layer
#         # logits = self.classifier(logits)
#         fusion = torch.maximum(text_features, fusion)
#         return self.softmax(fusion)


def get_class_weights(dataframe):
    # Count labels per class / subtype of Fake News in training set split
    # in sorted order 0, 1 and put into label_count list
    label_count = [dataframe["label"].value_counts().sort_index(0)[0],
                   dataframe["label"].value_counts().sort_index(0)[1]]

    # Calculate weights per class by subtracting from 1 label_count per class divided
    # by sum of all label_counts
    cl_weights = [1 - (x / sum(label_count)) for x in label_count]

    cl_weights = torch.FloatTensor(cl_weights)
    return cl_weights


# Calculate class weights on basis of training split dataframe and print weight tensor
class_weights = get_class_weights(df_train)
print(class_weights)
import os

os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
EPOCHS = 5
# Initializing Adam optimizer for trainable parameters with initial learning rate of 3e-5, RAdam
total_steps = len(train_data_loader) * EPOCHS
# Initializing weighted Cross Entropy Loss function and assignment to device
loss_function = nn.CrossEntropyLoss().to(device)


def train_model(custom_model: nn.Module, data_loader, loss, opt, dvc, num_examples):
    print("Training model in progress..")
    print("-" * 15)
    custom_model = custom_model.train()
    train_losses = []
    correct_preds = 0
    for elm in tqdm(data_loader):
        input_ids = elm["input_ids"].to(dvc)
        attention_mask = elm["attention_mask"].to(dvc)
        images = elm["image"].to(dvc)
        labels = elm["label"].to(dvc)

        # take IMG_size images
        # images = images[:10]

        opt.zero_grad()
        outputs = custom_model(
            title_input_ids=input_ids,
            title_attention_mask=attention_mask,
            images=images
        )

        _, preds = torch.max(outputs, dim=1)
        # tr_loss = loss(outputs, labels)
        tr_loss = loss(outputs.squeeze(), labels.float())
        correct_preds += torch.sum(preds == labels)
        train_losses.append(tr_loss.item())
        tr_loss.backward()
        nn.utils.clip_grad_norm_(custom_model.parameters(), max_norm=1.0)
        opt.step()

    # Return train_acc and train_loss values
    return correct_preds.double() / num_examples, np.mean(train_losses)


def evaluate_model(model, data_loader, loss_function, device, num_examples):
    print("validation of the model in progress...")
    print("-" * 100)
    model = model.eval()
    val_losses = []
    correct_preds = 0
    with torch.no_grad():
        for data in tqdm(data_loader):
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            images = data["image"].to(device)
            labels = data["label"].to(device)

            outputs = model(
                title_input_ids=input_ids,
                title_attention_mask=attention_mask,
                images=images
            )

            _, preds = torch.max(outputs, dim=1)

            vl_loss = loss_function(outputs.squeeze(), labels.float())
            correct_preds += torch.sum(preds == labels)
            val_losses.append(vl_loss.item())
    return correct_preds.double() / num_examples, np.mean(val_losses)


model = MultiModalModel()
optimizer = AdamW(model.parameters(), lr=1e-4, correct_bias=False)
model.to(device)
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
        test_data_loader,
        loss_function,
        device,
        len(df_test)
    )

    print(f"Val   loss {val_loss} | Accuracy {val_acc}")
    print()

print()
print("Completed Training!")
print("-" * 20)

# ### Plotting the output results

# In[ ]:


# Plotting training and validation accuracy curves across the epochs
plt.plot(train_acc.cpu(), color="green", label="Training Accuracy")
plt.plot(val_acc.cpu(), color="red", label="Validation Accuracy")

plt.title("Training History")
# Defining x- and y-axis labels
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend()
plt.show()

# In[ ]:


plt.plot(train_loss.cpu(), color="blue", label="Training Loss")
plt.plot(val_loss.cpu(), color="orange", label="Validation Loss")
plt.title("Training History")
plt.ylabel("Cross Entropy Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()


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

    with torch.no_grad():
        for data in tqdm(data_loader):
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            images = data["image"].to(device)
            labels = data["label"].to(device)

            outputs = model(
                title_input_ids=input_ids,
                title_attention_mask=attention_mask,
                image=images
            )
            _, preds = torch.max(outputs, dim=1)
            test_loss = loss_function(outputs.squeeze(), labels.float())
            correct_preds += torch.sum(preds == labels)
            test_losses.append(test_loss.item())
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_labels.extend(labels)
    test_acc = correct_preds.double() / num_examples
    test_loss = np.mean(test_losses)
    predictions = torch.stack(predictions)
    prediction_probs = torch.stack(prediction_probs)
    real_labels = torch.stack(real_labels)

    # Return test_acc, test_loss, predictions, prediction_probs, real_labels
    return test_acc, test_loss, predictions, prediction_probs, real_labels


# In[ ]:


# Testing model on test data split and initilaizing test values
test_acc, test_loss, y_preds, y_prediction_probs, y_test = test_model(
    model,
    test_data_loader,
    loss_function,
    device,
    len(df_test)
)

# In[ ]:


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


print("auc score for the model: ",
      roc_auc_score(y_test.cpu(), 1 - y_prediction_probs.cpu()[:, 1], multi_class="ovo", average="macro"))


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


plot_auc(y_test.cpu(), y_prediction_probs.cpu())
