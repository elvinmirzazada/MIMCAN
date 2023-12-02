#!/usr/bin/env python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from custom_fusion import CrossAttention
import torch
from torch import nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from transformers import AdamW, DistilBertModel
from dataset_class import DatasetClass
from transformers import DistilBertTokenizer
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from PIL import Image
import read_data
import os
import warnings
warnings.filterwarnings("ignore")

read_data.path_images = 'gossipcop_images'
data = read_data.load_dataset_gossipcop()
df_train, df_test = train_test_split(data,
                                     random_state=104,
                                     test_size=0.15,
                                     shuffle=True)

print(f'Training data size: {len(df_train)}, Test data size: {len(df_test)}')
CLASS_NAMES = ["Real", "Fake"]

# Importing needed modules for DistilBert model (using deep learning language models to perform tokenization)
TITLE_TOKENIZER = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
MAX_LEN = 80
IMG_SIZE = 3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if str(device) == "cpu":
    print("CPU is allocated.")
else:
    print("GPU is allocated.")

# Transform function for image processing (training)
# Performing data augmentation by random resizing, cropping
# and flipping images in order to artificially create new
# image data per training epoch
train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
    transforms.ColorJitter(),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Just normalization for validation
val_transform = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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
            for file_name in os.listdir(f"{read_data.path_images}/{post_id}"):
                if len(images) >= IMG_SIZE:
                    break
                image = None
                try:
                    f = os.path.join(f"{read_data.path_images}/{post_id}", file_name)
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
            for file_name in os.listdir(f"{read_data.path_images}/{post_id}"):
                if len(images) >= IMG_SIZE:
                    break
                image = None
                try:
                    f = os.path.join(f"{read_data.path_images}/{post_id}", file_name)
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


def main_train_data_loader(df, title_tokenizer, max_len, batch_size):
    dataset = DatasetClass(
        post_id=df["id"].to_numpy(),
        title=df["title"].to_numpy(),
        label=df["label"].to_numpy(),
        title_tokenizer=title_tokenizer,
        max_len=max_len)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_batch, num_workers=2, pin_memory=True,
                      prefetch_factor=2)


def val_create_data_loader(df, title_tokenizer, max_len, batch_size):
    dataset = DatasetClass(
        post_id=df["id"].to_numpy(),
        title=df["title"].to_numpy(),
        label=df["label"].to_numpy(),
        title_tokenizer=title_tokenizer,
        max_len=max_len)

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_batch_val, num_workers=2, pin_memory=True,
                      prefetch_factor=2)


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
BATCH_SIZE = 16
train_data_loader = main_train_data_loader(df_train, TITLE_TOKENIZER, MAX_LEN, BATCH_SIZE)
test_data_loader = val_create_data_loader(df_test, TITLE_TOKENIZER, MAX_LEN, BATCH_SIZE)


class MultimodalModel(nn.Module):
    def __init__(self, num_classes, hidden_size=2, num_heads=2):
        super(MultimodalModel, self).__init__()
        self.title_module = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.image_module = models.resnet34(pretrained="imagenet")
        self.drop = nn.Dropout(p=0.3)
        self.fc_title = nn.Linear(in_features=self.title_module.config.hidden_size, out_features=num_classes, bias=True)
        # Reshaping image feature tensor (1, 1000) --> (1, 2)
        self.cross_attention = CrossAttention(hidden_size, num_heads)
        self.fc_image = nn.Linear(in_features=1000, out_features=num_classes, bias=True)
        self.softmax = nn.Softmax(dim=1)

        # Linear layer for final classification
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, title_input_ids, title_attention_mask, images):
        # Returning title feature tensor of shape (768,)
        title_last_hidden_states = self.title_module(
            input_ids=title_input_ids,
            attention_mask=title_attention_mask,
            return_dict=False
        )
        # reconfigure model, decrease drop layer.
        title_pooled_output = title_last_hidden_states[0][:, 0, :]
        title_output = self.fc_title(self.drop(title_pooled_output))

        images_outputs = []
        for img in images:
            img = torch.unsqueeze(img, 0)
            images_outputs.append(self.fc_image(self.drop(self.image_module(img))))

        attention_outs = []
        for io in images_outputs:
            title_output = title_output.to(device)
            image_output = io.to(device)
            attention_outs.append(self.cross_attention(image_output, title_output))

        attention_outs = torch.cat(attention_outs, dim=1)

        # Fusion of textual and visual feature tensors to multi-modal feature tensor(Use max fusion strategy)
        fusion = torch.maximum(title_output, attention_outs)
        classifier = self.classifier(fusion)
        return self.softmax(classifier)


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

EPOCHS = 5
# Initializing Adam optimizer for trainable parameters with initial learning rate of 3e-5, RAdam
total_steps = len(train_data_loader) * EPOCHS
# Initializing weighted Cross Entropy Loss function and assignment to device
loss_function = nn.CrossEntropyLoss(weight=class_weights).to(device)


def train_model(custom_model, data_loader, loss, opt, dev, num_examples):
    print("Training model in progress..")
    print("-" * 100)
    custom_model = custom_model.train()
    train_losses = []
    correct_preds = 0
    for elm in tqdm(data_loader):
        input_ids = elm["input_ids"].to(dev)
        attention_mask = elm["attention_mask"].to(dev)
        images = elm["image"].to(dev)
        labels = elm["label"].to(dev)

        # take IMG_size images
        images = images[:IMG_SIZE]

        opt.zero_grad()
        outputs = custom_model(
            title_input_ids=input_ids,
            title_attention_mask=attention_mask,
            images=images
        )

        _, preds = torch.max(outputs, dim=1)
        tr_loss = loss(outputs, labels)
        correct_preds += torch.sum(preds == labels)
        train_losses.append(tr_loss.item())
        tr_loss.backward()
        nn.utils.clip_grad_norm_(custom_model.parameters(), max_norm=1.0)
        opt.step()

    # Return train_acc and train_loss values
    return correct_preds.double() / num_examples, np.mean(train_losses)


def evaluate_model(custom_model, data_loader, loss, dev, num_examples):
    print("validation of the model in progress...")
    print("-" * 100)
    custom_model = custom_model.eval()
    val_losses = []
    correct_preds = 0
    with torch.no_grad():
        for elm in tqdm(data_loader):
            input_ids = elm["input_ids"].to(dev)
            attention_mask = elm["attention_mask"].to(dev)
            images = elm["image"].to(dev)
            labels = elm["label"].to(dev)

            # take IMG_size images
            images = images[:IMG_SIZE]

            outputs = custom_model(
                title_input_ids=input_ids,
                title_attention_mask=attention_mask,
                images=images
            )

            _, preds = torch.max(outputs, dim=1)

            val_loss = loss(outputs, labels)
            correct_preds += torch.sum(preds == labels)
            val_losses.append(val_loss.item())
    return correct_preds.double() / num_examples, np.mean(val_losses)


model = MultimodalModel(len(CLASS_NAMES))
optimizer = AdamW(model.parameters(), lr=1e-5, correct_bias=False)
model.to(device)
best_accuracy = 0
EPOCHS = 5

# Iteration times the total number of epochs
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("-" * 100)

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

    print(f"Val loss {val_loss} | Accuracy {val_acc}")
    print()

# k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
# scores = []
# for train_idx, val_idx in k_fold.split(df_train):
#
#     train_loader = main_train_data_loader(df_train.iloc[train_idx], title_tokenizer, MAX_LEN, BATCH_SIZE)
#     val_loader = val_create_data_loader(df_train.iloc[val_idx], title_tokenizer, MAX_LEN, BATCH_SIZE)
#
#     model = MultimodalModel(len(CLASS_NAMES))
#     model.to(device)
#     optimizer = AdamW(model.parameters(), lr=1e-6, correct_bias=False)
#     loss_function = nn.CrossEntropyLoss(weight=class_weights).to(device)
#
#     for epoch in range(5):
#         train_acc, train_loss = train_model(
#             model,
#             train_loader,
#             loss_function,
#             optimizer,
#             device,
#             len(df_train.iloc[train_idx])
#         )
#
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         val_acc, val_loss = evaluate_model(
#             model,
#             val_data_loader,
#             loss_function,
#             device,
#             len(df_val)
#         )
#
#         scores.append((val_acc, val_loss))
#         print((val_acc.tolist(), val_loss.tolist()))

# print(scores)
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
            image3 = data["image3"].to(device)
            # image4 = data['image4'].to(device)
            labels = data["label"].to(device)
            post_ids = data['post_id']

            outputs = model(
                title_input_ids=input_ids,
                title_attention_mask=attention_mask,
                image1=image1,
                image2=image2,
                image3=image3,
                # image4=image4
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
