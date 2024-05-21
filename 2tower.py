import requests
import os
import zipfile
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import torch
import torch.nn as nn
import torch.nn.functional as F


# 下载和解压数据集
if not os.path.exists("./dataset"):
    os.mkdir("dataset")
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    zip_content = requests.get(url=url).content
    file_path = "./dataset/ml-1m.zip"
    with open(file_path, "wb") as f:
        f.write(zip_content)

    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall("./dataset")


dataset_path = "./dataset/ml-1m"

import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Transform input using linear layers
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        # Reshape for multi-head attention
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # Einsum does matrix multiplication for query*keys for each training example
        # with every other key
        attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(attention / (self.head_dim ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

# Testing the modified SelfAttention class
N = 2  # batch size
seq_length = 3  # sequence length
embed_size = 8  # embedding size

values = torch.randn(N, seq_length, embed_size)
keys = torch.randn(N, seq_length, embed_size)
query = torch.randn(N, seq_length, embed_size)

attention_modified = SelfAttention(embed_size, 4)
output_modified = attention_modified(values, keys, query, None)

print(output_modified.shape)


class Tower(nn.Module):
    def __init__(self, embed_size, heads, num_layers, forward_expansion, max_length, dropout, device):
        super(Tower, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.layers = nn.ModuleList(
            [
                SelfAttention(embed_size, heads)
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, x, x, mask)

        x = self.dropout(x)
        x = self.fc_out(x)
        return x

class DualTowerModel(nn.Module):
    def __init__(
        self,
        user_embed_size,
        item_embed_size,
        heads,
        num_layers,
        forward_expansion,
        max_length,
        dropout,
        device,
    ):
        super(DualTowerModel, self).__init__()
        self.user_tower = Tower(user_embed_size, heads, num_layers, forward_expansion, max_length, dropout, device)
        self.item_tower = Tower(item_embed_size, heads, num_layers, forward_expansion, max_length, dropout, device)

        self.fc = nn.Linear(user_embed_size + item_embed_size, 1)
        self.device = device

    def forward(self, user_embed, item_embed, user_mask, item_mask):
        user_out = self.user_tower(user_embed, user_mask)
        item_out = self.item_tower(item_embed, item_mask)

        # Concatenate user and item embeddings
        combined = torch.cat((user_out, item_out), dim=1)
        out = torch.sigmoid(self.fc(combined))
        return out

# Set the device to "cuda" if available, otherwise use "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
user_embed_size = 128
item_embed_size = 128
heads = 4
num_layers = 2
forward_expansion = 4
max_length = 100
dropout = 0.1

# Initialize the model
model = DualTowerModel(user_embed_size, item_embed_size, heads, num_layers, forward_expansion, max_length, dropout, device)
model = model.to(device)

model


ratings = pd.read_csv(os.path.join(dataset_path, "ratings.dat"), sep='::', engine='python', names=['user_id', 'movie_id', 'rating', 'timestamp'])


# Define the number of users and movies in the dataset
num_users = ratings['user_id'].nunique()
num_movies = ratings['movie_id'].nunique()

# Define the embedding layers for users and movies
user_embedding = nn.Embedding(num_users + 1, user_embed_size)
movie_embedding = nn.Embedding(num_movies + 1, item_embed_size)
user_embedding.to(device)
movie_embedding.to(device)

# Prepare the dataset for training
def prepare_data(ratings):
    user_ids = ratings['user_id'].values
    movie_ids = ratings['movie_id'].values
    ratings = ratings['rating'].values

    # Convert ratings to binary labels (1 for rating >= 4, 0 otherwise)
    labels = (ratings >= 4).astype(int)

    return user_ids, movie_ids, labels

user_ids, movie_ids, labels = prepare_data(ratings)

# Move the data to the device
user_ids = torch.tensor(user_ids, dtype=torch.long).to(device)
movie_ids = torch.tensor(movie_ids, dtype=torch.long).to(device)
labels = torch.tensor(labels, dtype=torch.float).unsqueeze(1).to(device)

user_ids, movie_ids, labels


# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.to(device=device)
# Define the training loop
def train_model(model, user_ids, movie_ids, labels, num_epochs=5):
    for epoch in range(num_epochs):
        # Set the model to training mode
        model.train()

        # Forward pass
        user_embed = user_embedding(user_ids)
        movie_embed = movie_embedding(movie_ids)
        user_mask = None  # We don't have padding in user embeddings, so no mask needed
        movie_mask = None  # We don't have padding in movie embeddings, so no mask needed

        outputs = model(user_embed, movie_embed, user_mask, movie_mask)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            predictions = torch.round(torch.sigmoid(outputs))
            accuracy = torch.mean((predictions == labels).float())

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")

# Start training
train_model(model, user_ids, movie_ids, labels)
