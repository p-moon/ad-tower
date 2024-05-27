import requests
import os
import zipfile
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

ratings = pd.read_csv(os.path.join(dataset_path, "ratings.dat"), sep='::', engine='python', names=['user_id', 'movie_id', 'rating', 'timestamp'])
ratings.head()

# 创建用户和电影ID的映射
original_user_ids = ratings['user_id'].unique()
original_movie_ids = ratings['movie_id'].unique()

user_id_map = {id: idx for idx, id in enumerate(original_user_ids)}
movie_id_map = {id: idx for idx, id in enumerate(original_movie_ids)}

# 应用映射
ratings['user_id'] = ratings['user_id'].map(user_id_map)
ratings['movie_id'] = ratings['movie_id'].map(movie_id_map)

# 将数据集分为训练集和测试集
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

class UserTower(nn.Module):
    def __init__(self, user_num, item_num, emb_size, hidden_size):
        super(UserTower, self).__init__()
        self.user_emb = nn.Embedding(user_num, emb_size)
        self.item_emb = nn.Embedding(item_num, emb_size)
        self.fc1 = nn.Linear(emb_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, emb_size)

    def forward(self, user_id, item_ids):
        user_emb = self.user_emb(user_id)
        item_embs = self.item_emb(item_ids)
        # 使用self-attention计算用户最近浏览的素材的权重
        attention_weights = F.softmax(torch.matmul(item_embs, user_emb.unsqueeze(2)), dim=1)
        user_rep = torch.sum(attention_weights * item_embs, dim=1)
        # 将用户表示和用户嵌入拼接
        user_rep = torch.cat((user_emb, user_rep), dim=1)
        # 通过MLP得到最终的 user tower 输出
        user_rep = F.relu(self.fc1(user_rep))
        user_rep = self.fc2(user_rep)
        return user_rep

class ItemTower(nn.Module):
    def __init__(self, item_num, emb_size, hidden_size):
        super(ItemTower, self).__init__()
        self.item_emb = nn.Embedding(item_num, emb_size)
        self.fc1 = nn.Linear(emb_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, emb_size)

    def forward(self, item_id):
        item_emb = self.item_emb(item_id)
        # 通过MLP得到最终的 item tower 输出
        item_rep = F.relu(self.fc1(item_emb))
        item_rep = self.fc2(item_rep)
        return item_rep

def train_model(user_tower, item_tower, user_id, item_ids, ratings, optimizer):
    user_id = user_id.to(device)
    item_ids = item_ids.to(device)
    ratings = ratings.to(device)
    user_tower.train()
    item_tower.train()
    optimizer.zero_grad()
    user_rep = user_tower(user_id, item_ids)
    item_rep = item_tower(item_ids)
    # 计算预测评分
    pred_scores = torch.sum(user_rep * item_rep, dim=1)
    loss = F.mse_loss(pred_scores, ratings)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_model(user_tower, item_tower, data_loader):
    user_tower.eval()
    item_tower.eval()
    total_loss = 0
    total_samples = 0
    with tqdm(total=len(data_loader), desc="evaluate progress:") as pbar:
        with torch.no_grad():
            for batch in data_loader:
                user_id, item_ids, ratings = batch
                user_id = user_id.to(device)
                item_ids = item_ids.to(device)
                ratings = ratings.to(device)
                user_rep = user_tower(user_id, item_ids)
                item_rep = item_tower(item_ids)
                # 计算预测评分
                pred_scores = torch.sum(user_rep * item_rep, dim=1)
                loss = F.mse_loss(pred_scores, ratings, reduction='sum')
                total_loss += loss.item()
                total_samples += ratings.size(0)
                pbar.update()
    rmse = np.sqrt(total_loss / total_samples)
    return rmse

user_num = len(user_id_map)
item_num = len(movie_id_map)

# 创建TensorDataset时，将用户ID和电影ID转换为浮点数
train_dataset = TensorDataset(torch.tensor(train_data['user_id'].values, dtype=torch.long), 
                              torch.tensor(train_data['movie_id'].values, dtype=torch.long), 
                              torch.tensor(train_data['rating'].values, dtype=torch.float))  # 将rating转换为torch.float

test_dataset = TensorDataset(torch.tensor(test_data['user_id'].values, dtype=torch.long), 
                             torch.tensor(test_data['movie_id'].values, dtype=torch.long), 
                             torch.tensor(test_data['rating'].values, dtype=torch.float))  # 将rating转换为torch.float

emb_size = 64
hidden_size = 128
batch_size = 16  # 减小 batch size 以减少显存使用
epochs = 10
learning_rate = 0.001
user_tower = UserTower(user_num, item_num, emb_size, hidden_size)
item_tower = ItemTower(item_num, emb_size, hidden_size)

user_tower.to(device)
item_tower.to(device)

optimizer = torch.optim.Adam(list(user_tower.parameters()) + list(item_tower.parameters()), lr=learning_rate)
# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

with tqdm(total=epochs * len(train_loader), desc="train progress:") as pbar:
    for epoch in range(epochs):
        for batch in train_loader:
            user_ids, item_ids, ratings = batch
            loss = train_model(user_tower, item_tower, user_ids, item_ids, ratings, optimizer)
            pbar.set_postfix({'Loss': loss, 'LR': optimizer.param_groups[0]['lr']})
            pbar.update()

# 使用批处理的方式进行评估
rmse = evaluate_model(user_tower, item_tower, test_loader)
print(f'Test RMSE: {rmse}')
