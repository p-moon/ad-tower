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
import wandb

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

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert emb_size % num_heads == 0, "Embedding size must be divisible by number of heads"
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads

        self.query = nn.Linear(emb_size, emb_size)
        self.key = nn.Linear(emb_size, emb_size)
        self.value = nn.Linear(emb_size, emb_size)
        self.fc_out = nn.Linear(emb_size, emb_size)

    def forward(self, user_emb, item_embs):
        batch_size = user_emb.shape[0]

        # Linear projections
        queries = self.query(user_emb).view(batch_size, self.num_heads, self.head_dim)
        keys = self.key(item_embs).view(batch_size, -1, self.num_heads, self.head_dim)
        values = self.value(item_embs).view(batch_size, -1, self.num_heads, self.head_dim)

        # Scaled Dot-Product Attention
        energy = torch.einsum("bhq,bnhk->bhqn", [queries, keys])
        attention = torch.softmax(energy / (self.head_dim ** (1/2)), dim=3)

        out = torch.einsum("bhqn,bnhk->bhqk", [attention, values]).reshape(batch_size, -1, self.emb_size)
        out = self.fc_out(out)

        # Sum over the sequence dimension to get the final user representation
        user_rep = torch.sum(out, dim=1)
        return user_rep

class UserTower(nn.Module):
    def __init__(self, user_num, item_num, emb_size, hidden_size, num_heads):
        super(UserTower, self).__init__()
        self.user_emb = nn.Embedding(user_num, emb_size)
        self.item_emb = nn.Embedding(item_num, emb_size)
        self.self_attention = MultiHeadSelfAttention(emb_size, num_heads)
        self.fc1 = nn.Linear(emb_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, emb_size)

    def forward(self, user_id, item_ids):
        user_emb = self.user_emb(user_id)  # (batch_size, emb_size)
        item_embs = self.item_emb(item_ids)  # (batch_size, seq_len, emb_size)
        user_rep = self.self_attention(user_emb, item_embs)  # (batch_size, emb_size)
        # 将用户表示和用户嵌入拼接
        user_rep = torch.cat((user_emb, user_rep), dim=1)  # (batch_size, emb_size * 2)
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
    all_predictions = []
    all_ratings = []
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
            all_predictions.extend(pred_scores.cpu().numpy())
            all_ratings.extend(ratings.cpu().numpy())
            loss = F.mse_loss(pred_scores, ratings, reduction='sum')
            total_loss += loss.item()
            total_samples += ratings.size(0)
    rmse = np.sqrt(total_loss / total_samples)

    # 创建评估报告
    data = [[pred, true] for pred, true in zip(all_predictions, all_ratings)]
    table = wandb.Table(data=data, columns=["Predicted", "Actual"])
    wandb.log({"Evaluation Results": table, "Test RMSE": rmse})

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
num_heads = 8
batch_size = 16  # 减小 batch size 以减少显存使用
epochs = 10
learning_rate = 0.001

# 初始化wandb
wandb.init(project="user-item-tower", config={
    "emb_size": emb_size,
    "hidden_size": hidden_size,
    "num_heads": num_heads,
    "batch_size": batch_size,
    "epochs": epochs,
    "learning_rate": learning_rate
})

user_tower = UserTower(user_num, item_num, emb_size, hidden_size, num_heads)
item_tower = ItemTower(item_num, emb_size, hidden_size)

user_tower.to(device)
item_tower.to(device)

optimizer = torch.optim.Adam(list(user_tower.parameters()) + list(item_tower.parameters()), lr=learning_rate)
# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

with tqdm(total=epochs * len(train_loader)) as pbar:
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_loader:
            user_ids, item_ids, ratings = batch
            loss = train_model(user_tower, item_tower, user_ids, item_ids, ratings, optimizer)
            epoch_loss += loss
            pbar.set_postfix({'Loss': loss, 'LR': optimizer.param_groups[0]['lr']})
            wandb.log({"train Loss": loss})
            wandb.log({"learn rate": optimizer.param_groups[0]['lr']})
            pbar.update()
        
        wandb.log({"Train epoch Loss": epoch_loss / len(train_loader)})

        # 在每个epoch结束时评估模型
        rmse = evaluate_model(user_tower, item_tower, test_loader)
        wandb.log({"Test RMSE": rmse})
        print(f'Epoch {epoch + 1}/{epochs}, Test RMSE: {rmse}')

# 结束wandb
wandb.finish()


if not os.path.exists("./models"):
    os.mkdir("models")

torch.save({
    'user_tower_state_dict': user_tower.state_dict(),
    'item_tower_state_dict': item_tower.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'train_loss': epoch_loss,
    'test_rmse': rmse
}, './models/model_checkpoint.pth')

# 加载模型
checkpoint = torch.load('./models/model_checkpoint.pth')
user_tower.load_state_dict(checkpoint['user_tower_state_dict'])
item_tower.load_state_dict(checkpoint['item_tower_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
train_loss = checkpoint['train_loss']
test_rmse = checkpoint['test_rmse']

# 设置模型为推理模式
user_tower.eval()
item_tower.eval()
