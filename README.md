
一款通过`ml-1m`数据集设计的简单精巧双塔模型推荐系统。


## 多头注意力机制

```python
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

```

## 用户塔和素材塔的设计

```python
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
```

