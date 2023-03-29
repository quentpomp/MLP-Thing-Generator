## I ASKED OPEN AI TO MAKE ITSELF

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.token2id = {token: i for i, token in enumerate(vocab)}
        self.id2token = {i: token for token, i in self.token2id.items()}

    def tokenize(self, text):
        tokens = text.split()
        return [self.token2id[token] for token in tokens]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.pos_encoding = self.generate_encoding(max_seq_len, d_model)

    def forward(self, x):
        pos_encodings = self.pos_encoding[:x.size(1), :].unsqueeze(0)
        return x + pos_encodings.to(x.device)

    def generate_encoding(self, max_seq_len, d_model):
        pos_encoding = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_size = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.size(0)

        # linear layer to project queries, keys, and values
        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)

        # scaled dot product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        attention = torch.matmul(scores, v)

        # concatenate heads and pass through final linear layer
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        x = self.fc(attention)

        return x

class Feedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.multihead_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.feedforward = Feedforward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # multi-head attention and residual connection
        attention = self.multihead_attn(x)
        x = self.norm1(x + self.dropout(attention))

        # feedforward and residual connection
        ff = self.feedforward(x)
        x = self.norm2(x + self.dropout(ff))

        return x

class GPT(nn.Module):
    def __init__(self, num_tokens, d_model=256, num_heads=8, num_layers=6, d_ff=1024, dropout=0.1):
        super().__init__()
        self.tokenizer = Tokenizer(num_tokens)
        self.embedding = nn.Embedding(num_tokens, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, num_tokens)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.fc(x)
        return x


# Define a small example dataset
class ExampleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.long)

    def __len__(self):
        return len(self.data)

train_data = ExampleDataset([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]])
train_loader = DataLoader(train_data, batch_size=2, shuffle=True)

# Define the GPT model and optimizer
model = GPT(num_tokens=15)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(10):
    running_loss = 0.0
    for inputs in train_loader:
        targets = inputs[:, 1:]
        inputs = inputs[:, :-1]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.transpose(1, 2), targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}: loss={running_loss:.3f}")

# Evaluate the model
test_inputs = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
with torch.no_grad():
    model.eval()
    output_logits = model(test_inputs)
    output_probs = nn.functional.softmax(output_logits.squeeze(0), dim=-1)
    print(f"Next token probabilities: {output_probs.tolist()}")