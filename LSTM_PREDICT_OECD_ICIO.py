import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 数据加载和预处理
def load_data(years, data_dir):
    data = []
    for year in years:
        file_path = os.path.join(data_dir, f"{year}_SML.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 {file_path} 不存在")
        df = pd.read_csv(file_path, header=0, index_col=0)
        data.append(df.values)
    return np.stack(data, axis=0)

def preprocess_data(data):
    num_years, num_rows, num_cols = data.shape
    input_dim = num_rows * num_cols
    data_flat = data.reshape(num_years, input_dim)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_flat)
    return torch.FloatTensor(data_scaled), scaler, input_dim

# 设定数据路径和年份
data_dir = "./ICIO DATA"
years = list(range(2010, 2021))
data = load_data(years, data_dir)
data_tensor, scaler, input_dim = preprocess_data(data)

# LSTM 模型
def prepare_sequences(data_tensor, seq_len):
    X, y = [], []
    for i in range(len(data_tensor) - seq_len):
        X.append(data_tensor[i:i+seq_len])
        y.append(data_tensor[i+seq_len])
    return torch.stack(X), torch.stack(y)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# 训练和预测
def train_lstm(model, train_loader, num_epochs, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    writer = SummaryWriter("runs/icio_lstm")
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.close()

def predict_lstm(model, data_tensor, seq_len, future_years):
    model.eval()
    predictions = []
    last_sequence = data_tensor[-seq_len:].unsqueeze(0)
    with torch.no_grad():
        for _ in range(future_years):
            pred = model(last_sequence)
            predictions.append(pred.squeeze(0))
            last_sequence = torch.cat((last_sequence[:, 1:, :], pred.unsqueeze(1)), dim=1)
    return torch.stack(predictions)

# 训练 LSTM
seq_len = 5
X, y = prepare_sequences(data_tensor, seq_len)
train_size = int(0.8 * len(X))
train_X, val_X = X[:train_size], X[train_size:]
train_y, val_y = y[:train_size], y[train_size:]
train_dataset = TensorDataset(train_X, train_y)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

model = LSTMModel(input_dim=input_dim, hidden_dim=128, num_layers=2, output_dim=input_dim)
train_lstm(model, train_loader, num_epochs=200, learning_rate=0.001)

# 预测未来 ICIO 表
future_years = 5
predictions = predict_lstm(model, data_tensor, seq_len, future_years)
predictions_np = scaler.inverse_transform(predictions.numpy())
predictions_reshaped = predictions_np.reshape(future_years, data.shape[1], data.shape[2])

# # 保存预测结果
# for i, year in enumerate(range(2021, 2021 + future_years)):
#     pd.DataFrame(predictions_reshaped[i], index=data[0].index, columns=data[0].columns).to_csv(f"pred_{year}.csv")


# 保存预测结果
for i, year in enumerate(range(2021, 2021 + future_years)):
    # 假设 data[0] 是一个 numpy 数组，我们需要手动指定索引和列名
    # 如果原始数据的索引和列名已知，可以在这里指定
    index = None  # 替换为原始数据的索引
    columns = None  # 替换为原始数据的列名
    
    # 如果 data[0] 是一个 DataFrame，可以直接使用它的 index 和 columns
    if isinstance(data[0], pd.DataFrame):
        index = data[0].index
        columns = data[0].columns
    else:
        # 如果 data[0] 是一个 numpy 数组，假设索引和列名是整数
        index = range(data[0].shape[0])
        columns = range(data[0].shape[1])
    
    pd.DataFrame(predictions_reshaped[i], index=index, columns=columns).to_csv(f"pred_{year}.csv")
