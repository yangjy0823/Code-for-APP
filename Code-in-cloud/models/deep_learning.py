# models/deep_learning.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import copy

# --- 1. Model Architectures ---

class LSTMClassifier(nn.Module):
    """一个标准的LSTM分类器"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout, bidirectional=False):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        
        # LSTM层
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, seq_length, hidden_size * num_directions)
        
        # 我们只取序列最后一个时间步的输出进行分类
        last_time_step_out = lstm_out[:, -1, :]
        
        # Dropout 和全连接层
        out = self.dropout(last_time_step_out)
        out = self.fc(out)
        return out

class GRUClassifier(nn.Module):
    """一个标准的GRU分类器"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout, bidirectional=False):
        super(GRUClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(gru_output_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_time_step_out = gru_out[:, -1, :]
        out = self.dropout(last_time_step_out)
        out = self.fc(out)
        return out

class CNNLSTMClassifier(nn.Module):
    """一个结合CNN和LSTM的混合分类器"""
    def __init__(self, input_size, cnn_filters, kernel_size, lstm_hidden_size, lstm_num_layers, num_classes, dropout):
        super(CNNLSTMClassifier, self).__init__()
        
        # 1D CNN for feature extraction from sequence
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=cnn_filters, kernel_size=kernel_size, padding='same')
        self.relu = nn.ReLU()
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0
        )
        
        # Classifier head
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        
        # Conv1d expects (batch_size, channels, seq_length)
        x = x.permute(0, 2, 1)
        
        conv_out = self.conv1d(x)
        conv_out = self.relu(conv_out)
        
        # LSTM expects (batch_size, seq_length, features)
        conv_out = conv_out.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(conv_out)
        
        last_time_step_out = lstm_out[:, -1, :]
        
        out = self.dropout(last_time_step_out)
        out = self.fc(out)
        return out


# --- 2. PyTorch Trainer Class ---

class PyTorchTrainer:
    """一个可复用的PyTorch训练器，包含训练、验证和早停逻辑"""
    def __init__(self, model, criterion, optimizer, scheduler, device, patience=10):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.patience = patience
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.best_model_state_dict = None

    def _train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for data, targets in train_loader:
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            # Optional: Gradient Clipping
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(train_loader)

    def _validate_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def fit(self, train_loader, val_loader, epochs):
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate_epoch(val_loader)
            
            print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            # Scheduler step
            if self.scheduler:
                self.scheduler.step(val_loss)

            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                # Deep copy the model's state dict
                self.best_model_state_dict = copy.deepcopy(self.model.state_dict())
            else:
                self.epochs_no_improve += 1
            
            if self.epochs_no_improve >= self.patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break
        
        # Load the best model weights before returning
        if self.best_model_state_dict:
            self.model.load_state_dict(self.best_model_state_dict)
            
        return self.model, history