'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-10-25 15:42:19
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-10-26 13:14:20
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class TraditionalLSTM(nn.Module):
    def __init__(self, input_size=64, hidden_size=16, dropout=0.15):
        super(TraditionalLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        return out

class StackedLSTM(nn.Module):
    def __init__(self, input_size=64, hidden_sizes=[128, 64, 32, 16], dropout=0.15):
        super(StackedLSTM, self).__init__()
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        for i, hidden_size in enumerate(hidden_sizes):
            self.lstm_layers.append(nn.LSTM(input_size if i == 0 else hidden_sizes[i-1],
                                            hidden_size,
                                            batch_first=True))
            self.dropout_layers.append(nn.Dropout(dropout))

        self.fc = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x):
        for lstm, dropout in zip(self.lstm_layers, self.dropout_layers):
            x, _ = lstm(x)
            x = dropout(x)
        out = self.fc(x[:, -1, :])
        return out

class DynamicResidualStackedLSTM(nn.Module):
    def __init__(self, input_size=64, hidden_sizes=[128, 64, 32, 16, 8], dropout=0.15):
        super(DynamicResidualStackedLSTM, self).__init__()
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.dynamic_residual_layers = nn.ModuleList()

        for i, hidden_size in enumerate(hidden_sizes):
            lstm_input_size = input_size if i == 0 else hidden_sizes[i-1]
            self.lstm_layers.append(nn.LSTM(lstm_input_size, hidden_size, batch_first=True))
            self.dropout_layers.append(nn.Dropout(dropout))
            if i > 0: 
                self.dynamic_residual_layers.append(nn.Linear(lstm_input_size, hidden_size))

        self.fc = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x):
        for i, (lstm, dropout) in enumerate(zip(self.lstm_layers, self.dropout_layers)):
            residual = x
            x, _ = lstm(x)
            x = dropout(x)
            
            if i > 0: 
                dynamic_weight = torch.sigmoid(self.dynamic_residual_layers[i-1](residual))
                if residual.size(-1) != x.size(-1):
                    residual = self.dynamic_residual_layers[i-1](residual)
                x = x + dynamic_weight * residual

        out = self.fc(x[:, -1, :])
        return out

def train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=18, lr=0.01, patience=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if len(X_train.shape) == 2:
        X_train = X_train.reshape(X_train.shape[0], 1, -1)
        X_test = X_test.reshape(X_test.shape[0], 1, -1)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    best_val_loss = float('inf')
    no_improve_epochs = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y.unsqueeze(1)).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(test_loader)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improve_epochs += 1

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if no_improve_epochs >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break

    model.load_state_dict(torch.load('best_model.pth'))
    return model

def traditional_LSTM_model(X_train, y_train, X_test, y_test):
    model = TraditionalLSTM()
    return train_model(model, X_train, y_train, X_test, y_test)

def stack_LSTM_model(X_train, y_train, X_test, y_test):
    model = StackedLSTM()
    return train_model(model, X_train, y_train, X_test, y_test)

def dynamic_residuals_stack_LSTM_model(X_train, y_train, X_test, y_test):
    input_size = X_train.shape[-1] 
    model = DynamicResidualStackedLSTM(input_size=input_size)
    return train_model(model, X_train, y_train, X_test, y_test, epochs=200, lr=0.001)
