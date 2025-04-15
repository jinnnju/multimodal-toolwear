import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error
import timm
from torchvision.models import mobilenet_v2, resnet18

################### Regression Models
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM output
        out, _ = self.lstm(x)
        # Fully connected output using the last time step
        out = self.fc(out[:, -1, :])
        return out



class GRURegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRURegressor, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # GRU output
        out, _ = self.gru(x)
        # Fully connected output using the last time step
        out = self.fc(out[:, -1, :])
        return out


class BiLSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTMRegressor, self).__init__()
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 for bidirection

    def forward(self, x):
        out, _ = self.bilstm(x)         # out: [B, T, 2*H]
        out = self.fc(out[:, -1, :])    # 마지막 timestep만 추출
        return out                      # 출력: [B, output_size]

##################### Classification models

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # 입력 이미지가 (3, 128, 128)인 경우
        self.fc1 = nn.Linear(64 * 32 * 32, 128)  # = 65536
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))  # → (B, 64, 32, 32)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


class EfficientNetBinary(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientNetBinary, self).__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.model(x))


class MobileNetBinary(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetBinary, self).__init__()
        self.base = mobilenet_v2(pretrained=pretrained)
        self.base.classifier[1] = nn.Linear(self.base.last_channel, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.base(x)
        return self.sigmoid(x)


class ResNetClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetClassifier, self).__init__()
        self.resnet = resnet18(pretrained=pretrained)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.resnet(x)
        return self.sigmoid(x)



########Mlp model

class MLPStacking(nn.Module):
    def __init__(self, input_dim):
        super(MLPStacking, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

class RUL_MLP(nn.Module):
    def __init__(self, input_dim):
        super(RUL_MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)  