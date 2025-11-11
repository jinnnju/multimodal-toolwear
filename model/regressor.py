import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error
import timm
from torchvision.models import mobilenet_v2, resnet18


class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM output
        if x.dim() == 2:   # (B, D)인 경우
            x = x.unsqueeze(1)
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
        if x.dim() == 2:   # (B, D)인 경우
            x = x.unsqueeze(1)
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
        if x.dim() == 2:   # (B, D)인 경우
            x = x.unsqueeze(1)
        out, _ = self.bilstm(x)         # out: [B, T, 2*H]
        out = self.fc(out[:, -1, :])    # 마지막 timestep만 추출
        return out                      # 출력: [B, output_size]


import torch
import torch.nn as nn
class TransformerRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, **kwargs):
        super().__init__()
        d_model = hidden_size
        dropout = kwargs.get("dropout", 0.1)
        ff_mult = kwargs.get("ff_mult", 4)
        max_len = kwargs.get("max_len", 4096)
        pool = kwargs.get("pool", "cls")  # default를 cls로

        nhead = kwargs.get("nhead", 4)   # 명시 추천 (d_model % nhead == 0 가정)
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.input_proj = nn.Linear(input_size, d_model)
        self.input_ln   = nn.LayerNorm(d_model)            # 추가
        self.pos_emb    = nn.Embedding(max_len + 1, d_model)  # +1 (CLS용 pos 0)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_model))
        self.dropout    = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,            # Pre-LN로 변경
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers, enable_nested_tensor=False)

        assert pool in ("cls", "mean", "last")
        self.pool = pool
        self.out_ln = nn.LayerNorm(d_model)                # 추가
        self.fc = nn.Linear(d_model, output_size)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, key_padding_mask=None):
        if x.dim() == 2:   # (B, D)인 경우
            x = x.unsqueeze(1)
        # x: (B, T, input_size)
        B, T, _ = x.shape
        h = self.input_proj(x)
        h = self.input_ln(h)

        # CLS + pos
        cls = self.cls_token.expand(B, 1, -1)             # (B,1,d)
        h = torch.cat([cls, h], dim=1)                    # (B, T+1, d)

        pos_ids = torch.arange(T + 1, device=h.device).unsqueeze(0).expand(B, T + 1)  # 0..T (0=CLS)
        h = h + self.pos_emb(pos_ids)

        h = self.dropout(h)

        # key_padding_mask: (B, T) -> (B, T+1) with False at CLS
        if key_padding_mask is not None:
            pad = torch.zeros((key_padding_mask.shape[0], 1), dtype=key_padding_mask.dtype, device=key_padding_mask.device)
            key_padding_mask = torch.cat([pad, key_padding_mask], dim=1)

        h = self.encoder(h, src_key_padding_mask=key_padding_mask)  # (B, T+1, d)

        if self.pool == "cls":
            pooled = h[:, 0, :]
        elif self.pool == "last":
            pooled = h[:, -1, :]
        else:
            pooled = h[:, 1:, :].mean(dim=1)  # mean over tokens (exclude CLS)

        pooled = self.out_ln(pooled)
        return self.fc(pooled)

# class TransformerRegressor(nn.Module):
#     """
#     (B, T, input_size) -> (B, output_size)
#     - PyTorch 내장 TransformerEncoder 사용 (batch_first=True)
#     """
#     def __init__(self, input_size, hidden_size, num_layers, output_size, **kwargs):
#         super(TransformerRegressor, self).__init__()
#         d_model = hidden_size
#         dropout = kwargs.get("dropout", 0.1)
#         ff_mult = kwargs.get("ff_mult", 4)
#         max_len = kwargs.get("max_len", 4096)
#         pool = kwargs.get("pool", "last")  # "last" | "mean"

#         # nhead 자동 설정: d_model의 약수 중 합리적인 값 선택
#         nhead = kwargs.get("nhead", None)
#         if nhead is None or d_model % nhead != 0:
#             for cand in [8, 4, 2, 1]:
#                 if d_model % cand == 0:
#                     nhead = cand
#                     break

#         self.input_proj = nn.Linear(input_size, d_model)
#         self.pos_emb = nn.Embedding(max_len, d_model)

#         enc_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=ff_mult * d_model,
#             dropout=dropout,
#             batch_first=True,
#             norm_first=False,
#             activation="gelu",
#         )
#         self.encoder = nn.TransformerEncoder(
#             enc_layer,
#             num_layers=num_layers,
#             enable_nested_tensor=False
#         )

#         assert pool in ("last", "mean")
#         self.pool = pool
#         self.fc = nn.Linear(d_model, output_size)

#     def forward(self, x):
#         """
#         x: (B, T, input_size)
#         return: (B, output_size)
#         """
#         B, T, _ = x.shape
#         device = x.device

#         h = self.input_proj(x)  # (B, T, d_model)

#         pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # (B, T)
#         h = h + self.pos_emb(pos_ids)

#         h = self.encoder(h)  # (B, T, d_model)

#         if self.pool == "last":
#             pooled = h[:, -1, :]  # (B, d_model)
#         else:
#             pooled = h.mean(dim=1)

#         return self.fc(pooled)  # (B, output_size)
