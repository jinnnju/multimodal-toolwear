import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib  
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib  
import torch
import torch.nn as nn

def _scale_features_batch(features_t: torch.Tensor, fitted_scaler: StandardScaler | None, scaler_path: str):
    """
    features_t: torch.Tensor, shape (B, F) or (B, T, F)
    return: (scaled_features_t, fitted_scaler, shape_info)
    """
    X = features_t
    if X.dim() == 3:
        B, T, F = X.shape
        X2 = X.reshape(B * T, F).detach().cpu().numpy()   # (N, F)
        shape_info = ("3d", B, T, F)
    elif X.dim() == 2:
        B, F = X.shape
        X2 = X.detach().cpu().numpy()                     # (B, F)
        shape_info = ("2d", B, F)
    else:
        raise ValueError(f"features must be 2D or 3D, got {X.dim()} with shape {tuple(X.shape)}")

    if fitted_scaler is None:
        fitted_scaler = StandardScaler().fit(X2)
        joblib.dump(fitted_scaler, scaler_path)

    X2_scaled = fitted_scaler.transform(X2)

    if shape_info[0] == "3d":
        _, B, T, F = shape_info[0], shape_info[1], shape_info[2], shape_info[3]
        X_scaled = torch.from_numpy(X2_scaled).reshape(B, T, F)
    else:
        _, B, F = shape_info[0], shape_info[1], shape_info[2]
        X_scaled = torch.from_numpy(X2_scaled)

    return X_scaled, fitted_scaler, shape_info

def _ensure_seq_dim(x: torch.Tensor, shape_tag: str):
    """
    모델 입력을 (B, T, F)로 맞추기.
    - 2D에서 온 경우만 (B, 1, F)로 unsqueeze
    - 3D는 그대로
    """
    if shape_tag == "2d":
        return x.unsqueeze(1)  # (B, 1, F)
    return x  # already (B, T, F)

def train_model(model, train_loader, val_loader, criterion, optimizer, device, scaler_path='scaler.pkl', epochs=20):
    model.to(device)

    fitted_scaler = None
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # -------------------- train --------------------
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            # labels -> (B, 1) float32 on device
            labels = labels.to(torch.float32).view(-1, 1).to(device)

            # 스케일링(2D/3D 모두 지원)
            scaled_features_cpu, fitted_scaler, shape_info = _scale_features_batch(features, fitted_scaler, scaler_path)
            # 모델 입력 모양 맞추기
            scaled_features = _ensure_seq_dim(scaled_features_cpu, shape_info[0]).to(device).to(torch.float32)

            optimizer.zero_grad()
            outputs = model(scaled_features)   # expects (B, T, F)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # -------------------- val --------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                labels = labels.to(torch.float32).view(-1, 1).to(device)
                scaled_features_cpu, _, shape_info = _scale_features_batch(features, fitted_scaler, scaler_path)
                scaled_features = _ensure_seq_dim(scaled_features_cpu, shape_info[0]).to(device).to(torch.float32)

                outputs = model(scaled_features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        train_losses.append(train_loss / max(1, len(train_loader)))
        val_losses.append(val_loss / max(1, len(val_loader)))

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")

    # -------------------- plot --------------------
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title('Training and Validation Loss')
    plt.legend(); plt.grid(True); plt.show()

