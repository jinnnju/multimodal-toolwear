import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import joblib 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import numpy as np
from scipy.signal import savgol_filter
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from scipy.signal import savgol_filter

def evaluate_model(model, test_loader, device, scaler_path='scaler.pkl', visualize=True):
    """
    Torch 모델 평가 (Raw: (B,T,F) 또는 Feature: (B,D) 모두 지원)
    - 스케일러는 피처 축(F 또는 D) 기준으로 저장된 걸 로드해 사용
    - Raw 입력은 (B,T,F) 유지. 벡터 입력은 (B,D) 유지.
    """
    model.eval()
    scaler: StandardScaler = joblib.load(scaler_path)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for features, labels in test_loader:
            # --- Tensor -> numpy ---
            if torch.is_tensor(features):
                feats_np = features.detach().cpu().numpy()
            else:
                feats_np = np.asarray(features)

            if torch.is_tensor(labels):
                labels_np = labels.detach().cpu().numpy().reshape(-1)
            else:
                labels_np = np.asarray(labels).reshape(-1)

            # --- Scale by feature axis ---
            if feats_np.ndim == 3:
                # Raw: (B, T, F)
                B, T, F = feats_np.shape
                feats_2d = feats_np.reshape(-1, F)             # (-1, F)
                feats_2d = scaler.transform(feats_2d)          # scale by F
                feats_scaled = torch.from_numpy(feats_2d.reshape(B, T, F)).float().to(device)
            elif feats_np.ndim == 2:
                # Feature: (B, D)
                B, D = feats_np.shape
                feats_2d = scaler.transform(feats_np)          # (B, D)
                feats_scaled = torch.from_numpy(feats_2d).float().to(device)
            else:
                raise ValueError(f"Unexpected features shape: {feats_np.shape}")

            labels_t = torch.from_numpy(labels_np).float().to(device).view(-1, 1)

            # --- Forward ---
            outputs = model(feats_scaled)                      # Raw면 (B,T,F) 그대로, Feature면 (B,D)
            preds = outputs.detach().cpu().numpy().reshape(-1)

            all_preds.extend(preds)
            all_labels.extend(labels_np)

    # --- Metrics ---
    all_preds  = np.asarray(all_preds)
    all_labels = np.asarray(all_labels)

    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    mae  = mean_absolute_error(all_labels, all_preds)
    r2   = r2_score(all_labels, all_preds)
    mape = np.mean(np.abs((all_labels - all_preds) / (all_labels + 1e-8))) * 100
    explained_variance = explained_variance_score(all_labels, all_preds)

    print("Evaluation Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R²  : {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Explained Var: {explained_variance:.4f}")

    if visualize:
        plt.figure(figsize=(12, 6))
        t = np.arange(len(all_labels))
        plt.plot(t, all_labels, label="True", alpha=0.7, marker='o', linestyle='-')
        plt.plot(t, all_preds,  label="Pred", alpha=0.7, marker='x', linestyle='-')

        # savgol window는 길이보다 작고 홀수여야 함 → 자동 조정
        win = min(15, len(all_labels) - (1 - len(all_labels) % 2))
        if win >= 5 and win % 2 == 1:
            plt.plot(t, savgol_filter(all_labels, window_length=win, polyorder=2),
                     label="True Trend", linestyle='--', linewidth=1.5)
            plt.plot(t, savgol_filter(all_preds,  window_length=win, polyorder=2),
                     label="Pred Trend", linestyle='--', linewidth=1.5)

        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.title('True vs Predicted Over Time')
        plt.grid(True)
        plt.legend()
        plt.show()

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2 Score": r2,
        "MAPE": mape,
        "Explained Variance Score": explained_variance
    }, all_preds.tolist(), all_labels.tolist()

def evaluate_model_ML(model, X_test, y_test, visualize=True, expect_seq=False):
    """
    Torch 모델이지만 간단한 ML 스타일로 forward만 호출하는 경우.
    - expect_seq=False: X_test는 (N, D), 그대로 model(X)
    - expect_seq=True : X_test는 (N, T, F), 그대로 model(X)
    """
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32)
        if not expect_seq and X_t.dim() == 2:
            y_pred_t = model(X_t)
        elif expect_seq and X_t.dim() == 3:
            y_pred_t = model(X_t)    # (N,T,F) 입력을 기대
        else:
            raise ValueError(f"X_test shape {X_t.shape} not compatible with expect_seq={expect_seq}")
        y_pred = y_pred_t.cpu().numpy().reshape(-1)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
    explained_variance = explained_variance_score(y_test, y_pred)

    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "R2 Score": r2,
        "MAPE": mape,
        "Explained Variance Score": explained_variance,
    }

    if visualize:
        plt.figure(figsize=(12, 6))
        t = np.arange(len(y_test))
        plt.plot(t, y_test, label="True", alpha=0.7, marker='o', linestyle='-')
        plt.plot(t, y_pred, label="Pred", alpha=0.7, marker='x', linestyle='-')
        plt.xlabel('Sample Index'); plt.ylabel('Value'); plt.title('True vs Predicted Values')
        plt.grid(True); plt.legend(); plt.show()

    return metrics, y_pred, y_test



def evaluate_model_ML_s(model, X_test, y_test, visualize=True):
    # Scikit-learn 기반 모델의 예측
    y_pred = model.predict(X_test)

    # 평가 지표 계산
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    explained_variance = explained_variance_score(y_test, y_pred)

    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "R2 Score": r2,
        "MAPE": mape,
        "Explained Variance Score": explained_variance,
    }

    # 결과 시각화
    if visualize:
        plt.figure(figsize=(12, 6))
        time = range(len(y_test))  # Time axis

        # Plot true vs predicted values
        plt.plot(time, y_test, label="True Labels", color='green', alpha=0.7, linestyle='-', marker='o')
        plt.plot(time, y_pred, label="Predicted Labels", color='blue', alpha=0.7, linestyle='-', marker='x')

        # Labels, title, and legend
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.title('True vs Predicted Values')
        plt.legend()
        plt.grid(True)
        plt.show()

    return metrics, y_pred, y_test
