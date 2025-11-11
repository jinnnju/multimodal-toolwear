import numpy as np  
import torch  
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

import os
from PIL import Image
from torch.utils.data import Dataset

class MelSpectrogramDataset(Dataset):
    def __init__(self, data_dir, train_anomaly_threshold, transform=None):
        """
        data_dir: 이미지가 저장된 폴더 경로
        train_anomaly_threshold: anomaly로 간주할 시작 인덱스
        transform: torchvision.transforms 등
        """
        self.data_dir = data_dir
        self.train_anomaly_threshold = train_anomaly_threshold
        self.transform = transform

        valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        files = [f for f in os.listdir(data_dir) if f.lower().endswith(valid_ext)]
        self.image_files = sorted(files, key=lambda x: int(x.split("_")[0]))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # threshold를 인자로 받은 값으로 처리
        label = 1 if idx >= self.train_anomaly_threshold else 0

        if self.transform:
            image = self.transform(image)

        return image, label, idx  


def prepare_data(loader):
    """
    PyTorch DataLoader에서 데이터를 추출하여 NumPy 배열로 변환

    Args:
        loader (torch.utils.data.DataLoader): PyTorch DataLoader 객체
    
    Returns:
        tuple: (features, labels)
        - features (np.ndarray): 입력 데이터
        - labels (np.ndarray): 타겟 데이터
    """
    features, labels = [], []
    for batch_features, batch_labels in loader:
        features.extend(batch_features.numpy())  # PyTorch Tensor → NumPy 배열
        labels.extend(batch_labels.numpy())
    return np.array(features), np.array(labels)



##for regression model
def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy


# def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
#     train_losses, val_losses = [], []
#     best_val_loss = float("inf")

#     for epoch in range(num_epochs):
#         model.train()
#         running_loss, correct, total = 0.0, 0, 0

#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
#             predicted = (outputs > 0.5).float()
#             correct += (predicted == labels).sum().item()
#             total += labels.size(0)

#         train_loss = running_loss / len(train_loader)
#         train_acc = correct / total
#         val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

#         train_losses.append(train_loss)
#         val_losses.append(val_loss)

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(model.state_dict(), "best_model.pth")
#             print(f"Saved best model (Val Loss: {best_val_loss:.4f})")

#         print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

#     plot_loss_curve(train_losses, val_losses)
from tqdm import tqdm

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"✅ Saved best model (Val Loss: {best_val_loss:.4f})")

        print(f"[{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    plot_loss_curve(train_losses, val_losses)


def plot_loss_curve(train_losses, val_losses):
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(val_losses, label="Val Loss", marker="o", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.close()


### test for classifier 
def evaluate_test(model, test_loader):
    model.eval()
    correct, total = 0, 0
    predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            predicted = (outputs > 0.5).float()
            probabilities = outputs.sigmoid().cpu().numpy().flatten()  # 예측 확률값 추가

            # Accuracy 계산
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # 라벨을 squeeze()로 차원 맞춰서 저장
            for true_label, pred_prob in zip(labels.cpu().numpy().squeeze(), probabilities):
                predictions.append((int(true_label), pred_prob))  # 예측 확률 포함

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return predictions  # 예측 결과 반환
