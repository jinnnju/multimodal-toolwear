import torch
import torch.nn as nn
from tqdm import tqdm
import timm
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights 




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
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.base = mobilenet_v2(weights=weights)

        self.base.classifier[1] = nn.Linear(self.base.last_channel, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.base(x)
        return self.sigmoid(x)

class ResNetClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetClassifier, self).__init__()

        # > torchvision 0.13 
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = resnet18(weights=weights)

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        return self.sigmoid(x)


# Vision Transformer (ViT)
class ViTBinary(nn.Module):
    def __init__(self, pretrained=True, img_size=128):
        super(ViTBinary, self).__init__()
        self.model = timm.create_model(
            'vit_base_patch16_224',  
            pretrained=pretrained,
            num_classes=1, img_size=128  
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.model(x))



# DeiT (Data-efficient Image Transformers)
class DeiTBinary(nn.Module):
    def __init__(self, pretrained=True):
        super(DeiTBinary, self).__init__()
        self.model = timm.create_model(
            'deit_base_distilled_patch16_224',
            pretrained=pretrained,
            num_classes=1, img_size=128
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.model(x))



# Swin Transformer
class SwinBinary(nn.Module):
    def __init__(self, pretrained=True):
        super(SwinBinary, self).__init__()
        self.model = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=pretrained,
            num_classes=1, img_size=128
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.model(x))



# BEiT
class BEiTBinary(nn.Module):
    def __init__(self, pretrained=True):
        super(BEiTBinary, self).__init__()
        self.model = timm.create_model(
            'beit_base_patch16_224',
            pretrained=pretrained,
            num_classes=1, img_size=128
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.model(x))

