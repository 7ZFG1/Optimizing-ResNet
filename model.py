import torch
import torch.nn as nn
from torchvision import models

class OptimizedResNet101Classifier(nn.Module):
    def __init__(self, num_classes, frozen_layers):
        super(ResNet101Classifier, self).__init__()
        
        self.resnet = models.resnet101(pretrained=True)

        for name, param in self.resnet.named_parameters():
            for layer in frozen_layers:
                if layer in name:
                    param.requires_grad = False

        # Additional convolutional layer for feature extraction
        self.additional_conv = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        in_features = 512
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.additional_conv(x)

        x = self.fc(x)
        return x

class OptimizedResNet50Classifier(nn.Module):
    def __init__(self, num_classes, frozen_layers):
        super(ResNet50Classifier, self).__init__()
        
        self.resnet = models.resnet50(pretrained=True)

        for name, param in self.resnet.named_parameters():
            for layer in frozen_layers:
                if layer in name:
                    param.requires_grad = False

        self.additional_conv = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        in_features = 512
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.additional_conv(x)

        x = self.fc(x)
        return x

class ResNet101Classifier(nn.Module):
    def __init__(self, num_classes, frozen_layers):
        super(ResNet101Classifier, self).__init__()
        
        self.resnet = models.resnet101(pretrained=True)

        for name, param in self.resnet.named_parameters():
            for layer in frozen_layers:
                if layer in name:
                    param.requires_grad = False

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
    

class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes, frozen_layers):
        super(ResNet50Classifier, self).__init__()
        
        self.resnet = models.resnet50(pretrained=True)

        for name, param in self.resnet.named_parameters():
            for layer in frozen_layers:
                if layer in name:
                    param.requires_grad = False

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
