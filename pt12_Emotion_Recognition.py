# convolutional neural network
# limits on computational power usage

# implement CNN
# preprocess data(dataloader, etc)
# implement model class
# training loop (gradient, loss)
# testing

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import cv2 as cv

"""
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cv1 = nn.Conv2d(in_channels=1, out_channels=16,
                             kernel_size=3, padding=1)
        self.cv2 = nn.Conv2d(in_channels=16, out_channels=32,
                             kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 32 from cv2 output, 12 x 12 from Maxpooling twice with kernel_size = 2
        self.fc1 = nn.Linear(32 * 12 * 12, 16)
        self.fc2 = nn.Linear(16, 7)

    def forward(self, data):
        data = self.pool(F.relu(self.cv1(data)))
        data = self.pool(F.relu(self.cv2(data)))
        data = data.view(data.size(0), -1)
        data = F.relu(self.fc1(data))
        data = self.fc2(data)
        return data
"""
# making and training my own model computationally too expensive. divert to transfer learning

# data preprocessing
class EmotionDataset(Dataset):
    def __init__(self, archive, transform=None):
        self.image_paths = []
        self.labels = []
        self.archive = archive
        self.transform = transform

        for index, directory in enumerate(os.listdir(self.archive)):
            directory_path = os.path.join(self.archive, directory)
            for filepath in os.listdir(directory_path):
                path = os.path.join(directory_path, filepath)
                self.image_paths.append(path)
                self.labels.append(index)

    def __getitem__(self, index):
        image = cv.imread(self.image_paths[index], cv.IMREAD_COLOR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (224, 224))
        # tensor must be converted from (H, W, C) (numpy) to (C, H, W) (tensor)
        label = torch.tensor(int(self.labels[index]), dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.image_paths)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# loading dataloaders
train_archive = "archive/train"
test_archive = "archive/test"

train_data = EmotionDataset(train_archive, transform=transform)
test_data = EmotionDataset(test_archive, transform=transform)

train_dataloader = DataLoader(train_data, batch_size=5, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=5, shuffle=True)

# import mobilenet v2, change the final layer to have 7 labels
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)

model.train()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
n_epochs = 5

# train the model with seven emotions
for epoch in range(n_epochs):
    avg_loss = 0
    for feature, label in train_dataloader:
        prediction = model(feature)
        loss = criterion(prediction, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss
    avg_loss /= len(train_dataloader)
    print(avg_loss)

#testing accuracy
model.eval()
total = 0
correct = 0

with torch.no_grad():
    for features, labels in test_dataloader:
        predictions = model(features)
        _, predicted = torch.max(predictions, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")