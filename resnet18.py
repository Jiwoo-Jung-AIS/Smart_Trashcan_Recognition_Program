# image classification with ResNet18
import os, shutil
from sklearn.model_selection import train_test_split

# Paths
data_dir = '/Users/Wharton_Camp/waste_data2/dataset'  # original full dataset
train_dir = '/Users/Wharton_Camp/waste_data2/train'
test_dir = '/Users/Wharton_Camp/waste_data2/test'

# Create train/test folders
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Split 80/20 by class
for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = [os.path.join(class_path, f) for f in os.listdir(class_path)]
    train_files, test_files = train_test_split(images, test_size=0.2, random_state=42)

    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    for f in train_files:
        shutil.copy(f, os.path.join(train_dir, class_name))
    for f in test_files:
        shutil.copy(f, os.path.join(test_dir, class_name))

print("Dataset split into train and test.")

# -------------------- Rest of your ResNet18 code --------------------

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

batch_size = 64

# Now load the split datasets
trainset = ImageFolder(root=train_dir, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = ImageFolder(root=test_dir, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

classes = trainset.classes
print("Classes:", classes)


# functions to show an image
def imshow(img):
    img = img * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)


# show images
imshow(torchvision.utils.make_grid(images))


# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


# CNN model
device = torch.device("cpu")
weights = ResNet18_Weights.DEFAULT
net = resnet18(weights=weights)
net.fc = nn.Linear(net.fc.in_features, len(classes))
net = net.to(device)

# loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# Training the model
for epoch in range(10):  # loop over the dataset multiple times
    net.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:
            print(f'[{epoch + 1}, { + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')


# Save model
torch.save(net.state_dict(), './waste_resnet18.pth')

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))


# Testing the model
net = resnet18(weights=None)
net.fc = nn.Linear(net.fc.in_features, len(classes))
net.load_state_dict(torch.load(PATH))
net = net.to(device)
net.eval()

# Predictions
dataiter = iter(testloader)
images, labels = next(dataiter)

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))


# Overall accuracy
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


import pandas as pd

# Save training set paths
train_files = [path for path, label in trainset.samples]
train_labels = [label for path, label in trainset.samples]
pd.DataFrame({'path': train_files, 'label': train_labels}).to_csv('train_list.csv', index=False)

# Save test set paths
test_files = [path for path, label in testset.samples]
test_labels = [label for path, label in testset.samples]
pd.DataFrame({'path': test_files, 'label': test_labels}).to_csv('test_list.csv', index=False)
