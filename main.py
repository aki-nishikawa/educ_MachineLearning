import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from models import MyModel

DEVICE = 'cuda:0'
EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 0.001
model = MyModel().to(DEVICE)

def load_data(transform):
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    return train_loader, test_loader

def train(train_loader):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print('=' * 80)
    for epoch in range(EPOCHS):
        start_time = time.time()
        for i, (X, y) in enumerate(train_loader, 0):
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        print(
            f"| epoch : {epoch}/{EPOCHS} "
            f"| loss : {loss.item()} "
            f"| time : {time.time() - start_time} "
            f"|")
        print('-' * 80)

def test(test_loader):
    model.eval()
    label = []
    pred = []
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader, 0):
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            label.append(y.to('cpu').detach().numpy().copy())
            pred.append(model(X).to('cpu').detach().numpy().copy().argmax(axis=1))
    label = np.array(label).flatten()
    pred = np.array(pred).flatten()
    ans = (label == pred)
    print(np.sum(ans) / 10000)

if __name__ == "__main__":

    ### load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_loader, test_loader = load_data(transform=transform)

    train(train_loader)
    test(test_loader)