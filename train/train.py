import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from net import RL_model
import os

# 定义数据集类
class MyDataset(Dataset):
    def __init__(self, dataset_name, train=True):
        if dataset_name == 'mnist':
            self.dataset = datasets.MNIST('~/PycharmProjects/pythonProject1/DeepLearning/RL',
                                          download=True, train=train, transform=transforms.ToTensor())
        elif dataset_name == 'cifar10':
            self.dataset = datasets.CIFAR10('~/PycharmProjects/pythonProject1/DeepLearning/RL',
                                            download=True, train=train, transform=transforms.ToTensor())
        elif dataset_name == 'fashion_mnist':
            self.dataset = datasets.FashionMNIST('~/PycharmProjects/pythonProject1/DeepLearning/RL',
                                                 download=True, train=train, transform=transforms.ToTensor())
        else:
            raise ValueError('Unsupported dataset')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        return data, label

def train(model, device, dataset_name, epochs, batch_size, learning_rate, val_size=0.1):
    dataset = MyDataset(dataset_name)
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_size * num_train))

    np.random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data.view(-1, 784 if dataset_name == 'mnist' else 3072))
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 打印每个 epoch 的 loss
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}')

        # 验证集评估
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data.view(-1, 784 if dataset_name == 'mnist' else 3072))
                _, predicted = torch.max(output, 1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
        accuracy = correct / total
        print(f'Validation Accuracy: {accuracy:.4f}')

        # 保存最佳模型
        if accuracy > best_acc:
            best_acc = accuracy
            folder = 'save_model'
            if not os.path.exists(folder):
                os.mkdir('save_model')
            min_acc = best_acc
            print('save best model')
            torch.save(model.state_dict(), 'save_model/best_model.pth')
    print('Done!')


def inference(model, device, dataset_name):
    dataset = MyDataset(dataset_name, train=False)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data.view(-1, 784 if dataset_name == 'mnist' else 3072))
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
    accuracy = correct / len(dataset)
    print(f'Test Accuracy: {accuracy:.4f}')

# 设置超参数
dataset_name = 'mnist'  #可以选择'mnist', 'cifar10', 'fashion_mnist'等不同数据类型
epochs = 20
batch_size = 128
learning_rate = 0.001
input_dim = 784 if dataset_name == 'mnist' else 3072
hidden_dim = 256
output_dim = 10

# 初始化模型和设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RL_model(input_dim, hidden_dim, output_dim)
model.to(device)

# 训练模型
train(model, device, dataset_name, epochs, batch_size, learning_rate)
# 推理模型
inference(model, device, dataset_name)
#model.eval()
#torch.save(model.state_dict(), 'model.pth')
