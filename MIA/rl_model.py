'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

class RL_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout_rate=0.2):
        super(RL_model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.fc_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.fc_layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(dropout_rate))

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        for layer in self.fc_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

    def get_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float)
            action_probs = self.forward(state)
            action_probs = torch.softmax(action_probs, dim=1)
            action = torch.argmax(action_probs).item()
            return action

    def get_action_probs(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float)
            action_probs = self.forward(state)
            action_probs = torch.softmax(action_probs, dim=1)
            return action_probs

    def get_value(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float)
            value = self.forward(state)
            return value
'''

import torch
import torch.nn as nn

class RL_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout_rate=0.2):
        super(RL_model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.fc_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.fc_layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(dropout_rate))

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Ensure x is the correct shape (batch_size, input_dim)
        x = x.view(x.size(0), -1)  # Flatten input if necessary
        for layer in self.fc_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

    def get_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            action_probs = self.forward(state)
            action_probs = torch.softmax(action_probs, dim=1)
            action = torch.argmax(action_probs).item()
            return action

    def get_action_probs(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            action_probs = self.forward(state)
            action_probs = torch.softmax(action_probs, dim=1)
            return action_probs

    def get_value(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            value = self.forward(state)
            return value


# print("training attack model...")
# print("Training attack model for class 0...")
# print("training number is 11592")
# print("testing number is 213")
# print("Epoch: 4 Accuracy of the network on the training set: 67 %")
# print("Epoch: 4 Accuracy of the network on the test set: 67 %")
# print("Accuracy score for class 0:")
# print("0.675")
#
# print("Training attack model for class 1...")
# print("training number is 13511")
# print("testing number is 249")
# print("Epoch: 4 Accuracy of the network on the training set: 66 %")
# print("Epoch: 4 Accuracy of the network on the test set: 63 %")
# print("Accuracy score for class 1:")
# print("0.655")
#
# print("Training attack model for class 2...")
# print("training number is 12402")
# print("testing number is 199")
# print("Epoch: 4 Accuracy of the network on the training set: 67 %")
# print("Epoch: 4 Accuracy of the network on the test set: 68 %")
# print("Accuracy score for class 2:")
# print("0.6888888888888888")
#
# print("Training attack model for class 3...")
# print("training number is 12295")
# print("testing number is 205")
# print("Epoch: 4 Accuracy of the network on the training set: 65 %")
# print("Epoch: 4 Accuracy of the network on the test set: 71 %")
# print("Accuracy score for class 3:")
# print("0.71")
#
# print("Training attack model for class 4...")
# print("training number is 11523")
# print("testing number is 191")
# print("Epoch: 4 Accuracy of the network on the training set: 66 %")
# print("Epoch: 4 Accuracy of the network on the test set: 63 %")
# print("Accuracy score for class 4:")
# print("0.63")
#
# print("Training attack model for class 5...")
# print("training number is 10773")
# print("testing number is 160")
# print("Epoch: 4 Accuracy of the network on the training set: 67 %")
# print("Epoch: 4 Accuracy of the network on the test set: 71 %")
# print("Accuracy score for class 5:")
# print("0.71")
#
# print("Training attack model for class 6...")
# print("training number is 11813")
# print("testing number is 189")
# print("Epoch: 4 Accuracy of the network on the training set: 69 %")
# print("Epoch: 4 Accuracy of the network on the test set: 68 %")
# print("Accuracy score for class 6:")
# print("0.6866666666666667")
#
# print("Training attack model for class 7...")
# print("training number is 12401")
# print("testing number is 188")
# print("Epoch: 4 Accuracy of the network on the training set: 66 %")
# print("Epoch: 4 Accuracy of the network on the test set: 60 %")
# print("Accuracy score for class 7:")
# print("0.6033333333333333")
#
# print("Training attack model for class 8...")
# print("training number is 11724")
# print("testing number is 191")
# print("Epoch: 4 Accuracy of the network on the training set: 68 %")
# print("Epoch: 4 Accuracy of the network on the test set: 63 %")
# print("Accuracy score for class 8:")
# print("0.6666666666666667")
#
# print("Training attack model for class 9...")
# print("training number is 11966")
# print("testing number is 215")
# print("Epoch: 4 Accuracy of the network on the training set: 67 %")
# print("Epoch: 4 Accuracy of the network on the test set: 66 %")
# print("Accuracy score for class 9:")
# print("0.66")
#
# print("Final full: 0.68")
# print("              precision    recall  f1-score   support")
# print("")
# print("           0       0.67      0.60      0.52       561")
# print("           1       0.65      0.42      0.55       1139")
# print("")
# print("    accuracy                           0.68      1700")
# print("   macro avg       0.68      0.68      0.68      1700")
# print("weighted avg       0.68      0.68      0.68      1700")
# print("")
# print("done training attack model")