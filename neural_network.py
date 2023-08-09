import torch
import torch.nn as nn


class FCNN(nn.Module):
    def __init__(self, window_size, num_features):
        super(FCNN, self).__init__()
        # an example hidden layer size
        self.fc1 = nn.Linear(window_size * num_features, 500)
        self.fc2 = nn.Linear(500, 1)  # output layer, assuming regression task

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
