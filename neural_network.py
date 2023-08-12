import torch
import torch.nn as nn


class FCNN(nn.Module):
    def __init__(self, window_size, num_features):
        super(FCNN, self).__init__()
        self.num_features = num_features
        # an example hidden layer size
        self.fc1 = nn.Linear(window_size * num_features, 500)
        self.dropout1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(500, 250)
        self.batch_norm2 = nn.BatchNorm1d(250)
        self.dropout2 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(250, 1)

    def forward(self, x):
        # Flatten the input here if needed
#        x = x.reshape(-1, self.num_features)
        x = self.dropout1(torch.relu(self.fc1(x)))
        x = self.dropout2(torch.relu(self.batch_norm2(self.fc2(x))))
        return self.fc3(x)
