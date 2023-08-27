import torch
import torch.nn as nn


class FCNN(nn.Module):
    def __init__(self, window_size, num_features):
        super(FCNN, self).__init__()
        self.num_features = num_features
        self.window_size = window_size
        # an example hidden layer size
        self.fc1 = nn.Linear(window_size * num_features,
                             window_size)
        self.dropout1 = nn.Dropout(0.1)
        self.batch_norm1 = nn.BatchNorm1d(window_size)
        self.fc2 = nn.Linear(window_size, 100)
        self.batch_norm2 = nn.BatchNorm1d(100)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(100, 80)
        self.batch_norm3 = nn.BatchNorm1d(80)
        self.dropout3 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(80, 50)
        self.batch_norm4 = nn.BatchNorm1d(50)
        self.dropout4 = nn.Dropout(0.1)

        self.fc5 = nn.Linear(50, 1)

    def forward(self, x):
        # Flatten the input here if needed
        print("Original shape of x:", x.shape)
        x = x.reshape(-1, self.window_size * self.num_features)
        x = self.dropout1(torch.relu(self.batch_norm1(self.fc1(x))))
        x = self.dropout2(torch.relu(self.batch_norm2(self.fc2(x))))
        x = self.dropout3(torch.relu(self.batch_norm3(self.fc3(x))))
        x = self.dropout4(torch.relu(self.batch_norm4(self.fc4(x))))

        return self.fc5(x)
