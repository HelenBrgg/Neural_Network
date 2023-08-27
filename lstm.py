import torch.nn as nn
import torch.nn.functional as F


class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        """
        Initialize the model by setting up the layers.

        Parameters:
        input_dim : size of input feature dimension
        hidden_dim : size of hidden layer dimension
        num_layers : number of LSTM layers
        output_dim : size of output dimension
        """
        super(SimpleLSTM, self).__init__()

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers, batch_first=True, dropout=0.1)

        # Dropout layer for LSTM output
        self.lstm_dropout = nn.Dropout(0.1)

        # Intermediate fully connected layer (optional)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)

        # Dropout layer for fully connected layer
        self.fc_dropout = nn.Dropout(0.1)

        # Batch normalization layer
        self.bn = nn.BatchNorm1d(hidden_dim // 2)

        # Define the output layer
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        """
        Perform a forward pass of the model on some input and hidden state.

        Parameters:
        x : input data of shape (batch_size, seq_len, input_dim)
        """
        lstm_out, _ = self.lstm(x)
        x = self.lstm_dropout(lstm_out[:, -1, :])
        x = F.relu(self.fc1(x))
        x = self.fc_dropout(x)
        x = self.bn(x)
        y_pred = self.fc2(x)
        return y_pred
