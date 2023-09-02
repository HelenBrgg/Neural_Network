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
                            num_layers, batch_first=True)
        self.relu = nn.ReLU()

        self.fc_1 = nn.Linear(hidden_dim, hidden_dim//2)  # fully connected
        # fully connected last layer
        self.fc_2 = nn.Linear(hidden_dim//2, output_dim)

        # Define the output layer

    def forward(self, x):
        """
        Perform a forward pass of the model on some input and hidden state.
        Parameters:
        x : input data of shape (batch_size, seq_len, input_dim)
        """
        # Pass the input data through the LSTM layers
        lstm_out, _ = self.lstm(x)
        out = self.fc_1(lstm_out)  # first dense

        #out = self.fc_2(out)

        # Only take the output from the final timestep
        y_pred = self.fc_2(out[:, -1, :])
        return y_pred
