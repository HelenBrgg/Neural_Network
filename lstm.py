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

        # Define the output layer
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Perform a forward pass of the model on some input and hidden state.
        Parameters:
        x : input data of shape (batch_size, seq_len, input_dim)
        """
        # Pass the input data through the LSTM layers
        lstm_out, _ = self.lstm(x)

        # Only take the output from the final timestep
        y_pred = self.linear(lstm_out[:, -1, :])
        return y_pred
