import torch
import torch.nn as nn
import math


class TransformerEncoderRegressor(nn.Module):
    """
        Initialize the Transformer Encoder Regressor.

        Args:
        - num_features (int): Number of input features.
        - window_size (int): Size of the input window.
        - num_heads (int): Number of attention heads in the TransformerEncoderLayer.
        - num_layers (int): Number of layers in the TransformerEncoder.
        - dropout (float): Dropout for linear layer
        - d_model (int): Dimension of the model. Default is 512.
        - dim_feedforward (int): Dimension of the feedforward network. Default is 512.
        - dropouta (float): Dropout probability for Encoder. Default is 0.2.
    """

    def __init__(self, num_features, window_size, num_heads, num_layers, dropout, d_model=512, dim_feedforward=512, dropouta=0.2):
        super(TransformerEncoderRegressor, self).__init__()

        # Linear layer to project input features to d_model size
        self.dropout = dropout
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout)
        self.embedding = nn.Linear(num_features, d_model)
        self.dropout1 = nn.Dropout(dropout)
        # Encoder itself
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=num_heads, dropout=dropouta, dim_feedforward=dim_feedforward),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model * window_size, 1)  # Output prediction

    def forward(self, x):
        # Assuming x is of shape (batch_size, window_size, num_features)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # Transformer expects seq_len, batch, input_dim
        x = self.dropout1(x)
        transformer_output = self.transformer_encoder(x)
        transformer_output = transformer_output.permute(
            1, 0, 2)  # Convert back to batch_first
        flat_output = transformer_output.reshape(
            transformer_output.size(0), -1)
        prediction = self.fc(flat_output)

        return prediction


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
