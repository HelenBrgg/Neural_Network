import torch
import torch.nn as nn


class TransformerEncoderRegressor(nn.Module):
    def __init__(self, num_features, window_size, num_heads, num_layers, dropout, d_model=512, dim_feedforward=512, dropouta=0.2):
        super(TransformerEncoderRegressor, self).__init__()

        # Linear layer to project input features to d_model size
        self.embedding = nn.Linear(num_features, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=num_heads, dropout=dropouta, dim_feedforward=dim_feedforward),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model * window_size, 1)  # Output prediction

    def forward(self, x):
        # print(model)
        # Assuming x is of shape (batch_size, window_size, num_features)
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # Transformer expects seq_len, batch, input_dim
        x = self.dropout1(x)
        transformer_output = self.transformer_encoder(x)
        transformer_output = transformer_output.permute(
            1, 0, 2)  # Convert back to batch_first
        flat_output = transformer_output.reshape(
            transformer_output.size(0), -1)
        prediction = self.fc(flat_output)

        return prediction
