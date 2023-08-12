import torch
import torch.nn as nn


class FCNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FCNNEncoder, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class FCNNDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FCNNDecoder, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class EncoderDecoderModel():
    def __init__(self, input_dim_enc, input_dim_dec, hidden_dim_enc, hidden_dim_dec, output_dim_enc, output_dim_dec):
        super(EncoderDecoderModel, self).__init__()

        self.input_dim_enc = input_dim_enc
        self.input_dim_dec = input_dim_dec
        self.hidden_dim_enc = hidden_dim_enc
        self.hidden_dim_dec = hidden_dim_dec
        self.output_dim_enc = output_dim_enc
        self.output_dim_dec = output_dim_dec

        encoder_model = FCNNEncoder(input_dim_enc=self.input_dim_enc,
                                    hidden_dim_enc=self.hidden_dim_enc, output_dim_enc=self.output_dim_enc)
        decoder_model = FCNNDecoder(input_dim_dec=self.input_dim_dec,
                                    hidden_dim_dec=self.hidden_dim_dec, output_dim_dec=self.output_dim_dec)

    def forward(self, x):
        encoder_output = self.encoder_model(x)
        decoder_output = self.decoder(encoder_output)
