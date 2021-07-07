from torch import nn
import torch

class Encoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, num_layers=2, \
        use_rnn=True, use_gru=True, use_lstm=True):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size =input_size
        self.output_size = output_size
        self.use_rnn = use_rnn
        self.use_gru = use_gru
        self.use_lstm = use_lstm
        if self.use_rnn:
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq, input_size)
        # or:
        if self.use_gru:
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        if self.use_lstm:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        if self.use_lstm:
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        # Forward propagate RNN
        if self.use_rnn:
            out, _ = self.rnn(x, h0)  
        # or:
        if self.use_gru:
            out, _ = self.gru(x, h0)
        if self.use_lstm:
            out, _ = self.lstm(x, (h0,c0))  
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out: (n, 128)
        out = self.fc(out)
        # out: (n, 10)
        return out