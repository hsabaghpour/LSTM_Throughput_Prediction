# models/attention_lstm.py
import torch
import torch.nn as nn

class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def attention(self, lstm_output, hidden):
        hidden = hidden.squeeze(0)  # remove batch size dim
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        attn_weights = torch.softmax(attn_weights, dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        return attn_applied

    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        attn_applied = self.attention(lstm_out, hidden[-1])
        output = torch.cat((attn_applied, hidden[-1]), dim=1)
        output = torch.relu(self.attn_combine(output))
        output = self.out(output)
        return output
