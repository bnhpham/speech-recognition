import torch
import torch.nn as nn

class LSTM_Model(torch.nn.Module):
    def __init__(self, idim, odim, hidden_dim):
        torch.nn.Module.__init__(self)

        self.lstm = nn.LSTM(
            input_size=idim,
            hidden_size=hidden_dim,
            num_layers= 3,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
            )

        self.output_layer = nn.Linear(hidden_dim * 2, odim)

    def forward(self, audio_feat):
        output = torch.flatten(audio_feat, start_dim=2)
       
        output, _ = self.lstm(output)
        output = self.output_layer(output)
        
        return output