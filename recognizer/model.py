import torch
import torch.nn as nn

class DNN_Model(torch.nn.Module):
    def __init__(self, idim, odim, hidden_dim):
        torch.nn.Module.__init__(self)
        
        self.net = nn.Sequential(
            nn.Linear(idim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.output_layer = nn.Linear(hidden_dim, odim)

    def forward(self, audio_feat):
        
        # Flatten-Layer => output [BS, f len, (f dim × c dim)]
        output = torch.flatten(audio_feat, start_dim=2)
       
        output = self.net(output)
        output = self.output_layer(output)
        
        return output