import torch
import torch.nn as nn
import torch.nn.functional as F

class Classification(torch.nn.Module):
    def __init__(self, idim=39, odim=1, hidden_dim=512):
        torch.nn.Module.__init__(self)
        # Define three fully connected layers followed by ReLU activation funtion
        self.net = nn.Sequential(
            nn.Linear(idim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Define a fully connected output (classification) layer
        # Sigmoid function computes probability
        self.clf = nn.Sequential(
            nn.Linear(hidden_dim, odim),
            nn.Sigmoid()
        )

    def forward(self, audio_feat):
        """
        Input: 
            audio_feat: <tensor.FloatTensor> the audio features in a tensor
        Return: 
            The predicted posterior probabilities
        """
  
        # Reshaping as linear layer only accepts inputs of size (n, idim)
        BS, f_len, idim = audio_feat.shape
        feat = audio_feat.view(BS * f_len, idim)

        # Features mapped by 3 fully connected layers
        mapped_feat = self.net(feat)
        mapped_feat = mapped_feat.view(BS, f_len, -1)
        
        # Average the mapped representation over the sequence dimension
        avg_mapped_feat = torch.mean(mapped_feat, dim=1)

        # Return the predicted probabilities by classification layer and Sigmoid function.
        return self.clf(avg_mapped_feat)

