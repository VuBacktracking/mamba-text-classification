import torch
import torch.nn as nn

class MambaClassificationHead(nn.Module):
    def __init__(self, d_model, num_classes, **kwargs):
        super(MambaClassificationHead, self).__init__()
        
        # Use a linear layer to perform classification based on the input with size d_model and the number of classes to classify num_classes.
        self.classification_head = nn.Linear(d_model, num_classes, **kwargs)
        
    def forward(self, hidden_states):
        return self.classification_head(hidden_states)