import torch
import torch.nn as nn

class MambaClassificationHead(nn.Module):
    def __init__(self, d_model, num_classes, **kwargs):
        super(MambaClassificationHead, self).__init__()
        
        # Sử dụng một lớp tuyến tính để thực hiện phân loại dựa trên đầu vào có kích thước d_model và num_classes cần phân loại.
        self.classification_head = nn.Linear(d_model, num_classes, **kwargs)
    def forward(self, hidden_states):
        return self.classification_head(hidden_states)
    