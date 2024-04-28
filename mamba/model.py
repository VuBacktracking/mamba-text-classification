import numpy as np
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.hf import load_config_hf,load_state_dict_hf
from collections import namedtuple
import torch.nn as nn
import torch

from cfg.config import MambaConfig
from mamba.head import MambaClassificationHead

class MambaTextClassification(MambaLMHeadModel):
    def __init__(
        self,
        config: MambaConfig,
        initializer_cfg = None,
        device = None,
        dtype = None,
    ) -> None:
        super().__init__(config, initializer_cfg, device, dtype)
        
        # Tạo một đầu phân loại sử dụng MambaClassificationHead với kích thước đầu vào là d_model và số lớp là 2.
        self.classification_head = MambaClassificationHead(d_model=config.d_model, num_classes=2)
        
        del self.lm_head
    
    def forward(self, input_ids, attention_mask = None, labels = None):
        # Truyền input_ids qua model gốc để nhận hidden_states.
        hidden_states = self.backbone(input_ids)
        
        # Lấy trung bình của hidden_states theo chiều thứ 2 để tạo ra [CLS] feature đại điện
        mean_hidden_states = hidden_states.mean(dim = 1)
        
        # Đưa mean_hidden_states qua đầu phân loại để nhận logits.
        logits = self.classification_head(mean_hidden_states)
        
        if labels is None:
            ClassificationOuptput = namedtuple("ClassificationOutput", ["logits"])
            return ClassificationOuptput(logits = logits)
        else:
            ClassificationOutput = namedtuple("ClassificationOutput", ["loss", "logits"])
            
            # Sử dụng hàm mất mát CrossEntropyLoss để tính loss.
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
            return ClassificationOuptput(loss = loss, logits = logits)
    def predict(self, text, tokenizer, id2label = None):
        input_ids = torch.tensor(tokenizer(text)['input_ids'], device = "cuda")[None]
        with torch.no_grad():
            logits = self.forward(input_ids).logits[0]
            label = np.argmax(logits.cpu().numpy())
            
        if id2label is not None:
            return id2label[label]
        else:
            return label
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name, device = None, dtype = None, **kwargs):
        # Tải cấu hình từ model đã được train trước đó.
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        
        # Khởi tạo model từ cấu hình và chuyển nó đến thiết bị và kiểu dữ liệu mong muốn.
        model = cls(config, device = device, dtype = dtype, **kwargs)
        
        # Tải trạng thái model đã được train trước đó.
        model_state_dict = load_state_dict_hf(pretrained_model_name, device = device, dtype = dtype)
        model.load_state_dict(model_state_dict , strict=False)
        
        # In ra các tham số embedding mới được khởi tạo.
        print("Newly initialized embedding:", set(model.state_dict() .keys()) - set(model_state_dict.keys()))
        return model