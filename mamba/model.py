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
        
        # Create a classification head using MambaClassificationHead with input size of d_model and number of classes 2.
        self.classification_head = MambaClassificationHead(d_model=config.d_model, num_classes=2)
        
        del self.lm_head
    
    def forward(self, input_ids, attention_mask = None, labels = None):
        # Pass input_ids through the backbone model to receive hidden_states.
        hidden_states = self.backbone(input_ids)
        
        # Take the mean of hidden_states along the second dimension to create a representative [CLS] feature.
        mean_hidden_states = hidden_states.mean(dim = 1)
        
        # Pass mean_hidden_states through the classification head to get logits.
        logits = self.classification_head(mean_hidden_states)
        
        if labels is None:
            ClassificationOuptput = namedtuple("ClassificationOutput", ["logits"])
            return ClassificationOuptput(logits = logits)
        else:
            ClassificationOutput = namedtuple("ClassificationOutput", ["loss", "logits"])
            
            # Use CrossEntropyLoss loss function to compute the loss.
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
        # Load the configuration from the pre-trained model.
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        
        # Initialize the model from the configuration and move it to the desired device and data type.
        model = cls(config, device = device, dtype = dtype, **kwargs)
        
        # Load the state of the pre-trained model.
        model_state_dict = load_state_dict_hf(pretrained_model_name, device = device, dtype = dtype)
        model.load_state_dict(model_state_dict , strict=False)
        
        # Print the newly initialized embedding parameters.
        print (" Newly initialized embedding :", 
              set(model.state_dict().keys()) - set(model_state_dict.keys())
        )

        return model.to(device)