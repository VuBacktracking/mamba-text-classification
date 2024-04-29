import json
import os
from transformers import Trainer
import torch

# Define a class MambaTrainer inheriting from the Trainer class.
class MambaTrainer(Trainer):
    # Define a function compute_loss to compute the loss during training.
    def compute_loss(self, model, inputs, return_outputs=False):
        # Get the input_ids and labels values from inputs.
        input_ids = inputs.pop("input_ids") 
        labels = inputs.pop('labels')
        
        # Call the forward function of the model with input_ids and labels to get the results.
        outputs = model(input_ids=input_ids , labels=labels)
        
        # Get the loss value from the model's outputs.
        loss = outputs.loss
        
        # Return both loss and outputs if return_outputs is True, otherwise only return loss.
        return (loss, outputs) if return_outputs else loss
    
    def save_model(self, output_dir = None, _internal_call = False):
        # Check if the output directory is not specified, use the default directory from the 'args' argument.
        if output_dir is None:
            output_dir = self.args.output_dir
            
        # If the output directory does not exist, create it.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save the PyTorch model's state to the 'pytorch_model.bin' file in the output directory.
        torch.save(self.model.state_dict(), f"{output_dir}/pytorch_model.bin")
        
        # Save the tokenizer's state to the output directory.
        self.tokenizer.save_pretrained(output_dir)
        
        # Save the model's configuration to the 'config.json' file in the output directory.
        with open(f'{output_dir}/config.json', 'w') as f:  
            json.dump(self.model.config.to_dict(), f)
