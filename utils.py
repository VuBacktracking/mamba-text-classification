import numpy as np
import evaluate

# Load the "accuracy" module from the evaluate library.
accuracy = evaluate.load("accuracy")

# Create a preprocessing function to encode text and truncate strings longer than the maximum input token length.
def preprocess_function(tokenizer, examples):
    samples = tokenizer(examples["text"], truncation=True)
    samples.pop('attention_mask')
    return samples

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Get the index of the class with the highest probability in predictions.
    predictions = np.argmax(predictions, axis=1)
    
    # Use the "accuracy" module to compute accuracy based on predictions and labels.
    return accuracy.compute(predictions=predictions, references=labels)