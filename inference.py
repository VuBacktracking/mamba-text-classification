from mamba.model import MambaTextClassification
from transformers import AutoTokenizer
from datasets import load_dataset

# Load IMDb dataset
imdb = load_dataset("imdb")

# Load the pre-trained Mamba model
model = MambaTextClassification.from_pretrained("vubacktracking/mamba_text_classification")
model.to("cuda")

# Load the tokenizer associated with the Mamba model
tokenizer = AutoTokenizer.from_pretrained("vubacktracking/mamba_text_classification")

# Set pad token id to eos token id
tokenizer.pad_token_id = tokenizer.eos_token_id

# Mapping from label id to label
id2label = {0: "NEGATIVE", 1: "POSITIVE"}

# Select a sample text from the test set
sample_text = imdb['test'][100]['text']
sample_label = imdb['test'][100]['label']

# Predict using the model
predicted_label = model.predict(sample_text, tokenizer, id2label)

# Print the results
print(f'Text: {sample_text}\nGround Truth: {id2label[sample_label]}\nPredicted: {predicted_label}')