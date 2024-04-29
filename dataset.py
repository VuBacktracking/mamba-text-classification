import random

class ImdbDataset:
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def return_train_dataset(self):
        # Perform text encoding
        train_dataset = self.data["train"].map(self.preprocess_function, batched=True)
        return train_dataset

    def return_test_dataset(self, eval_ratio = 0.1):
        random.seed(42)
        # Perform text encoding
        test_dataset = self.data["test"].map(self.preprocess_function, batched=True)
        # Create an evaluation dataset for evaluation during training
        # Due to the large number of test samples, only take a sample of 1% of the test dataset for evaluation
        total_samples = len(test_dataset)
        eval_samples = int(eval_ratio * total_samples)
        eval_indices = random.sample(range(total_samples), eval_samples)
        eval_dataset = test_dataset.select(eval_indices)
        return test_dataset, eval_dataset

    def preprocess_function(self, examples):
        samples = self.tokenizer(
            examples['text'],
            truncation=True
        )

        samples.pop('attention_mask')
        return samples