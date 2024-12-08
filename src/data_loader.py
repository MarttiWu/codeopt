import os
import jsonlines
from torch.utils.data import Dataset, DataLoader


class CodeOptimizationDataset(Dataset):
    """
    Custom dataset for loading code optimization examples.
    Each sample contains a query (slow code) and a target (optimized code).
    """
    def __init__(self, file_path, tokenizer=None, max_seq_length=512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # Load data from JSONL file
        with jsonlines.open(file_path, 'r') as reader:
            for obj in reader:
                self.samples.append(obj)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        query = sample["query"]
        target = sample["reference"]

        if self.tokenizer:
            # Tokenize query and target
            query_tokens = self.tokenizer(
                query,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            target_tokens = self.tokenizer(
                target,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            return {
                "input_ids": query_tokens["input_ids"].squeeze(0),
                "attention_mask": query_tokens["attention_mask"].squeeze(0),
                "labels": target_tokens["input_ids"].squeeze(0),
                "query": query,
                "reference": target,
            }
        else:
            # Return raw text when tokenizer is not available
            return {
                "query": query,
                "reference": target,
            }


def create_data_loaders(train_file, test_file, tokenizer=None, batch_size=32, max_seq_length=512):
    """
    Create DataLoaders for training and test datasets.
    """
    train_dataset = CodeOptimizationDataset(train_file, tokenizer, max_seq_length)
    test_dataset = CodeOptimizationDataset(test_file, tokenizer, max_seq_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader