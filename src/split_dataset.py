import json
import random

def split_dataset(input_file, train_file, test_file, split_ratio=0.8):
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]

    random.shuffle(data)

    split_index = int(len(data) * split_ratio)
    train_data = data[:split_index]
    test_data = data[split_index:]

    with open(train_file, 'w') as train_f:
        for item in train_data:
            train_f.write(json.dumps(item) + '\n')

    with open(test_file, 'w') as test_f:
        for item in test_data:
            test_f.write(json.dumps(item) + '\n')

input_file = "processed_data/python/test.jsonl"
train_file = "processed_data/python/train_split.jsonl"
test_file = "processed_data/python/test_split.jsonl"
split_dataset(input_file, train_file, test_file, split_ratio=0.8)