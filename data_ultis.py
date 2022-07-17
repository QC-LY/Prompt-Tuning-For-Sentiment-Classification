import torch
import json
from torch.utils.data import Dataset

class ScData(Dataset):
    def __init__(self, args, data_path, label_path, tokenizer):
        self.data = json.load(open(data_path, 'r'))
        self.label = json.load(open(label_path, 'r'))
        self.tokenizer = tokenizer
        self.max_len = args.max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input = self.tokenizer.encode_plus(self.data[index], max_length=self.max_len, padding='max_length', truncation=True)
        label = self.label[index]
        input_ids = torch.tensor(input.input_ids, dtype=torch.long)
        attention_mask = torch.tensor(input.attention_mask, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)
        data_sample = (input_ids, attention_mask, label)
        return data_sample