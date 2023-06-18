import os
import json
import random
import logging

from itertools import chain
from copy import deepcopy

import torch

from tqdm import tqdm

from .utils.data import (
    pad_ids, truncate_sequences, truncate_sequences_dual
)

class FactLinkingDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer, split_type):
        self.args = args
        self.dataroot = args.dataroot
        # self.task_type = args.task

        self.tokenizer = tokenizer
        self.split_type = split_type

        if split_type not in ['train', 'val', 'test']:
            raise 'Dataset split_type must be one of "train", "val" or "test"'

        with open(os.path.join(self.dataroot,
                               f'neural_kg_data_{split_type}.json'), 'r') as f:
            real_data = json.load(f)
        with open(os.path.join(self.dataroot, f'fake_data_{split_type}.json'), 'r') as f:
            fake_data = json.load(f)

        self.encodings = self.tokenizer(real_data + fake_data, 
                                        truncation=True, 
                                        padding=True)

        self.labels = [1]*len(real_data) + [0]*len(fake_data)


    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) 
                    for key, val in self.encodings.items()}

        item['labels'] = torch.tensor(self.labels[index])
        return item

    
    def __len__(self):
        return len(self.labels)
