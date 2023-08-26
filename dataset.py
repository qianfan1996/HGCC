import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from util import config


tokenizer = BertTokenizer.from_pretrained(config['bert_path'])

class MoseiDataset(Dataset):
    def __init__(self, data_dic, device, mode="train"):
        self.data_dic = data_dic[mode]
        dic = tokenizer(list(self.data_dic['raw_text']), padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        self.input_ids = dic["input_ids"]
        self.attention_mask = dic["attention_mask"]
        self.audio = self.data_dic['audio']
        self.video = self.data_dic['vision']
        self.label = self.data_dic['regression_labels']
        self.device = device

    def _to_tensor(self, input_ids, attention_mask, audio, video, label):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        audio = torch.FloatTensor(audio).to(self.device)
        video = torch.FloatTensor(video).to(self.device)
        label = torch.FloatTensor([label]).to(self.device)
        return input_ids, attention_mask, audio, video, label

    def __getitem__(self, idx):
        input_ids, attention_mask, audio, video, label = self.input_ids[idx], self.attention_mask[idx], self.audio[idx], self.video[idx], self.label[idx]
        input_ids, attention_mask, audio, video, label = self._to_tensor(input_ids, attention_mask, audio, video, label)

        return input_ids, attention_mask, audio, video, label

    def __len__(self):
        return len(self.data_dic['raw_text'])