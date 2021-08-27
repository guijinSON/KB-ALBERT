from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class TrainDataset(Dataset):
    def __init__(self, sentence1, sentence2, label,tokenizer):
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.label = label
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        x1 = self.sentence1[idx]
        x2 = self.sentence2[idx]
        if self.label[idx] == 'entailment':
            y = 0 
        elif self.label[idx] == 'neutral':
            y = 1
        elif self.label[idx] == 'contradiction':
            y = 2 

        return self.tokenize(x1,x2), y

    def tokenize(self,sentence1,sentence2):
        return self.tokenizer(sentence1,sentence2, max_length=64,padding="max_length", truncation=True, return_tensors='pt')['input_ids']
