import os 
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)
from transformers import AlbertConfig, AlbertForSequenceClassification
from utils import load_model
from data import Dataset

#### For Colab ####
#from google.colab import drive
#drive.mount('/content/drive')
#os.chdir('/content/drive/MyDrive/KB_NLP/kb-albert-char')
#from tokenization_kbalbert import KbAlbertCharTokenizer

tokenizer,model = load_model()
train = load_data()
dev = load_data(split='dev')

train_ds = Dataset(train['sentence1'].values.tolist(),train['sentence1'].values.tolist(),train['gold_label'].values,tokenizer)
dataloader = DataLoader(train_ds, batch_size=128, shuffle=True)

dev_ds = Dataset(dev['sentence1'].values.tolist(),dev['sentence1'].values.tolist(),dev['gold_label'].values,tokenizer)
dev_dataloader = DataLoader(dev_ds, batch_size=128, shuffle=True)

epochs = 10
model_num = 5
device = torch.device('cuda:0')
loss_func = nn.BCEWithLogitsLoss(reduction='sum')
optimizer = torch.optim.Adam([{'params': model.albert.parameters(), 'lr':1e-5},
                              {'params': model.classifier.parameters(), 'lr':5e-3}])

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = len(dataloader)/4, gamma=0.3)

for iter in range(epochs):
    n = 0 
    for x,y in tqdm(dataloader): 
        input_ids = x['input_ids'].squeeze().to(device)
       # attention_mask = x['attention_mask'].to(device)
       # token_type_ids = x['token_type_ids'].to(device)
        y = y.type(torch.FloatTensor).to(device)
        output = model(input_ids).logits.squeeze()
       # output = model(input_ids,attention_mask=attention_mask,token_type_ids = token_type_ids)
        loss = loss_func(output,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if n % 200 ==0:
            with torch.no_grad():
                accuracy = 0
                tot = 0
                for x,y in dev_dataloader:
                    x = x['input_ids'].squeeze().to(device)
                    if len(x.shape) == 3:
                        x = x.squeeze()
                    y = y.to(device)
                    logits = model(x).logits
                    acc = [1 if n >0.5 else 0 for n in logits.squeeze().cpu().numpy()]
                    print(acc - y.cpu().numpy())
                    tot += len(y)
            print(f'{iter} Epoch Running | {n} Step | Loss: {loss.detach().item():.3f} | Accuracy: {accuracy/tot*100 :.2f}')
        n+=1
    torch.save(model.state_dict(), f'/content/drive/MyDrive/KB_NLP/model_{model_num}_{iter}.pth')
