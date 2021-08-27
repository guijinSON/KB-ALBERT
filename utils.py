import os 
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from transformers import AlbertConfig, AlbertForSequenceClassification
from pororo.models.brainbert import BrainRobertaModel

def load_model(PATH='/content/drive/MyDrive/KB_NLP/kb-albert-char/model'):
    tokenizer = KbAlbertCharTokenizer.from_pretrained('/content/drive/MyDrive/KB_NLP/kb-albert-char/model')
    model = AlbertForSequenceClassification.from_pretrained('/content/drive/MyDrive/KB_NLP/kb-albert-char/model')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        model = model.to(device)
    return tokenizer,model

def load_data(PATH='https://raw.githubusercontent.com/kakaobrain/KorNLUDatasets/master/KorNLI/{split}.ko.tsv',split='train'):
    if split == 'train':
        PATH = PATH.format(split='snli_1.0_train')
    elif split =='test':
        PATH = PATH.format(split='xnli.test')
    elif split =='dev':
        PATH = PATH.format(split='xnli.dev')
    csv = pd.read_csv(PATH,sep='\t')
    csv = csv.dropna()
    return csv 

def Zero_Shot_TC(sent,labels,template="이 문장은 {label}에 관한 것이다."):
    model = BrainRobertaModel.load_model("bert/brainbert.base.ko.kornli", 'ko').eval()
    cands = [template.format(label=label) for label in labels]
    result = dict()
    for label, cand in zip(labels, cands):
        tokens = model.encode(sent,cand,add_special_tokens=True,no_separator=False)
        pred = model.predict("sentence_classification_head",tokens,return_logits=True,)[:, [0, 2]]
        prob = pred.softmax(dim=1)[:, 1].item() * 100
        result[label] = round(prob, 2)
    return result

