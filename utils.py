import os 
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from transformers import AlbertConfig, AlbertForSequenceClassification,AutoTokenizer
from pororo.models.brainbert import BrainRobertaModel
from kbalbert.tokenization_kbalbert import KbAlbertCharTokenizer 

def load_model(PATH='/content/drive/MyDrive/KB_NLP/kb-albert-char/model'):
    tokenizer = KbAlbertCharTokenizer.from_pretrained(PATH)
    #tokenizer = AutoTokenizer.from_pretrained(PATH)
    model = AlbertForSequenceClassification.from_pretrained(PATH,num_labels=3)
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
    csv = csv.dropna(axis=0,how='any').reset_index(drop=True)
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

def binary_score(score,label1,label2):
    text=''
    if score > 50:
        text = f'강한 {label1}'
    elif score>0:
        text = f'약한 {label1}'
    elif -50<score<0:
        text = f'약한 {label2}'
    elif score<-50:
        text = f'강한 {label2}'
    return text

def increase_font():
  from IPython.display import Javascript
  display(Javascript('''
  for (rule of document.styleSheets[0].cssRules){
    if (rule.selectorText=='body') {
      rule.style.fontSize = '17px'
      break
    }
  }
  '''))
