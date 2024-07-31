import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

input_file_path ="input.txt"
batch_size=32
block_size=8
n_embd=32
max_iterval=5000
learning_rate=0.0001
device='cpu'
device='cuda:5' if torch.cuda.is_available() else 'cpu'

eval_iterval=200
torch.manual_seed=(1337)


with open(input_file_path, 'r') as f:
    text = f.read()
print(f"length of dataset in characters: {len(text):,}")
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

data= torch.tensor(encode(text),dtype=torch.long)
n=int(0.9*len(data))
train_data=data[:n]
val_data=data[n:]



def get_batch(split):
    #任意找block_size个连续字符
    data=train_data if split=='train' else val_data
    ix=torch.randint(len(data)-block_size,(batch_size,))
    x=torch.stack([data[i:i+block_size] for i in ix])   
    y=torch.stack([data[i+1:i+block_size+1] for i in ix]) 
    x,y=x.to(device),y.to(device)
    return x,y

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,n_embd)#最终输出是一个向量，表示可能是每个字母的概率
        self.position_embedding_table=nn.Embedding(block_size,n_embd) 
        self.lm_head=nn.Linear(n_embd,vocab_size)

    def forward(self,idx,targets=None):
        B,T=idx.shape
        tok_emb=self.token_embedding_table(idx) #(B,T,C)
        pos_emb=self.position_embedding_table(torch.arange(T,device=device))
        x=pos_emb+tok_emb
        logits=self.lm_head(x) #(B,T,vocab_size)
        
        if targets is None:
            loss=None
        else:
            B,T,C=logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits,targets)
        return logits,loss
    
    def generate(self,idx,max_new_tokens):
        #idx is (B,T)
        for _ in range(max_new_tokens):
            logits,loss=self(idx)
            logits=logits[:,-1,:]
            probs=F.softmax(logits,dim=-1)
            idx_next=torch.multinomial(probs,num_samples=1)
            idx=torch.cat((idx,idx_next),dim=1)
        return idx


model=BigramLanguageModel()
model=model.to(device)
optimizer=torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    out={}
    model.eval()
    for split in ['train','val']:
        losses=torch.zeros(eval_iterval)
        for k in range(eval_iterval):
            xb,yb=get_batch(split)
            logits,loss=model(xb,yb)
            losses[k]=loss.item()
        out[split]=losses.mean()
    model.train()
    return out

for iter in range(max_iterval):
    if  iter % eval_iterval==0:
        losses=estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb,yb=get_batch('train')
    logits,loss=model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

contenxt=torch.zeros((1,1),dtype=torch.long,device=device)
print(decode(model.generate(contenxt,max_new_tokens=500)[0].tolist()))
