import os, sys
import random
from functools import partial

import csv
import numpy as np
import matplotlib.pyplot as plt

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import datasets, transforms
from torchvision.datasets import Omniglot

# SkLearn
from sklearn.metrics import accuracy_score

# Logging
import wandb 
from tqdm import tqdm 

# Load and Save datasets
import pickle

# ------------ Configuration -----------------------

LOG = False
RUN = 1

DATASET         = "INSTACART"

if DATASET == "DISEASE":
    file_path = "./data/disease_data/"
    NUM_TYPES = 211
else:
    file_path = "./data/recom_dataset/"
    NUM_TYPES = 134

BATCH_SIZE      = 128
EPOCHS          = 15
LEARNING_RATE   = 0.001
WEIGHT_DECAY    = 0.01
DROPOUT         = 0

PLOT_TSNE       = True 
SAVE_SAMPLES    = False

if torch.cuda.is_available():
    DEVICE  = torch.device("cuda:0")
else:
    DEVICE  = torch.device("cpu")

if LOG:
    wandb.init(project="ADRL_A3_Q3",
        config = {
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "run": RUN,
        }
    )

# ------------ Data Class --------------------------

train_data = pickle.load(open(file_path+"train.pkl", 'rb'), encoding='latin-1')
val_data   = pickle.load(open(file_path+"dev.pkl", 'rb'), encoding='latin-1')
test_data  = pickle.load(open(file_path+"test.pkl", 'rb'), encoding='latin-1')

print(len(train_data), train_data[1][0], train_data[1][1])
if DATASET == "DISEASE":
    target_events  = pickle.load(open(file_path+"events.pkl", 'rb'), encoding='latin-1')['target_id']

from torch.nn.utils.rnn import pad_sequence

# ----------- Batching the data -----------

def collate_fn(batch_data):
    batch_event_set = [[x['type_event'] for x in data[0]] for data in batch_data]

    anchor, positive, negative = [], [], []
    for event_set_samples in batch_event_set:
        for event_set in event_set_samples:
            anchor_sample, positive_sample = random.choices(event_set, k=2)

            negative_sample = random.randint(1, NUM_TYPES)
            tries = 10
            while negative_sample in event_set and tries > 0:
                negative_sample = random.randint(1, NUM_TYPES)
                tries -= 1
            
            anchor.append(anchor_sample)
            positive.append(positive_sample)
            negative.append(negative_sample)

    anchor = torch.Tensor(anchor)
    positive = torch.Tensor(positive)
    negative = torch.Tensor(negative)
    return anchor, positive, negative


train_loader  = DataLoader(train_data, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader    = DataLoader(val_data, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader   = DataLoader(test_data, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

class EmbeddingNetwork(nn.Module):
    def __init__(self, embedding=None, freeze=False, random_init=False):
        super(EmbeddingNetwork, self).__init__()
        self.embedding = nn.Embedding(NUM_TYPES+4, 128, padding_idx=0)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, anc, pos, neg):
        x = self.embedding(anc)
        x = self.fc1(x)
        x = F.relu(x)
        anc_emb = self.fc2(x)
        
        x = self.embedding(pos)
        x = self.fc1(x)
        x = F.relu(x)
        pos_emb = self.fc2(x)

        x = self.embedding(neg)
        x = self.fc1(x)
        x = F.relu(x)
        neg_emb = self.fc2(x)

        return anc_emb, pos_emb, neg_emb

embedder = EmbeddingNetwork()
embedder.to(DEVICE)

print(embedder)

loss_function = nn.BCELoss()
optimizer     = optim.Adam(embedder.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

for ep in range(1, EPOCHS+1):
    epoch_loss = 0
    for anc, pos, neg in tqdm(train_loader):
        anc = anc.to(DEVICE).long()
        pos = pos.to(DEVICE).long()
        neg = neg.to(DEVICE).long()
        
        anc_emb, pos_emb, neg_emb = embedder(anc, pos, neg)

        pos_rep = torch.sigmoid(anc_emb * pos_emb)
        neg_rep = torch.sigmoid(anc_emb * neg_emb)

        pos_labels = torch.ones_like(pos_rep)
        neg_labels = torch.zeros_like(neg_rep)

        loss = loss_function(pos_rep, pos_labels)
        loss += loss_function(neg_rep, neg_labels)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print("Epoch: ", ep, "Loss: ", epoch_loss/len(train_loader))

torch.save(embedder.embedding, "instacart_embedding.pt")