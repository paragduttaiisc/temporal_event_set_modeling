import os, sys, math
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

from random import randint

# ------------ Configuration -----------------------

LOG = True
RUN = 1
RUN_NUM         = randint(1, 1000000)
print("Run Number: ", RUN_NUM)

D_MODEL         = 128
MAX_LEN         = 128
DATASET         = "DISEASE"     #   "INSTACART"

if DATASET == "DISEASE":
    file_path = "./data/disease_data/"
    NUM_TYPES = 211
    SCALE_TIME = 365                        # YEARLY SCALE
    TARGET_TYPES = 124
    EMBEDDING_PATH = "./pretrained_embeddings/disease_embedding.pt"
else:
    file_path = "./data/recom_dataset/"
    NUM_TYPES = 134
    SCALE_TIME = 30                         # MONTHLY SCALE
    TARGET_TYPES = 134
    EMBEDDING_PATH = "./pretrained_embeddings/instacart_embedding.pt"

BATCH_SIZE      = 512
EPOCHS          = 100
LEARNING_RATE   = 0.003
WEIGHT_DECAY    = 0
DROPOUT         = 0.1

PLOT_TSNE       = True 
SAVE_SAMPLES    = False

if torch.cuda.is_available():
    DEVICE  = torch.device("cuda:1")
else:
    DEVICE  = torch.device("cpu")

if LOG:
    wandb.init(project="ACML",
        config = {
            "dataset": DATASET,
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
    target_dict = {}
    target_events  = pickle.load(open(file_path+"events.pkl", 'rb'), encoding='latin-1')['target_id']
    print("Number of target events: ", len(target_events))

    id = 0
    for i in target_events:
        target_dict[i] = id
        id += 1
        


from torch.nn.utils.rnn import pad_sequence

# ----------- Batching the data -----------

def collate_fn(batch_data):
    batch_event_set = [[x['type_event'] for x in data[0]] for data in batch_data]

    batch_time_since_start = [[x['time_since_start'] / SCALE_TIME for x in data[0]] for data in batch_data]
    
    batch_event_set_sequence = []
    batch_event_set_sequence_mask = []
    batch_event_size = []
    for batch in batch_event_set:
        mask = 1
        event_set_sequence = [NUM_TYPES + 2]                  # [CLS]
        event_set_sequence_mask = [mask]  
        event_size = []              
        for event_set in batch:
            #if len(event_set_sequence) + len(event_set) >= MAX_LEN:
            #    break
            event_set_sequence.extend(event_set)
            event_set_sequence.append(NUM_TYPES + 1)          # [SEP] 
            event_set_sequence_mask.extend([mask]*(len(event_set)+1))
            mask += 1
            event_size.append(len(event_set)) 
        batch_event_set_sequence.append(event_set_sequence)
        batch_event_set_sequence_mask.append(event_set_sequence_mask)
        batch_event_size.append(event_size)
    
    max_seq_len = max([len(x) for x in batch_event_set_sequence])

    assert max_seq_len <= MAX_LEN, "Increase max length " + str(max_seq_len)

    padded_batch_event_set_sequence = [x + [0]*(MAX_LEN - len(x)) for x in batch_event_set_sequence]
    padded_batch_event_set_sequence_mask = [x + [0]*(MAX_LEN - len(x)) for x in batch_event_set_sequence_mask]

    max_event_size_len = max([len(x) for x in batch_event_size])
    padded_batch_event_size = [x + [0]*(max_event_size_len - len(x)) for x in batch_event_size]

    max_time_seq_len = max([len(x) for x in batch_time_since_start])
    padded_batch_time_since_start = [x + [0]*(max_time_seq_len - len(x)) for x in batch_time_since_start]

    padded_batch_event_set_sequence = torch.Tensor(padded_batch_event_set_sequence).long()
    padded_batch_event_set_sequence_mask = torch.Tensor(padded_batch_event_set_sequence_mask).long()
    padded_batch_time_since_start = torch.Tensor(padded_batch_time_since_start)
    padded_batch_event_size = torch.Tensor(padded_batch_event_size)

    target_batch_time      = torch.Tensor([data[1] / SCALE_TIME for data in batch_data])
    target_batch_event_set = [data[2] for data in batch_data]

    multi_hot_target_events = []
    for i in range(len(target_batch_event_set)):
        multi_hot = [0 for i in range(TARGET_TYPES)]
        for j in target_batch_event_set[i]:
            if DATASET == "DISEASE" and j in target_dict:
                multi_hot[target_dict[j]] = 1
            if DATASET != "DISEASE":
                multi_hot[j-1] = 1
        multi_hot_target_events.append(multi_hot)
    multi_hot_target_events = torch.Tensor(multi_hot_target_events)
    
    return padded_batch_event_set_sequence, padded_batch_time_since_start, padded_batch_event_set_sequence_mask, padded_batch_event_size, multi_hot_target_events, target_batch_time


train_loader  = DataLoader(train_data, BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True)
val_loader    = DataLoader(val_data, BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True)
test_loader   = DataLoader(test_data, BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True)

class HierarchicalNetwork(nn.Module):
    def __init__(self, embedding):
        super(HierarchicalNetwork, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
                embedding, freeze=False)

        self.sequence_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=D_MODEL, nhead=4, dim_feedforward=256, dropout=DROPOUT,
                batch_first=True), 
            num_layers=2)

        self.sequence_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=D_MODEL, nhead=4, dim_feedforward=256, dropout=DROPOUT,
                batch_first=True), 
            num_layers=2)

        self.event_set_predictor_mean = nn.Linear(D_MODEL, TARGET_TYPES)
        self.time_predictor_mean      = nn.Linear(D_MODEL, 1)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(DEVICE) # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(DEVICE)
    
    def forward(self, event_set, time, mask, set_size):
        x = self.embedding(event_set)

        #position = torch.arange(MAX_LEN).unsqueeze(1)
        pos_enc = torch.zeros(event_set.shape[0], MAX_LEN, D_MODEL)
        temp_enc = torch.zeros(event_set.shape[0], MAX_LEN, D_MODEL)                

        div_term = torch.exp(torch.arange(0, D_MODEL) * (-math.log(10000.0) / D_MODEL)).repeat(BATCH_SIZE, MAX_LEN, 1).to(DEVICE)
        #print(div_term.shape, mask.shape)
        pos_enc  = torch.zeros(event_set.shape[0], MAX_LEN, D_MODEL).to(DEVICE)
        pe = torch.sin(mask.unsqueeze(-1) * div_term)
        po = torch.cos(mask.unsqueeze(-1) * div_term)
        pos_enc[mask % 2 == 0] = pe[mask % 2 == 0]
        pos_enc[mask % 2 == 1] = po[mask % 2 == 1]
        """
        temp_enc  = torch.zeros(event_set.shape[0], MAX_LEN, D_MODEL).to(DEVICE)

        time_ext = time.unsqueeze(dim=1).repeat(1, MAX_LEN, 1)
        mask_ext = mask.unsqueeze(-1) -1
        mask_ext[mask_ext < 0] = 0
        time_sel = torch.gather(time_ext, 2, mask_ext)
        pe = torch.sin(time_sel * div_term)
        po = torch.cos(time_sel * div_term)
        temp_enc[mask % 2 == 0] = pe[mask % 2 == 0]
        temp_enc[mask % 2 == 1] = po[mask % 2 == 1]
        
        size_enc  = torch.zeros(event_set.shape[0], MAX_LEN, D_MODEL).to(DEVICE)

        set_size = time.unsqueeze(dim=1).repeat(1, 512, 1)
        mask_ext = mask.unsqueeze(-1) -1
        mask_ext[mask_ext < 0] = 0
        size_sel = torch.gather(set_size, 2, mask_ext)
        pe = torch.sin(size_sel * div_term)
        po = torch.cos(size_sel * div_term)
        size_enc[mask % 2 == 0] = pe[mask % 2 == 0]
        size_enc[mask % 2 == 1] = po[mask % 2 == 1]
        """
        x = x + pos_enc #+ temp_enc + size_enc

        #mask_pad = mask.clone()
        #mask_pad[mask_pad > 0] = 1 
        #enc_out = self.sequence_encoder(x)

        x = self.sequence_decoder(x, memory=x)

        event_set_pred_mean = self.event_set_predictor_mean(x[:, 0, :])
        time_pred_mean = self.time_predictor_mean(x[:, 0, :])

        event_set_pred = event_set_pred_mean #+ event_set_pred_std * self.N.sample(event_set_pred_mean.shape)
        time_pred = time_pred_mean # + time_pred_std * self.N.sample(time_pred_mean.shape)
        return event_set_pred, time_pred


embedding = torch.load(EMBEDDING_PATH).to(DEVICE)
model = HierarchicalNetwork(embedding.weight)
model.to(DEVICE)

optimizer     = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
#toptimizer     = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

event_set_loss_function = F.binary_cross_entropy_with_logits
time_loss_function = F.huber_loss
mae_loss_function = F.l1_loss

def dice_score(pred, target, smooth):
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

best_val_loss = float("infinity")
for ep in range(1, EPOCHS + 1):
    epoch_dice, epoch_huber, epoch_mae, epoch_loss = 0, 0, 0, 0
    itera = tqdm(train_loader)
    for event_set, time, mask, set_size, target_event_set, target_time in itera:
        event_set = event_set.to(DEVICE)
        time = time.to(DEVICE)
        mask = mask.to(DEVICE)
        set_size = set_size.to(DEVICE)
        target_event_set = target_event_set.to(DEVICE)
        target_time = target_time.to(DEVICE)

        pred_event_set, pred_time = model(event_set, time, mask, set_size)
        
        event_loss = event_set_loss_function(pred_event_set, target_event_set)
        dice_loss  = dice_score(torch.sigmoid(pred_event_set), target_event_set, 0.1)

        time_loss  = time_loss_function(pred_time.squeeze(), target_time)
        mae_loss   = mae_loss_function(pred_time.squeeze(), target_time)
        #print(event_loss.item(), time_loss.item(), dice_loss.item(), mae_loss.item())

        loss = (0.8*event_loss) + (0.3*time_loss) + (1*(1-dice_loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_dice += dice_loss.item()
        epoch_huber += time_loss.item()
        epoch_mae += mae_loss.item()

        itera.set_postfix({"dice_loss": dice_loss.item(), "bce": event_loss.item(), "huber": time_loss.item()})

    if LOG: 
        wandb.log({"train_dice": epoch_dice/len(train_loader)})
        wandb.log({"train_mae": epoch_mae/len(train_loader)})
        wandb.log({"train_loss": epoch_loss/len(train_loader)})
    print(ep, epoch_loss/len(train_loader), epoch_dice/len(train_loader), epoch_huber/len(train_loader), epoch_mae/len(train_loader))

    epoch_dice, epoch_huber, epoch_mae, epoch_loss = 0, 0, 0, 0
    itera = tqdm(val_loader)
    for event_set, time, mask, set_size, target_event_set, target_time in itera:
        event_set = event_set.to(DEVICE)
        time = time.to(DEVICE)
        mask = mask.to(DEVICE)
        set_size = set_size.to(DEVICE)
        target_event_set = target_event_set.to(DEVICE)
        target_time = target_time.to(DEVICE)

        pred_event_set, pred_time = model(event_set, time, mask, set_size)
        
        event_loss = event_set_loss_function(pred_event_set, target_event_set)
        dice_loss  = dice_score(torch.sigmoid(pred_event_set), target_event_set, 0.1)

        time_loss  = time_loss_function(pred_time.squeeze(), target_time)
        mae_loss   = mae_loss_function(pred_time.squeeze(), target_time)
        #print(event_loss.item(), time_loss.item(), dice_loss.item(), mae_loss.item())

        loss =  (0.8*event_loss) + (0.3*time_loss) + (1*(1-dice_loss))

        epoch_loss += loss.item()
        epoch_dice += dice_loss.item()
        epoch_huber += time_loss.item()
        epoch_mae += mae_loss.item()
    if LOG: 
        wandb.log({"val_dice": epoch_dice/len(val_loader)})
        wandb.log({"val_mae": epoch_mae/len(val_loader)})
        wandb.log({"val_loss": epoch_loss/len(val_loader)})
    print("[VAL]", epoch_loss/len(val_loader), epoch_dice/len(val_loader), epoch_huber/len(val_loader), epoch_mae/len(val_loader))

    if best_val_loss >= epoch_loss: # and (ep > MAX_EPOCH//2 or ep > 15):
        print("Saving Model")

        torch.save(model.state_dict(), "models/"+str(RUN_NUM)+".pt")
        best_val_loss = epoch_loss

    epoch_dice, epoch_huber, epoch_mae, epoch_loss = 0, 0, 0, 0
    itera = tqdm(test_loader)
    for event_set, time, mask, set_size, target_event_set, target_time in itera:
        event_set = event_set.to(DEVICE)
        time = time.to(DEVICE)
        mask = mask.to(DEVICE)
        set_size = set_size.to(DEVICE)
        target_event_set = target_event_set.to(DEVICE)
        target_time = target_time.to(DEVICE)

        pred_event_set, pred_time = model(event_set, time, mask, set_size)
        
        event_loss = event_set_loss_function(pred_event_set, target_event_set)
        dice_loss  = dice_score(torch.sigmoid(pred_event_set), target_event_set, 0.1)

        time_loss  = time_loss_function(pred_time.squeeze(), target_time)
        mae_loss   = mae_loss_function(pred_time.squeeze(), target_time)
        #print(event_loss.item(), time_loss.item(), dice_loss.item(), mae_loss.item())

        loss =  (0.8*event_loss) + (0.3*time_loss) + (1*(1-dice_loss))

        epoch_loss += loss.item()
        epoch_dice += dice_loss.item()
        epoch_huber += time_loss.item()
        epoch_mae += mae_loss.item()
    if LOG: 
        wandb.log({"test_dice": epoch_dice/len(test_loader)})
        wandb.log({"test_mae": epoch_mae/len(test_loader)})
        wandb.log({"test_loss": epoch_loss/len(test_loader)})
    print("[TEST]", epoch_loss/len(test_loader), epoch_dice/len(test_loader), epoch_huber/len(test_loader), epoch_mae/len(test_loader))
