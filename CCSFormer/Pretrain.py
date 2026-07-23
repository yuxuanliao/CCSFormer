import pandas as pd
from pandas import *
import numpy as np
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import torch
import torch.nn as nn
import torch.optim as optim
import random

from rdkit.Chem import Descriptors
from rdkit.Chem.Descriptors import ExactMolWt

from PubChemFP import GetPubChemFPs
from rdkit.Chem import AllChem,MACCSkeys
from rdkit.Chem import rdFingerprintGenerator
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import dill
with open('../data/vocab.pkl', 'rb') as f:
    dictionary= dill.load(f)

data = pd.read_csv('../data/Pretraining dataset.csv')
data.head()

dsmiles = list(data['SMILES'])
smiles_list = dsmiles
print(len(smiles_list))

ssmiles_list = []
for smiles in tqdm(smiles_list):
    mol = Chem.MolFromSmiles(smiles)
    smi = Chem.MolToSmiles(mol)  # standardize
    ssmiles_list.append(smi)

def mws_list(smiles):
    mw = []
    for i in tqdm(range(len(smiles))):
        smi = smiles[i]
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        cal_mass = ExactMolWt(mol)
        mw.append(cal_mass)
    return mw
        
mw_list = mws_list(ssmiles_list)

def mol_to_sentence(mol: Chem.Mol, radius: int = 1) -> list:
    info = {}
    AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)

    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]
    atom_dic = {x: {r: None for r in range(radius + 1)} for x in mol_atoms}

    for element in info:
        for atom_idx, r in info[element]:
            atom_dic[atom_idx][r] = element

    sentence = []
    for atom in atom_dic:
        if atom_dic[atom][radius]:
            sentence.append(str(atom_dic[atom][radius]))
    return sentence

def token_smiles(smi, vocab):
    mol = Chem.MolFromSmiles(smi)
    sentence = mol_to_sentence(mol, radius=1)
    token_ids = [
        vocab.stoi.get(token, vocab.unk_index) for token in sentence
    ]
    return token_ids

def mask_tokens(tokens, vocab, mask_prob=0.15):
    masked_tokens = []
    labels = []
    
    for token in tokens:
        if np.random.rand() < mask_prob:
            rand_num = np.random.rand()
            if rand_num < 0.8: 
                masked_tokens.append(vocab.mask_index)
                labels.append(token)
            elif rand_num < 0.9: 
                masked_tokens.append(np.random.choice(list(vocab.stoi.values())))
                labels.append(token)
            else: 
                masked_tokens.append(token)
                labels.append(token) 
        else:
            masked_tokens.append(token)
            labels.append(-100)
            
    return masked_tokens, labels


class PretrainDataset(Dataset):
    def __init__(self, smiles_list, mws, vocab, max_len=250, mask_prob=0.15):
        self.smiles_list = smiles_list
        self.mws = mws
        self.vocab = vocab
        self.max_len = max_len
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        tokens = token_smiles(smiles, self.vocab)
        masked_tokens, labels = mask_tokens(tokens, self.vocab, self.mask_prob)
        input_ids = [self.vocab.cls_index]+masked_tokens[:self.max_len-2] + [self.vocab.sep_index] +[self.vocab.pad_index] * (self.max_len-2 - len(masked_tokens))
        labels = [-100]+labels[:self.max_len-2]+ [-100] + [-100] * (self.max_len-2 - len(labels))
        
        molecular_weights = torch.tensor(self.mws[idx], dtype=torch.float)
        
        return torch.tensor(input_ids), torch.tensor(labels), molecular_weights

from sklearn.model_selection import train_test_split

smiles_train, smiles_temp, mw_train, mw_temp = train_test_split(
    ssmiles_list, mw_list, test_size=0.2, random_state=42)

smiles_val, smiles_test, mw_val, mw_test = train_test_split(
    smiles_temp, mw_temp, test_size=0.5, random_state=42)

dataset_train = PretrainDataset(smiles_train, mw_train, dictionary)
dataset_val = PretrainDataset(smiles_val, mw_val, dictionary)
dataset_test = PretrainDataset(smiles_test, mw_test, dictionary)

batch_size = 128
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, en_layers, ff_dim, max_len):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, embed_dim, padding_idx=0)
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first = True
            ),
            num_layers = en_layers
        )
        
        self.masked_fc = nn.Linear(embed_dim, vocab_size)
        self.nb_fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        seq_len = x.size(1)
        position_ids = torch.arange(0, seq_len).unsqueeze(0).to(x.device)
        embedded = self.embedding(x) + self.position_embedding(position_ids)
        bert_out = self.encoder(embedded)

        masked_pred = self.masked_fc(bert_out)
        nb_pred = self.nb_fc(bert_out[:, 0, :])
        
        return masked_pred, nb_pred

model = TransformerModel(
    vocab_size = 14073,
    embed_dim = 128,
    num_heads = 4,
    en_layers = 2,
    ff_dim = 512,
    max_len = 250
).to(device)

import os

def save_model(model, epoch, save_dir="models-1"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"model_epoch_{epoch}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at epoch {epoch}: {save_path}")

def pretrain(model, dataloader_train, dataloader_val, optimizer, criterion_masked,
                 epochs=150, save_interval=10, save_dir="models", loss_file = "loss.csv"):

    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []
    
    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        train_loss = 0
        for input_ids, labels, labels_mw in dataloader_train:
            optimizer.zero_grad()
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            labels_mw = labels_mw.to(device)
            
            outputs_masked, outputs_mw = model(input_ids)  # [batch_size, seq_len, vocab_size]
            outputs_masked = outputs_masked.view(-1, outputs_masked.size(-1))  # 展平
            labels = labels.view(-1)
            loss_masked = criterion_masked(outputs_masked, labels)
            
            loss_mw = F.huber_loss(outputs_mw.squeeze(), labels_mw)
            loss_mw_norm = (loss_mw - 0.0)/(10.0 - 0.0)
            
            loss = 0.7*loss_masked + 0.3*loss_mw_norm
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(dataloader_train)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_ids, labels, labels_mw in dataloader_val:
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                labels_mw = labels_mw.to(device)

                outputs_masked, outputs_mw = model(input_ids)  # [batch_size, seq_len, vocab_size]
                outputs_masked = outputs_masked.view(-1, outputs_masked.size(-1))
                labels = labels.view(-1)
                loss_masked = criterion_masked(outputs_masked, labels)
                
                loss_mw = F.huber_loss(outputs_mw.squeeze(), labels_mw)
                loss_mw_norm = (loss_mw - 0.0)/(10.0 - 0.0)
                
                loss = 0.7*loss_masked + 0.3*loss_mw_norm

                val_loss += loss.item()
        val_loss /= len(dataloader_val)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if epoch % save_interval == 0:
            save_model(model, epoch, save_dir)

    loss_data = {
        "Epoch": list(range(1, epochs + 1)),
        "Train Loss": train_losses,
        "Val Loss": val_losses,
    }
    df = pd.DataFrame(loss_data)
    df.to_csv(loss_file, index=False)

    print(f"Training log saved to {loss_file}")

    model.load_state_dict(best_model_state)
    return model

optimizer_pretrain = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion_masked = nn.CrossEntropyLoss(ignore_index=-100)

pretrain(model, dataloader_train, dataloader_val, optimizer_pretrain, criterion_masked, epochs=250, save_interval=10, 
             save_dir="model-1", loss_file = "loss-1.csv")


def load_model(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Loaded model from {checkpoint_path}")
    return model


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, r2_score
from numpy import mean, median, abs, sum, cumsum, histogram, sqrt

def metrics(y_true, y_pred):   #r2_score(y_true, y_pred)
    RelativeError = [abs(y_pred[i]-y_true[i])/y_true[i] for i in range(len(y_true))]
    R2_Score = r2_score(y_true, y_pred)
    abs_y_err = [abs(y_pred[i]-y_true[i]) for i in range(len(y_true))]
    mae = mean(abs_y_err)
    mdae = median(abs_y_err)
    MSE = mean_squared_error(y_true, y_pred)
    RMSE = sqrt(mean_squared_error(y_true, y_pred))
    print("R2 Score :", round(R2_Score, 3))
    print("Mean Absolute Error :", round(mae, 2))
    print("Median Absolute Error :", round(mdae, 2))
    print("Median Relative Error :", round(np.median(RelativeError) * 100, 2), '%')
    print("Mean Relative Error :", round(np.mean(RelativeError) * 100, 2), '%')
    print("Root Mean Squared Error :", round(RMSE, 2))
    print("Mean Squared Error :", round(MSE, 2))
    return R2_Score, np.median(RelativeError) * 100

def evaluate_mlm(model, dataloader, device = "cuda"):
    model.eval()
    all_targets = []
    all_predictions = []
    total_tokens = 0
    correct_predictions = 0

    criterion_masked = nn.CrossEntropyLoss(ignore_index=-100)
            
    with torch.no_grad():
        for input_ids, labels, labels_mw in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            labels_mw = labels_mw.to(device)

            outputs_masked, outputs_mw = model(input_ids)  # [batch_size, seq_len, vocab_size]
            outputs_masked = outputs_masked.view(-1, outputs_masked.size(-1))
            labels = labels.view(-1)  
            loss_masked = criterion_masked(outputs_masked, labels)
            
            loss_mw = F.huber_loss(outputs_mw.squeeze(), labels_mw)
            loss_mw_norm = (loss_mw - 0.0)/(10.0 - 0.0)
            
            loss = 0.7*loss_masked + 0.3*loss_mw_norm

            predictions = torch.argmax(outputs_masked, dim = -1)
            label_tokens = (labels != -100)
            correct_predictions += (predictions[label_tokens] == labels[label_tokens]).sum().item()
            total_tokens += label_tokens.sum().item()
            all_predictions.extend(outputs_mw.squeeze().tolist())
            all_targets.extend(labels_mw.tolist())
    
    # 转为张量
    all_targets = torch.tensor(all_targets)
    all_predictions = torch.tensor(all_predictions)

    r2, medre = metrics(all_targets, all_predictions)
    accuracy = correct_predictions / total_tokens if total_tokens > 0 else 0.0
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy, r2, medre

checkpoint_path = "Pretraining_model.pth"
model = load_model(model, checkpoint_path)
accuracy = evaluate_mlm(model, dataloader_test, device = "cuda")