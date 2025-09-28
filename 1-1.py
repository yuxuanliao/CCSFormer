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

data = pd.read_csv(r'e:\250811\code\1\2\Mfgs3\pretrained\pubchem217208filter_smrt.csv')
data.head()

dsmiles = list(data['SMILES'])
smiles_list = dsmiles
print(len(smiles_list))

ssmiles_list = []
for smiles in tqdm(smiles_list):
    mol = Chem.MolFromSmiles(smiles)
    smi = Chem.MolToSmiles(mol)  # standardize
    ssmiles_list.append(smi)

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Descriptors import ExactMolWt

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

import pandas as pd
from pandas import *
import numpy as np
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from PubChemFP import GetPubChemFPs
from rdkit.Chem import AllChem,MACCSkeys
from rdkit.Chem import rdFingerprintGenerator
import torch.nn.functional as F
RDLogger.DisableLog('rdApp.*')

import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import Dataset, DataLoader

import dill
with open('e:/250811/code/1/2/Mfgs3/pretrained/vocab.pkl', 'rb') as f:
    dictionary= dill.load(f)

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

# 定义 SMILES 字符串
# smiles2 = "N1([H])C([H])=NC([H])=N1"
# tok = token_smiles(smi=smiles2, vocab = dictionary)
# print(tok)

# 随机掩码
def mask_tokens(tokens, vocab, mask_prob=0.15):
    masked_tokens = []
    labels = []
  
    for token in tokens:
        # if token in maskable_tokens:  # 仅对特定 token 进行掩码
        if np.random.rand() < mask_prob:
            masked_tokens.append(vocab.mask_index)
            labels.append(token)
        else:
            masked_tokens.append(token)
            labels.append(-100)  # 不计算损失

    return masked_tokens, labels

# a, b = mask_tokens(tok, dictionary)
# print(a)
# print(b)

from torch.utils.data import Dataset, DataLoader

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

# 划分训练集和临时集（验证集 + 测试集）
smiles_train, smiles_temp, mw_train, mw_temp = train_test_split(
    ssmiles_list, mw_list, test_size=0.2, random_state=42)

# 划分验证集和测试集
smiles_val, smiles_test, mw_val, mw_test = train_test_split(
    smiles_temp, mw_temp, test_size=0.5, random_state=42)

# 打印数据集划分结果
print(f"训练集大小: {len(smiles_train)}, 验证集大小: {len(smiles_val)}, 测试集大小: {len(smiles_test)}")

# 创建数据集
dataset_train = PretrainDataset(smiles_train, mw_train, dictionary)
dataset_val = PretrainDataset(smiles_val, mw_val, dictionary)
dataset_test = PretrainDataset(smiles_test, mw_test, dictionary)

input_ids, labels, mw = dataset_train[5]
print("输入ID序列:", input_ids)
print("labels:", labels)
print("mw:", mw)

#创建数据加载器
batch_size = 128
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

import torch
import torch.nn as nn
import torch.nn.functional as F

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

# 初始化模型
model = TransformerModel(
    vocab_size = 14073,
    embed_dim = 128,
    num_heads = 4,
    en_layers = 2,
    ff_dim = 512,
    max_len = 250
).to(device)  # 将模型移至GPU

import os

def save_model(model, epoch, save_dir="models-1"):
    """保存模型到指定目录"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"model_epoch_{epoch}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at epoch {epoch}: {save_path}")

def pretrain_val(model, dataloader_train, dataloader_val, optimizer, criterion_masked,
                 epochs=150, save_interval=10, save_dir="models", loss_file = "loss.csv"):

    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []
    
    for epoch in tqdm(range(1, epochs + 1)):  # Epoch 从 1 开始
        # 训练阶段
        model.train()
        train_loss = 0
        for input_ids, labels, labels_mw in dataloader_train:
            optimizer.zero_grad()
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            labels_mw = labels_mw.to(device)
            
            outputs_masked, outputs_mw = model(input_ids)  # [batch_size, seq_len, vocab_size]
            outputs_masked = outputs_masked.view(-1, outputs_masked.size(-1))  # 展平
            labels = labels.view(-1)  # 展平
            loss_masked = criterion_masked(outputs_masked, labels)
            
            loss_mw = F.huber_loss(outputs_mw.squeeze(), labels_mw)
            loss_mw_norm = (loss_mw - 0.0)/(10.0 - 0.0)
            
            loss = 0.7*loss_masked + 0.3*loss_mw_norm
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(dataloader_train)
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_ids, labels, labels_mw in dataloader_val:
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                labels_mw = labels_mw.to(device)

                outputs_masked, outputs_mw = model(input_ids)  # [batch_size, seq_len, vocab_size]
                outputs_masked = outputs_masked.view(-1, outputs_masked.size(-1))  # 展平
                labels = labels.view(-1)  # 展平
                loss_masked = criterion_masked(outputs_masked, labels)
                
                loss_mw = F.huber_loss(outputs_mw.squeeze(), labels_mw)
                loss_mw_norm = (loss_mw - 0.0)/(10.0 - 0.0)
                
                loss = 0.7*loss_masked + 0.3*loss_mw_norm

                val_loss += loss.item()
        val_loss /= len(dataloader_val)
        val_losses.append(val_loss)

        # 保存最优模型状态
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

        # 打印训练信息
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # 每隔 save_interval 个 epoch 保存一次模型
        if epoch % save_interval == 0:
            save_model(model, epoch, save_dir)

    # 使用 Pandas 保存损失值为 CSV 文件
    loss_data = {
        "Epoch": list(range(1, epochs + 1)),
        "Train Loss": train_losses,
        "Val Loss": val_losses,
    }
    df = pd.DataFrame(loss_data)
    df.to_csv(loss_file, index=False)  # 保存为 CSV 文件

    print(f"Training log saved to {loss_file}")

    # 加载最优模型
    model.load_state_dict(best_model_state)
    return model

# 优化器和损失函数
optimizer_pretrain = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion_masked = nn.CrossEntropyLoss(ignore_index=-100)
# criterion_mw = nn.MSELoss()

# pretrain_val(model, dataloader_train, dataloader_val, optimizer_pretrain, criterion_masked, epochs=250, save_interval=10, 
#              save_dir="model-1", loss_file = "loss-1.csv")

import matplotlib.pyplot as plt
log_file = 'e:/250811/code/1/2/Mfgs3/pretrained/loss-1.csv'
df = pd.read_csv(log_file)
plt.plot(df["Epoch"], df["Train Loss"], label="Train Loss")
plt.plot(df["Epoch"], df["Val Loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim((0, 1))
plt.legend()
plt.title("Training and Validation Loss")
plt.show()

def load_model(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Loaded model from {checkpoint_path}")
    return model

checkpoint_path = "e:/250811/code/1/2/Mfgs3/pretrained/model_epoch_250.pth"
model = load_model(model, checkpoint_path)

import torch
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
    # total_loss = 0
    total_tokens = 0
    correct_predictions = 0

    criterion_masked = nn.CrossEntropyLoss(ignore_index=-100)
    # criterion_mw = nn.MSELoss()
            
    with torch.no_grad():
        for input_ids, labels, labels_mw in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            labels_mw = labels_mw.to(device)

            outputs_masked, outputs_mw = model(input_ids)  # [batch_size, seq_len, vocab_size]
            outputs_masked = outputs_masked.view(-1, outputs_masked.size(-1))  # 展平
            labels = labels.view(-1)  # 展平
            loss_masked = criterion_masked(outputs_masked, labels)
            
            loss_mw = F.huber_loss(outputs_mw.squeeze(), labels_mw)
            loss_mw_norm = (loss_mw - 0.0)/(10.0 - 0.0)
            
            loss = 0.7*loss_masked + 0.3*loss_mw_norm
            # total_loss += loss.item() * labels.size(0) * labels.size(1)
            # total_loss += loss.item()

            # predictions = outputs.argmax(dim = -1)
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

    print(f"loss: {loss:.4f}")
            
    # avg_loss = total_loss / total_tokens
    # perplexity = torch.exp(torch.tentor(avg_loss))
    accuracy = correct_predictions / total_tokens if total_tokens > 0 else 0.0
    # return avg_loss, perplexity.item(), accuracy
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy, r2, medre

# test_smiles = ["CNCCNC1=C2C(=O)C3=C(O)C=CC(NCCNC)=C3C(=O)C2=C(O)C=C1", "CN1C=NC(C[C@H](N)C(O)=O)=C1"]
# test_dataset = PretrainDataset(test_smiles, vocab)
# test_dataloader = DataLoader(test_dataset, shuffle=False)
accuracy = evaluate_mlm(model, dataloader_test, device = "cuda")
# print(f"Accuracy: {accuracy * 100:.2f}%")

# checkpoint_path = "model-1\model_epoch_250.pth"
# model = load_model(model, checkpoint_path)
# accuracy = evaluate_mlm(model, dataloader_test, device = "cuda")

# METLIN-CCS 微调
###########################################################################

data2 = pd.read_csv('e:/250811/code/1/58033.csv')
data2.head()

dsmiles2 = list(data2['SMILES'])
dadduct2 = list(data2['Adduct'])
dccs2 = list(data2['CCS'])

sdsmiles2 = []
for smiles2 in tqdm(dsmiles2):
    mol = Chem.MolFromSmiles(smiles2)
    smi = Chem.MolToSmiles(mol)  # standardize
    sdsmiles2.append(smi)
print(sdsmiles2[4])

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from rdkit.Chem.Descriptors import ExactMolWt

def smiles_to_ecfps(smiles, radius=2, nBits=1024):
    """Generate ECFP fingerprints"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return torch.zeros(nBits, dtype=torch.float)
    ecfps_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    ecfps_fp = ecfps_gen.GetFingerprintAsNumPy(mol)
    return torch.tensor(ecfps_fp, dtype=torch.float)

def smiles_to_PubChem(smiles):
    """Generate PubChem fingerprints"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return torch.zeros(881, dtype=torch.float)
    mol2 = Chem.AddHs(mol)
    return torch.tensor(GetPubChemFPs(mol2), dtype=torch.float)

def smiles_to_Pharmacophere(smiles):
    """Generate Pharmacophore fingerprints"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return torch.zeros(441, dtype=torch.float)
    fp_phaErGfp = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)
    return torch.tensor(fp_phaErGfp, dtype=torch.float)

def generate_combined_fp(smiles, radius=2, nBits=1024):
    """Generate combined ECFP, PubChem, and Pharmacophore fingerprints"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return torch.zeros(nBits + 881 + 441, dtype=torch.float)
    
    # Generate ECFP
    fp_morgan = smiles_to_ecfps(smiles, radius, nBits)
    # Generate PubChem
    fp_pubchem = smiles_to_PubChem(smiles)
    # Generate Pharmacophore
    fp_pharmacophere = smiles_to_Pharmacophere(smiles)
    # Concatenate fingerprints
    combined_fp = torch.cat([fp_morgan, fp_pubchem, fp_pharmacophere], dim=0)
    return combined_fp

from rdkit import Chem
from rdkit.Chem import Descriptors

def nbonds_list(smiles):
    nb = []
    for i in tqdm(range(len(smiles))):
        smi = smiles[i]
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        number_of_bonds = mol.GetNumBonds()
        nb.append(number_of_bonds)
    return nb

dnb2 = nbonds_list(sdsmiles2)

import torch
from torch.utils.data import Dataset

class CCSDataset(Dataset):
    def __init__(self, smiles_list, ion_types, ccs_values, nbonds, vocab, max_len=250):
        """
        初始化数据集
        """
        self.smiles_list = smiles_list
        self.ion_types = ion_types
        self.ccs_values = ccs_values
        self.nbonds = nbonds
        self.vocab = vocab
        self.max_len = max_len

        # 创建离子类型的映射表，将离子类型转换为索引
        self.ion_type_to_index = {'[M+H]+': 0, '[M+Na]+': 1, '[M-H]-': 2}
        self.num_ion_types = len(self.ion_type_to_index)  # 离子类型的总数

    def __len__(self):
        """
        返回数据集的大小
        """
        return len(self.smiles_list)

    def __getitem__(self, idx):
        """
        获取数据集中的一个样本
        """
        # 获取SMILES字符串并进行token化
        smiles = self.smiles_list[idx]
        tokens = token_smiles(smiles, self.vocab)
        
        input_ids = [self.vocab.cls_index] + tokens[:self.max_len-2] + [self.vocab.sep_index] + [self.vocab.pad_index] * (self.max_len-2 - len(tokens))

        # 计算combined FP (ECFP + PubChem)
        combined_fp = generate_combined_fp(smiles)

        # 将ion_types转换为独热编码
        ion_type = self.ion_types[idx]
        ion_index = self.ion_type_to_index[ion_type]
        ion_encoding = torch.eye(self.num_ion_types)[ion_index]

        # 获取CCS值并转换为张量
        ccs_value = torch.tensor(self.ccs_values[idx], dtype=torch.float)
        nbonds = torch.tensor(self.nbonds[idx], dtype=torch.long)

        return torch.tensor(input_ids), ion_encoding, ccs_value, combined_fp, nbonds

# 样本数据
smiles_list2 = sdsmiles2
ion_list2 = dadduct2
ccs_list2 = dccs2
nbonds_list2 = dnb2

from sklearn.model_selection import train_test_split

# 划分训练集和临时集（验证集 + 测试集）
smiles_train2, smiles_temp2, ccs_train2, ccs_temp2, ion_train2, ion_temp2, nbonds_train2, nbonds_temp2 = train_test_split(
    smiles_list2, ccs_list2, ion_list2, nbonds_list2,
    test_size=0.2, random_state=42)

# 划分验证集和测试集
smiles_val2, smiles_test2, ccs_val2, ccs_test2, ion_val2, ion_test2, nbonds_val2, nbonds_test2 = train_test_split(
    smiles_temp2, ccs_temp2, ion_temp2, nbonds_temp2,
    test_size=0.5, random_state=42)

# 打印数据集划分结果
print(f"训练集大小: {len(smiles_train2)}, 验证集大小: {len(smiles_val2)}, 测试集大小: {len(smiles_test2)}")

# 创建数据集
ccsdataset_train = CCSDataset(smiles_train2, ion_train2, ccs_train2, nbonds_train2, dictionary)
ccsdataset_val = CCSDataset(smiles_val2, ion_val2, ccs_val2, nbonds_val2, dictionary)
ccsdataset_test = CCSDataset(smiles_test2, ion_test2, ccs_test2, nbonds_test2, dictionary)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 14   ###最开始的是128 改为14
ccsdataloader_train = DataLoader(ccsdataset_train, batch_size=batch_size, shuffle=True)
ccsdataloader_val = DataLoader(ccsdataset_val, batch_size=batch_size, shuffle=False)
ccsdataloader_test = DataLoader(ccsdataset_test, batch_size=batch_size, shuffle=False)


class CCSPredictor(nn.Module):
    def __init__(self, base_model):
        super(CCSPredictor, self).__init__()
        self.base_model = base_model
        
        # Enhanced fingerprint processing for combined fingerprints (1024 + 881 + 441 = 2346)
        self.fp_processor = nn.Sequential(
            nn.Linear(2346, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
        )
        
        # Cross-attention mechanism
        self.to_query = nn.Linear(128, 128)
        self.to_key = nn.Linear(384, 128)
        self.to_value = nn.Linear(384, 128)
        
        # CCS prediction head - updated input dimension: 128 (cls_out) + 128 (attn_output) + 1 (nbonds) + 3 (ion_type) = 260
        self.ccs_head = nn.Sequential(
            nn.Linear(128 + 128 + 1 + 3, 384),  # cls_out + attn_output + nbonds + ion_type
            nn.ReLU(),
            nn.Linear(384, 384),
            nn.ReLU(),
            nn.Linear(384, 384),
            nn.ReLU(),
            nn.Linear(384, 384),
            nn.ReLU(),
            nn.Linear(384, 384),
            nn.ReLU(),
            nn.Linear(384, 1),
        )
    
    def forward(self, x, combined_fp, nb, adducts):
        seq_len = x.size(1)
        position_ids = torch.arange(0, seq_len).unsqueeze(0).to(x.device)
        embedded = self.base_model.embedding(x) + self.base_model.position_embedding(position_ids)
        transformer_out = self.base_model.encoder(embedded)
        
        # Use CLS token for prediction
        cls_out = transformer_out[:, 0, :]

        # Process combined fingerprints
        fp_out = self.fp_processor(combined_fp)
        
        # Cross-attention between transformer output and fingerprints
        q = self.to_query(cls_out)
        k = self.to_key(fp_out)
        v = self.to_value(fp_out)
        
        attn_scores = torch.matmul(q.unsqueeze(1), k.unsqueeze(2)).squeeze(2)
        attn_scores = attn_scores / (k.size(-1) ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = attn_weights.unsqueeze(1).bmm(v.unsqueeze(1)).squeeze(1)
        
        # Add nbonds dimension
        nb = nb.unsqueeze(-1)
        
        # Concatenate cls_out, attn_output, nbonds, and adducts
        x = torch.cat((cls_out, attn_output, nb, adducts), dim=-1)
        
        # Predict CCS
        ccs_pred = self.ccs_head(x)
        return ccs_pred

import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
def ccs_train(model, train_loader, val_loader, optimizer, epochs, save_interval=10, save_dir = "models", loss_file = "loss.csv"):
    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []
    
    # Add cosine annealing scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    for epoch in tqdm(range(1, epochs + 1)):
        # Training
        model.train()
        train_loss = 0
        for input_ids, ion_one_hot, labels_ccs, combined_fp, input_nb in train_loader:
            input_ids = input_ids.to(device)  # 将数据移动到GPU
            ion_one_hot = ion_one_hot.to(device)  # 将独热编码移动到GPU
            labels_ccs = labels_ccs.to(device)  # 将目标值移动到GPU
            combined_fp = combined_fp.to(device)
            input_nb = input_nb.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, combined_fp, input_nb, ion_one_hot)
            loss = F.huber_loss(outputs.squeeze(), labels_ccs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_ids, ion_one_hot, labels_ccs, combined_fp, input_nb in val_loader:
                input_ids = input_ids.to(device) 
                ion_one_hot = ion_one_hot.to(device)
                labels_ccs = labels_ccs.to(device)
                combined_fp = combined_fp.to(device)
                input_nb = input_nb.to(device)
                
                outputs = model(input_ids, combined_fp, input_nb, ion_one_hot)
                loss = F.huber_loss(outputs.squeeze(), labels_ccs)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Step the scheduler after each epoch
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 保存最优模型状态
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

        # 打印训练信息
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")

        # 每隔 save_interval 个 epoch 保存一次模型
        if epoch % save_interval == 0:
            save_model(model, epoch, save_dir)

    # 使用 Pandas 保存损失值为 CSV 文件
    loss_data = {
        "Epoch": list(range(1, epochs + 1)),
        "Train Loss": train_losses,
        "Val Loss": val_losses,
    }
    df = pd.DataFrame(loss_data)
    df.to_csv(loss_file, index=False)  # 保存为 CSV 文件

    print(f"Training log saved to {loss_file}")

    # 加载最优模型
    model.load_state_dict(best_model_state)
    return model

ccs_model = CCSPredictor(model).cuda()

#optimizer_ccs = torch.optim.Adam(ccs_model.parameters(), lr=1e-4)
optimizer_ccs = torch.optim.AdamW(ccs_model.parameters(), lr=1e-4, weight_decay=1e-4)
# criterion_ccs = nn.MSELoss()

# ccs_train(ccs_model, ccsdataloader_train, ccsdataloader_val, optimizer_ccs, epochs=200, save_interval=1, 
#           save_dir="models-ccs-1-58033-bs14", loss_file = 'loss-ccs-1-58033-bs14.csv')



import torch
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

def evaluate_preds(model, test_loader, device = "cuda"):
    model.eval()
    all_targets = []
    all_predictions = []
    
    # 在测试集上获取预测值和实际值
    with torch.no_grad():
        for smiles_batch, ion_one_hot_batch, ccs_batch, combined_fp, nb in test_loader:
            smiles_batch = smiles_batch.to(device)
            ion_one_hot_batch = ion_one_hot_batch.to(device)
            ccs_batch = ccs_batch.to(device)
            combined_fp = combined_fp.to(device)
            nb = nb.to(device)
            
            outputs = model(smiles_batch, combined_fp, nb, ion_one_hot_batch)
            # print(outputs.squeeze())
            all_predictions.extend(outputs.squeeze().tolist())
            all_targets.extend(ccs_batch.tolist())

    r2, medre = metrics(all_targets, all_predictions)
    return all_targets, all_predictions

checkpoint_path = r"e:\250811\code\1\2\Mfgs3\pretrained\cos\selectmodel\connected1\models-ccs-1-58033-bs14\model_epoch_169.pth"
checkpoint_path = "e:/250811/code/1/2/Mfgs3/pretrained/cos/selectmodel/connected1/average_model-ccs_weighted.pth"
ccsmodel = load_model(ccs_model, checkpoint_path)

# Get predictions and targets
targets, preds = evaluate_preds(ccsmodel, ccsdataloader_test)

# Create DataFrame and export to CSV
results_df = pd.DataFrame({
    'True_CCS': targets,
    'Pred_CCS': preds
})

# Save to CSV file
results_df.to_csv('58033test_predictions_vs_targets.csv', index=False)
print(f"Results saved to 58033test_predictions_vs_targets.csv")
print(f"Number of samples: {len(results_df)}")

# #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
dataex = pd.read_csv(r'e:\文章撰写\data2\glucase.csv')
dataex.head()

dsmilesex = list(dataex['SMILES'])
dadductex = list(dataex['Adduct'])
dccsex = list(dataex['True CCS'])

sdsmilesex = []
for smilesex in tqdm(dsmilesex, desc="Standardize extest SMILES"):
    mol = Chem.MolFromSmiles(smilesex)
    smi = Chem.MolToSmiles(mol)  # standardize
    sdsmilesex.append(smi)
print(sdsmilesex[4])

dnbex = nbonds_list(sdsmilesex)
# ccsdataset_extest = CCSDataset(sdsmilesex, dadductex, dccsex, dnbex, dictionary)
# ccsdataloader_extest = DataLoader(ccsdataset_extest, batch_size = 14, shuffle=False)

# # Get predictions and targets
# targets, preds = evaluate_preds(ccsmodel, ccsdataloader_extest)

# # Create DataFrame and export to CSV
# results_df = pd.DataFrame({
#     'True_CCS': targets,
#     'Pred_CCS': preds
# })

# # Save to CSV file
# results_df.to_csv('58033meanccs_predictions_vs_targets.csv', index=False)
# print(f"Results saved to 58033meanccs_predictions_vs_targets.csv")
# print(f"Number of samples: {len(results_df)}")


def build_model_inputs(test_smi, test_adduct, test_nb, dictionary, device):
    tokens = token_smiles(test_smi, dictionary)
    input_ids = [dictionary.cls_index] + tokens[:248] + [dictionary.sep_index] + [dictionary.pad_index] * (248 - len(tokens))
    ion_type_to_index = {'[M+H]+': 0, '[M+Na]+': 1, '[M-H]-': 2}
    ion_index = ion_type_to_index.get(test_adduct, 0)
    ion_encoding = torch.eye(3, device=device)[ion_index].unsqueeze(0)
    combined_fp = generate_combined_fp(test_smi).unsqueeze(0).to(device)
    nb = torch.tensor([test_nb], dtype=torch.long, device=device)
    input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)
    return input_ids, combined_fp, nb, ion_encoding

def enable_mc_dropout_only(module):
    # 仅开启Dropout，保持BN在eval，避免BN因batch=1而不稳定
    if isinstance(module, nn.Dropout):
        module.train()
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        module.eval()
    for child in module.children():
        enable_mc_dropout_only(child)

@torch.no_grad()
def mc_dropout_predict_large(ccsmodel, input_ids, combined_fp, nb, ion_encoding, T=20):
    # 基于MC Dropout的T次前向，返回均值与标准差
    ccsmodel.eval()
    enable_mc_dropout_only(ccsmodel)  # 仅开启Dropout
    preds = []
    for _ in range(T):
        pred = ccsmodel(input_ids, combined_fp, nb, ion_encoding).squeeze().detach().float().item()
        preds.append(pred)
    mean = float(np.mean(preds))
    std = float(np.std(preds, ddof=1)) if len(preds) > 1 else 0.0
    return mean, std

# 推理
results = []
for idx in tqdm(range(len(sdsmilesex)), desc="Predicting"):
    test_smi = sdsmilesex[idx]
    test_adduct = dadductex[idx]
    test_ccs = float(dccsex[idx])
    test_nb = int(dnbex[idx])

    input_ids, combined_fp, nb, ion_encoding = build_model_inputs(
        test_smi, test_adduct, test_nb, dictionary, device
    )

    ccsmodel.eval()
    with torch.no_grad():
        pred = ccsmodel(input_ids, combined_fp, nb, ion_encoding).squeeze().detach().float().item()
    mean1, std1 = mc_dropout_predict_large(ccsmodel, input_ids, combined_fp, nb, ion_encoding, T=20)


    results.append({
        "SMILES": test_smi,
        "Adduct": test_adduct,
        "True_CCS": test_ccs,
        "Pred_CCS": pred,
        "PredMean_CCS": mean1,
        "std": std1
    })

# 输出结果
results_df = pd.DataFrame(results)
print(results_df.head())
results_df.to_csv("58033ensemble-mean-std_glu.csv", index=False)

# 评估指标（使用Pred_CCS均值）
y_true = results_df["True_CCS"].astype(float).values
y_pred = results_df["PredMean_CCS"].astype(float).values
metrics(y_true, y_pred)

# #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# import matplotlib.pyplot as plt
# log_file = 'loss-ccs-1-58033.csv'
# df = pd.read_csv(log_file)
# plt.plot(df["Epoch"], df["Train Loss"], label="Train Loss")
# plt.plot(df["Epoch"], df["Val Loss"], label="Val Loss")
# plt.ylim(0, 4)
# plt.xlim(0, 200)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.title("Training and Validation Loss")
# plt.show()

import os
import torch

# 取验证集上性能最好的 N 个 Epoch 的模型参数，并进行加权参数平均
def weighted_average_best_models(model, model_dir, val_losses, top_n=5, save_path="weighted_average_model.pth"):
    # 获取验证集最好的 top_n 次 Epoch 索引
    best_epochs = sorted(range(len(val_losses)), key=lambda i: val_losses[i])[:top_n]
    print(f"Best epochs: {best_epochs}")
    
    best_model_states = []
    val_losses_subset = []

    # 加载最佳 Epoch 的模型参数
    for epoch in best_epochs:
        model_path = os.path.join(model_dir, f"model_epoch_{epoch+1}.pth")
        if os.path.exists(model_path):
            model_state = torch.load(model_path)
            best_model_states.append(model_state)
            val_losses_subset.append(val_losses[epoch])  # 保存对应的验证损失
            print(f"Loaded model from epoch {epoch+1} at {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

    # 确保有至少一个模型
    if not best_model_states:
        raise ValueError("No valid model files found for averaging.")

    # 计算权重：验证损失的倒数归一化
    weights = [1 / loss for loss in val_losses_subset]  # 验证损失的倒数
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]  # 归一化权重
    print(f"Calculated weights: {weights}")

    # 参数加权平均
    averaged_state_dict = {}
    for key in best_model_states[0]:  # 遍历每个参数
        averaged_state_dict[key] = sum(
            state[key] * weight for state, weight in zip(best_model_states, weights)
        )

    # 加载平均后的参数到模型
    model.load_state_dict(averaged_state_dict)

    # 保存最终加权平均模型
    torch.save(model.state_dict(), save_path)
    print(f"Weighted averaged model saved to {save_path}")
    return model

# # 加载训练日志以获取验证损失
# loss_file = r'e:\250811\code\1\2\Mfgs3\pretrained\cos\selectmodel\connected1\loss-ccs-1-58033-bs14.csv'
# loss_data = pd.read_csv(loss_file)
# val_losses = loss_data['Val Loss'].tolist()

# # 定义模型和模型保存路径
# model_dir = "e:/250811/code/1/2/Mfgs3/pretrained/cos/selectmodel/connected1/models-ccs-1-58033-bs14/"
# # ccs_model = CCSPredictor(model).cuda()

# # 平均验证集上最好的 5 次模型参数
# average_model_path = "e:/250811/code/1/2/Mfgs3/pretrained/cos/selectmodel/connected1/average_model-ccs_weighted.pth"
# ccs_model = weighted_average_best_models(ccs_model, model_dir, val_losses, top_n=5, save_path=average_model_path)

checkpoint_path = "e:/250811/code/1/2/Mfgs3/pretrained/cos/selectmodel/connected1/average_model-ccs_weighted.pth"
ccsmodel = load_model(ccs_model, checkpoint_path)

# Get predictions and targets
targets, preds = evaluate_preds(ccsmodel, ccsdataloader_test)

# # Create DataFrame and export to CSV
# results_df = pd.DataFrame({
#     'True_CCS': targets,
#     'Pred_CCS': preds
# })

# # Save to CSV file
# results_df.to_csv('ccs_predictions_vs_targets.csv', index=False)
# print(f"Results saved to ccs_predictions_vs_targets.csv")
# print(f"Number of samples: {len(results_df)}")


# 下游任务模型定义
class CCSPredictor3(nn.Module):
    def __init__(self, base_model):
        super(CCSPredictor3, self).__init__()
        self.base_model = base_model
    
    def forward(self, x, combined_fp, nb, adducts):
        seq_len = x.size(1)
        position_ids = torch.arange(0, seq_len).unsqueeze(0).to(x.device)
        embedded = self.base_model.base_model.embedding(x) + self.base_model.base_model.position_embedding(position_ids)
        transformer_out = self.base_model.base_model.encoder(embedded)
        
        # Use CLS token for prediction
        cls_out = transformer_out[:, 0, :]

        # Process combined fingerprints
        fp_out = self.base_model.fp_processor(combined_fp)
        
        # Cross-attention between transformer output and fingerprints
        q = self.base_model.to_query(cls_out)
        k = self.base_model.to_key(fp_out)
        v = self.base_model.to_value(fp_out)

        attn_scores = torch.matmul(q.unsqueeze(1), k.unsqueeze(2)).squeeze(2)
        attn_scores = attn_scores / (k.size(-1) ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = attn_weights.unsqueeze(1).bmm(v.unsqueeze(1)).squeeze(1)
        
        # Add nbonds dimension
        nb = nb.unsqueeze(-1)
        
        # Concatenate features
        x = torch.cat((cls_out, attn_output, nb, adducts), dim=-1)
        
        # Predict CCS
        ccs_pred = self.base_model.ccs_head(x)
        return ccs_pred


data3 = pd.read_csv('e:/250811/code/1/F8196.csv')
data3.head()

dsmiles3 = list(data3['SMILES'])
dadduct3 = list(data3['Adduct'])
dccs3 = list(data3['CCS'])
print(dsmiles3[4])

sdsmiles3 = []
for smiles3 in tqdm(dsmiles3):
    mol = Chem.MolFromSmiles(smiles3)
    smi = Chem.MolToSmiles(mol)  # standardize
    sdsmiles3.append(smi)
print(sdsmiles3[4])

dnb3 = nbonds_list(sdsmiles3)

# 样本数据
smiles_list3 = sdsmiles3
ion_list3 = dadduct3
ccs_list3 = dccs3
nbonds_list3 = dnb3

from sklearn.model_selection import train_test_split

# 划分训练集和临时集（验证集 + 测试集）
smiles_train3, smiles_temp3, ccs_train3, ccs_temp3, ion_train3, ion_temp3, nbonds_train3, nbonds_temp3 = train_test_split(
    smiles_list3, ccs_list3, ion_list3, nbonds_list3,
    test_size=0.2, random_state=42)

# 划分验证集和测试集
smiles_val3, smiles_test3, ccs_val3, ccs_test3, ion_val3, ion_test3, nbonds_val3, nbonds_test3 = train_test_split(
    smiles_temp3, ccs_temp3, ion_temp3, nbonds_temp3,
    test_size=0.5, random_state=42)

# 打印数据集划分结果
print(f"训练集大小: {len(smiles_train3)}, 验证集大小: {len(smiles_val3)}, 测试集大小: {len(smiles_test3)}")

# 创建数据集
ccsdataset_train3 = CCSDataset(smiles_train3, ion_train3, ccs_train3, nbonds_train3, dictionary)
ccsdataset_val3 = CCSDataset(smiles_val3, ion_val3, ccs_val3, nbonds_val3, dictionary)
ccsdataset_test3 = CCSDataset(smiles_test3, ion_test3, ccs_test3, nbonds_test3, dictionary)

batch_size = 14 
ccsdataloader_train2 = DataLoader(ccsdataset_train3, batch_size=batch_size, shuffle=True)
ccsdataloader_val2 = DataLoader(ccsdataset_val3, batch_size=batch_size, shuffle=False)
ccsdataloader_test2 = DataLoader(ccsdataset_test3, batch_size=batch_size, shuffle=False)


def ccs_train2(model, train_loader, val_loader, optimizer, epochs, save_interval=10, save_dir = "models", loss_file = "loss.csv"):
    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []
    
    # Add cosine annealing scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    for epoch in tqdm(range(1, epochs + 1)):
        # Training
        model.train()
        train_loss = 0
        for input_ids, ion_one_hot, labels_ccs, combined_fp, input_nb in train_loader:
            input_ids = input_ids.to(device)  # 将数据移动到GPU
            ion_one_hot = ion_one_hot.to(device)  # 将独热编码移动到GPU
            labels_ccs = labels_ccs.to(device)  # 将目标值移动到GPU
            combined_fp = combined_fp.to(device)
            input_nb = input_nb.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, combined_fp, input_nb, ion_one_hot)
            loss = F.huber_loss(outputs.squeeze(), labels_ccs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_ids, ion_one_hot, labels_ccs, combined_fp, input_nb in val_loader:
                input_ids = input_ids.to(device) 
                ion_one_hot = ion_one_hot.to(device)
                labels_ccs = labels_ccs.to(device)
                combined_fp = combined_fp.to(device)
                input_nb = input_nb.to(device)
                
                outputs = model(input_ids, combined_fp, input_nb, ion_one_hot)
                loss = F.huber_loss(outputs.squeeze(), labels_ccs)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Step the scheduler after each epoch
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 保存最优模型状态
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

        # 打印训练信息
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")

        # 每隔 save_interval 个 epoch 保存一次模型
        if epoch % save_interval == 0:
            save_model(model, epoch, save_dir)

    # 使用 Pandas 保存损失值为 CSV 文件
    loss_data = {
        "Epoch": list(range(1, epochs + 1)),
        "Train Loss": train_losses,
        "Val Loss": val_losses,
    }
    df = pd.DataFrame(loss_data)
    df.to_csv(loss_file, index=False)  # 保存为 CSV 文件

    print(f"Training log saved to {loss_file}")

    # 加载最优模型
    model.load_state_dict(best_model_state)
    return model


ccs_model2 = CCSPredictor3(ccsmodel).cuda()

# optimizer_ccs2 = torch.optim.Adam(ccs_model2.parameters(), lr=1e-4)
optimizer_ccs2 = torch.optim.AdamW(ccs_model2.parameters(), lr=1e-4, weight_decay=1e-4)

# ccs_train2(ccs_model2, ccsdataloader_train2, ccsdataloader_val2, optimizer_ccs2, epochs=200, save_interval=1, 
#           save_dir="models-ccs-1-58033-8196", loss_file = 'loss-ccs-1-58033-8196.csv')






import matplotlib.pyplot as plt
log_file = 'loss-ccs-1-58033-8196.csv'
df = pd.read_csv(log_file)
plt.plot(df["Epoch"], df["Train Loss"], label="Train Loss")
plt.plot(df["Epoch"], df["Val Loss"], label="Val Loss")
plt.ylim(0, 4)
plt.xlim(0, 150)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.show()

# # 加载训练日志以获取验证损失
# loss_file = 'loss-ccs-1-58033-8196.csv'
# loss_data = pd.read_csv(loss_file)
# val_losses = loss_data['Val Loss'].tolist()

# # 定义模型和模型保存路径
# model_dir = "models-ccs-1-58033-8196/"

# # 平均验证集上最好的 5 次模型参数
# average_model_path = "models-ccs-1-58033-8196_average_model-ccs_weighted.pth"
# ccs_model2 = weighted_average_best_models(ccs_model2, model_dir, val_losses, top_n=5, save_path=average_model_path)

checkpoint_path = r"e:\250811\code\1\2\Mfgs3\pretrained\cos\selectmodel\connected1\models-ccs-1-58033-8196_average_model-ccs_weighted.pth"
ccsmodel2 = load_model(ccs_model2, checkpoint_path)

# Get predictions and targets from second model
targets2, preds2 = evaluate_preds(ccsmodel2, ccsdataloader_test2)

# Create DataFrame and export to CSV for second model
results_df2 = pd.DataFrame({
    'True_CCS': targets2,
    'Pred_CCS': preds2
})

# Save to CSV file
results_df2.to_csv('8196testccs_predictions_vs_targets_model2.csv', index=False)
print(f"Model 2 results saved to ccs_predictions_vs_targets_model2.csv")
print(f"Number of samples: {len(results_df2)}")


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
@torch.no_grad()
def mc_dropout_predict_small(ccsmodel2, input_ids, combined_fp, nb, ion_encoding, T=20):
    # 基于MC Dropout的T次前向，返回均值与标准差
    ccsmodel2.eval()
    enable_mc_dropout_only(ccsmodel2)  # 仅开启Dropout
    preds = []
    for _ in range(T):
        pred = ccsmodel2(input_ids, combined_fp, nb, ion_encoding).squeeze().detach().float().item()
        preds.append(pred)
    mean = float(np.mean(preds))
    std = float(np.std(preds, ddof=1)) if len(preds) > 1 else 0.0
    return mean, std

dataex = pd.read_csv(r'e:\文章撰写\data2\glucase.csv')
dataex.head()

dsmilesex = list(dataex['SMILES'])
dadductex = list(dataex['Adduct'])
dccsex = list(dataex['True CCS'])

sdsmilesex = []
for smilesex in tqdm(dsmilesex, desc="Standardize extest SMILES"):
    mol = Chem.MolFromSmiles(smilesex)
    smi = Chem.MolToSmiles(mol)  # standardize
    sdsmilesex.append(smi)
print(sdsmilesex[4])

dnbex = nbonds_list(sdsmilesex)
# ccsdataset_extest = CCSDataset(sdsmilesex, dadductex, dccsex, dnbex, dictionary)
# ccsdataloader_extest = DataLoader(ccsdataset_extest, batch_size = 14, shuffle=False)

# # Get predictions and targets
# targets, preds = evaluate_preds(ccsmodel, ccsdataloader_extest)

# # Create DataFrame and export to CSV
# results_df = pd.DataFrame({
#     'True_CCS': targets,
#     'Pred_CCS': preds
# })

# # Save to CSV file
# results_df.to_csv('58033meanccs_predictions_vs_targets.csv', index=False)
# print(f"Results saved to 58033meanccs_predictions_vs_targets.csv")
# print(f"Number of samples: {len(results_df)}")

# 推理
results2 = []
for idx in tqdm(range(len(sdsmilesex)), desc="Predicting"):
    test_smi = sdsmilesex[idx]
    test_adduct = dadductex[idx]
    test_ccs = float(dccsex[idx])
    test_nb = int(dnbex[idx])

    input_ids, combined_fp, nb, ion_encoding = build_model_inputs(
        test_smi, test_adduct, test_nb, dictionary, device
    )

    ccsmodel2.eval()
    with torch.no_grad():
        pred = ccsmodel2(input_ids, combined_fp, nb, ion_encoding).squeeze().detach().float().item()    
    mean2, std2 = mc_dropout_predict_small(ccsmodel2, input_ids, combined_fp, nb, ion_encoding, T=20)

    results2.append({
        "SMILES": test_smi,
        "Adduct": test_adduct,
        "True_CCS": test_ccs,
        "Pred_CCS": pred,
        "PredMean_CCS": mean2,
        "std": std2
    })

# 输出结果
results_df = pd.DataFrame(results2)
print(results_df.head())
results_df.to_csv("8196best-mean-std_glu.csv", index=False)

# 评估指标（使用Pred_CCS均值）
y_true = results_df["True_CCS"].astype(float).values
y_pred = results_df["PredMean_CCS"].astype(float).values
metrics(y_true, y_pred)

# #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!





# 测试模型（基于AD与MC Dropout的自动选择）
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm
import torch
import torch.nn as nn

def combined_fp_selection(smiles, radius=2, nBits=1024):
    """Generate combined ECFP and PubChem fingerprints as ExplicitBitVect for TanimotoSimilarity"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return None  # 返回None，后续处理
    # ECFP
    fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    # PubChem
    try:
        from PubChemFP import GetPubChemFPs
        pubchem_fp_arr = GetPubChemFPs(Chem.AddHs(mol))
        fp_pubchem = DataStructs.CreateFromBitString(''.join(str(int(x)) for x in pubchem_fp_arr))
    except Exception as e:
        fp_pubchem = None
    # 合并为一个ExplicitBitVect（简单拼接）
    if fp_pubchem is not None:
        arr1 = np.zeros((nBits,), dtype=np.int32)
        arr2 = np.zeros((881,), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fp_morgan, arr1)
        DataStructs.ConvertToNumpyArray(fp_pubchem, arr2)
        arr = np.concatenate([arr1, arr2])
        fp_combined = DataStructs.CreateFromBitString(''.join(str(int(x)) for x in arr))
        return fp_combined
    else:
        return fp_morgan  # 只用ECFP

def bulk_ad_score(fp_query, fp_library, k=5):
    # 使用BulkTanimotoSimilarity加速，返回Top-K平均
    sims = DataStructs.BulkTanimotoSimilarity(fp_query, fp_library)
    if len(sims) == 0:
        return 0.0
    sims = sorted(sims, reverse=True)[:min(k, len(sims))]
    return float(np.mean(sims))


def compute_ad_scores_for_smiles(smiles_list, fp_library, k=5, radius=2, nBits=1024, desc="AD scores"):
    scores = []
    for smi in tqdm(smiles_list, desc=desc):
        fp = combined_fp_selection(smi, radius=radius, nBits=nBits)
        if fp is None:
            scores.append(0.0)
        else:
            scores.append(bulk_ad_score(fp, fp_library, k=k))
    return scores


def build_ad_index(train_smiles_stage1, val_smiles_stage1,
                   train_smiles_stage2, val_smiles_stage2,
                   k=5, radius=2, nBits=1024, q=0.05):
    # 训练库指纹
    fp_lib_stage1 = [combined_fp_selection(s, radius, nBits) for s in tqdm(train_smiles_stage1, desc="Build FP lib S1")]
    fp_lib_stage1 = [fp for fp in fp_lib_stage1 if fp is not None]

    fp_lib_stage2 = [combined_fp_selection(s, radius, nBits) for s in tqdm(train_smiles_stage2, desc="Build FP lib S2")]
    fp_lib_stage2 = [fp for fp in fp_lib_stage2 if fp is not None]

    # # 验证集AD分数
    # val_scores_s1 = compute_ad_scores_for_smiles(val_smiles_stage1, fp_lib_stage1, k=k, radius=radius, nBits=nBits, desc="AD S1 (val)")
    # val_scores_s2 = compute_ad_scores_for_smiles(val_smiles_stage2, fp_lib_stage2, k=k, radius=radius, nBits=nBits, desc="AD S2 (val)")

    # thr1 = float(np.quantile(val_scores_s1, q)) if len(val_scores_s1) > 0 else 0.0
    # thr2 = float(np.quantile(val_scores_s2, q)) if len(val_scores_s2) > 0 else 0.0

    # 直接使用已知阈值
    thr1 = 0.724799934396318  # 阈值高那么多，是不是每个分子测了三次，验证集中的分子与训练集中的分子重复率很高
    thr2 = 0.586951719976712
    # thr1 = 0.7
    # thr2 = 0.7

    ad_index = {
        "k": k, "radius": radius, "nBits": nBits, "q": q,
        "stage1": {"fp_lib": fp_lib_stage1, "thr": thr1, "name": "58033"},
        "stage2": {"fp_lib": fp_lib_stage2, "thr": thr2, "name": "8196"},
    }
    print(f"AD thresholds -> S1({ad_index['stage1']['name']}): {thr1:.4f}, S2({ad_index['stage2']['name']}): {thr2:.4f}")
    return ad_index


# ------------------- 构建AD索引与阈值 -------------------
# 使用前文已划分的阶段训练/验证SMILES
# 阶段1（58033）: smiles_train2 / smiles_val2
# 阶段2（8196） : smiles_train3 / smiles_val3
ad_index = build_ad_index(
    train_smiles_stage1=smiles_train2, val_smiles_stage1=smiles_val2,
    train_smiles_stage2=smiles_train3, val_smiles_stage2=smiles_val3,
    k=5, radius=2, nBits=1024, q=0.05
)

# ------------------- 加载测试集并推理 -------------------
data4 = pd.read_csv(r'e:\文章撰写\data2\glucase.csv')
dsmiles4 = list(data4['SMILES'])
dadduct4 = list(data4['Adduct'])
dccs4 = list(data4['True CCS'])

# 标准化SMILES
sdsmiles4 = []
for smiles4 in tqdm(dsmiles4, desc="Standardize test SMILES"):
    mol = Chem.MolFromSmiles(smiles4)
    smi = Chem.MolToSmiles(mol) if mol is not None else smiles4
    sdsmiles4.append(smi)

# 计算键数
dnb4 = nbonds_list(sdsmiles4)

# 推理
alpha=0.5 
eps=1e-8

pred_mean_58033 = [item["PredMean_CCS"] for item in results]
print(pred_mean_58033)

pred_mean_8196 = [item["PredMean_CCS"] for item in results2]
print(pred_mean_8196)

pred_std_58033 = [item["std"] for item in results]
print(pred_std_58033)

pred_std_8196 = [item["std"] for item in results2]
print(pred_mean_8196)

pred_58033 = [item["Pred_CCS"] for item in results]
print(pred_58033)

pred_8196 = [item["Pred_CCS"] for item in results2]
print(pred_8196)


resultsall = []
for idx in tqdm(range(len(sdsmiles4)), desc="Predicting with AD + MC Dropout"):
    test_smi = sdsmiles4[idx]
    test_adduct = dadduct4[idx]
    test_ccs = float(dccs4[idx])
    test_nb = int(dnb4[idx])

    # 计算AD分数
    fp_q = combined_fp_selection(test_smi, radius=ad_index["radius"], nBits=ad_index["nBits"])
    ad1 = bulk_ad_score(fp_q, ad_index["stage1"]["fp_lib"], k=ad_index["k"]) if fp_q is not None else 0.0
    ad2 = bulk_ad_score(fp_q, ad_index["stage2"]["fp_lib"], k=ad_index["k"]) if fp_q is not None else 0.0
    in1 = ad1 >= ad_index["stage1"]["thr"]
    in2 = ad2 >= ad_index["stage2"]["thr"]   

    mean1, std1 = pred_mean_58033[idx], pred_std_58033[idx]
    mean2, std2 = pred_mean_8196[idx], pred_std_8196[idx]

    # 选择策略
    if in2 and not in1:
        sel = "8196"; pred_mean, pred_std = mean2, std2; reason = "Only S2 in-domain"
    elif in1 and not in2:
        sel = "58033"; pred_mean, pred_std = mean1, std1; reason = "Only S1 in-domain"
    elif in1 and in2:
        if std1 <= std2:
            sel = "58033"; pred_mean, pred_std = mean1, std1; reason = "Both in-domain, choose lower uncertainty (S1)"
        else:
            sel = "8196"; pred_mean, pred_std = mean2, std2; reason = "Both in-domain, choose lower uncertainty (S2)"
    else:
        # 两者都不在域：加权集成（结合AD和不确定度倒数）
        # 归一化 AD
        ad_sum = ad1 + ad2 + eps
        nad1, nad2 = ad1 / ad_sum, ad2 / ad_sum
        # 归一化 不确定度倒数
        inv1, inv2 = 1.0 / (std1 + eps), 1.0 / (std2 + eps)
        inv_sum = inv1 + inv2 + eps
        ninv1, ninv2 = inv1 / inv_sum, inv2 / inv_sum
        w1 = alpha * nad1 + (1 - alpha) * ninv1
        w2 = alpha * nad2 + (1 - alpha) * ninv2
        wsum = w1 + w2 + eps
        w1, w2 = w1 / wsum, w2 / wsum
        pred_mean = float(w1 * mean1 + w2 * mean2)
        # 集成不确定度（近似）：方差的加权和 + 均值差异项
        var1, var2 = std1**2, std2**2
        pred_var = float(w1 * var1 + w2 * var2 + w1 * (mean1 - pred_mean) ** 2 + w2 * (mean2 - pred_mean) ** 2)
        pred_std = float(np.sqrt(max(pred_var, 0.0)))
        sel = "weighted_ensemble"
        reason = "Both out-of-domain, weighted by AD and 1/uncertainty"

    detail = {
        "AD_58033": ad1, "AD_8196": ad2,
        "Thr_58033": ad_index["stage1"]["thr"], "Thr_8196": ad_index["stage2"]["thr"],
        "InDomain_58033": in1, "InDomain_8196": in2,
        "PredMean_58033": mean1, "PredStd_58033": std1,
        "PredMean_8196": mean2, "PredStd_8196": std2,
        "Selected": sel, "Reason": reason
    }

    resultsall.append({
        "SMILES": test_smi,
        "Adduct": test_adduct,
        "True_CCS": test_ccs,
        "Pred_CCS": pred_mean,
        "Pred_Std": pred_std,
        "AD_58033": detail["AD_58033"],
        "AD_8196": detail["AD_8196"],
        "Thr_58033": detail["Thr_58033"],
        "Thr_8196": detail["Thr_8196"],
        "InDomain_58033": detail["InDomain_58033"],
        "InDomain_8196": detail["InDomain_8196"],
        "PredMean_58033": detail["PredMean_58033"],
        "PredStd_58033": detail["PredStd_58033"],
        "PredMean_8196": detail["PredMean_8196"],
        "PredStd_8196": detail["PredStd_8196"],
        "Selected_Model": detail["Selected"],
        "Reason": detail["Reason"],
    })

# 输出结果
results_df = pd.DataFrame(resultsall)
print(results_df.head())
results_df.to_csv("glu-testsetall_predict_with_ad_mc_dropout.csv", index=False)

# 评估指标（使用Pred_CCS均值）
y_true = results_df["True_CCS"].astype(float).values
y_pred = results_df["Pred_CCS"].astype(float).values
metrics(y_true, y_pred)
# 若希望只用不确定度或只用AD做双出域集成，可在select_model_and_predict中调整alpha为1.0或0.0。    





# def select_model_and_predict(test_smi, test_adduct, test_nb, ad_index, dictionary, device,
#                              T=20, alpha=0.5, eps=1e-8):
#     # 计算AD分数
#     fp_q = combined_fp_selection(test_smi, radius=ad_index["radius"], nBits=ad_index["nBits"])
#     ad1 = bulk_ad_score(fp_q, ad_index["stage1"]["fp_lib"], k=ad_index["k"]) if fp_q is not None else 0.0
#     ad2 = bulk_ad_score(fp_q, ad_index["stage2"]["fp_lib"], k=ad_index["k"]) if fp_q is not None else 0.0
#     in1 = ad1 >= ad_index["stage1"]["thr"]
#     in2 = ad2 >= ad_index["stage2"]["thr"]

#     # # 准备输入
#     # input_ids, combined_fp, nb, ion_encoding = build_model_inputs(test_smi, test_adduct, test_nb, dictionary, device)

#     # # 先计算两模型的MC Dropout不确定度
#     # mean1, std1 = mc_dropout_predict_large(ccsmodel, input_ids, combined_fp, nb, ion_encoding, T=T)
#     # print(mean1)
#     # mean2, std2 = mc_dropout_predict_small(ccsmodel2, input_ids, combined_fp, nb, ion_encoding, T=T)
#     # print(mean2)

#     # 选择策略
#     if in2 and not in1:
#         sel = "8196"; pred_mean, pred_std = mean2, std2; reason = "Only S2 in-domain"
#     elif in1 and not in2:
#         sel = "58033"; pred_mean, pred_std = mean1, std1; reason = "Only S1 in-domain"
#     elif in1 and in2:
#         if std1 <= std2:
#             sel = "58033"; pred_mean, pred_std = mean1, std1; reason = "Both in-domain, choose lower uncertainty (S1)"
#         else:
#             sel = "8196"; pred_mean, pred_std = mean2, std2; reason = "Both in-domain, choose lower uncertainty (S2)"
#     else:
#         # 两者都不在域：加权集成（结合AD和不确定度倒数）
#         # 归一化 AD
#         ad_sum = ad1 + ad2 + eps
#         nad1, nad2 = ad1 / ad_sum, ad2 / ad_sum
#         # 归一化 不确定度倒数
#         inv1, inv2 = 1.0 / (std1 + eps), 1.0 / (std2 + eps)
#         inv_sum = inv1 + inv2 + eps
#         ninv1, ninv2 = inv1 / inv_sum, inv2 / inv_sum
#         w1 = alpha * nad1 + (1 - alpha) * ninv1
#         w2 = alpha * nad2 + (1 - alpha) * ninv2
#         wsum = w1 + w2 + eps
#         w1, w2 = w1 / wsum, w2 / wsum
#         pred_mean = float(w1 * mean1 + w2 * mean2)
#         # 集成不确定度（近似）：方差的加权和 + 均值差异项
#         var1, var2 = std1**2, std2**2
#         pred_var = float(w1 * var1 + w2 * var2 + w1 * (mean1 - pred_mean) ** 2 + w2 * (mean2 - pred_mean) ** 2)
#         pred_std = float(np.sqrt(max(pred_var, 0.0)))
#         sel = "weighted_ensemble"
#         reason = "Both out-of-domain, weighted by AD and 1/uncertainty"

#     detail = {
#         "AD_58033": ad1, "AD_8196": ad2,
#         "Thr_58033": ad_index["stage1"]["thr"], "Thr_8196": ad_index["stage2"]["thr"],
#         "InDomain_58033": in1, "InDomain_8196": in2,
#         "PredMean_58033": mean1, "PredStd_58033": std1,
#         "PredMean_8196": mean2, "PredStd_8196": std2,
#         "Selected": sel, "Reason": reason
#     }
#     return pred_mean, pred_std, detail

# # ------------------- 构建AD索引与阈值 -------------------
# # 使用前文已划分的阶段训练/验证SMILES
# # 阶段1（58033）: smiles_train2 / smiles_val2
# # 阶段2（8196） : smiles_train3 / smiles_val3
# ad_index = build_ad_index(
#     train_smiles_stage1=smiles_train2, val_smiles_stage1=smiles_val2,
#     train_smiles_stage2=smiles_train3, val_smiles_stage2=smiles_val3,
#     k=5, radius=2, nBits=1024, q=0.05
# )

# # ------------------- 加载测试集并推理 -------------------
# data4 = pd.read_csv(r'e:\2507XX\baseline0708\50.csv')
# dsmiles4 = list(data4['SMILES'])
# dadduct4 = list(data4['Adduct'])
# dccs4 = list(data4['True CCS'])

# # 标准化SMILES
# sdsmiles4 = []
# for smiles4 in tqdm(dsmiles4, desc="Standardize test SMILES"):
#     mol = Chem.MolFromSmiles(smiles4)
#     smi = Chem.MolToSmiles(mol) if mol is not None else smiles4
#     sdsmiles4.append(smi)

# # 计算键数
# dnb4 = nbonds_list(sdsmiles4)

# # 推理
# results = []
# for idx in tqdm(range(len(sdsmiles4)), desc="Predicting with AD + MC Dropout"):
#     test_smi = sdsmiles4[idx]
#     test_adduct = dadduct4[idx]
#     test_ccs = float(dccs4[idx])
#     test_nb = int(dnb4[idx])

#     pred_mean, pred_std, info = select_model_and_predict(
#         test_smi, test_adduct, test_nb,
#         ad_index=ad_index,
#         dictionary=dictionary, device=device,
#         T=20, alpha=0.5
#     )

#     results.append({
#         "SMILES": test_smi,
#         "Adduct": test_adduct,
#         "True_CCS": test_ccs,
#         "Pred_CCS": pred_mean,
#         "Pred_Std": pred_std,
#         "AD_58033": info["AD_58033"],
#         "AD_8196": info["AD_8196"],
#         "Thr_58033": info["Thr_58033"],
#         "Thr_8196": info["Thr_8196"],
#         "InDomain_58033": info["InDomain_58033"],
#         "InDomain_8196": info["InDomain_8196"],
#         "PredMean_58033": info["PredMean_58033"],
#         "PredStd_58033": info["PredStd_58033"],
#         "PredMean_8196": info["PredMean_8196"],
#         "PredStd_8196": info["PredStd_8196"],
#         "Selected_Model": info["Selected"],
#         "Reason": info["Reason"],
#     })

# # 输出结果
# results_df = pd.DataFrame(results)
# print(results_df.head())
# results_df.to_csv("50-testset3_predict_with_ad_mc_dropout.csv", index=False)

# # 评估指标（使用Pred_CCS均值）
# y_true = results_df["True_CCS"].astype(float).values
# y_pred = results_df["Pred_CCS"].astype(float).values
# metrics(y_true, y_pred)
# # 若希望只用不确定度或只用AD做双出域集成，可在select_model_and_predict中调整alpha为1.0或0.0。














# # 测试模型
# import pandas as pd
# from rdkit import Chem, DataStructs
# from rdkit.Chem import AllChem
# from tqdm import tqdm

# # 加载训练集和模型
# trainset1 = pd.read_csv('e:/250811/code/1/58033.csv')
# trainset2 = pd.read_csv('e:/250811/code/1/F8196.csv')
# train_smiles1 = list(trainset1['SMILES'])
# train_smiles2 = list(trainset2['SMILES'])

# # 预生成训练集指纹
# def generate_fingerprint(smiles):
#     mol = Chem.MolFromSmiles(smiles)
#     return AllChem.GetMorganFingerprint(mol, 2)

# train_fps1 = [generate_fingerprint(smi) for smi in tqdm(train_smiles1, desc="FP58033")]
# train_fps2 = [generate_fingerprint(smi) for smi in tqdm(train_smiles2, desc="FP8196")]

# def tanimoto_coefficient(fp1, fp2):
#     return DataStructs.TanimotoSimilarity(fp1, fp2)

# # 加载测试集
# data4 = pd.read_csv(r'e:\250811\baseline0708\50.csv')
# dsmiles4 = list(data4['SMILES'])
# dadduct4 = list(data4['Adduct'])
# dccs4 = list(data4['True CCS'])

# # 预处理测试集
# sdsmiles4 = []
# for smiles4 in tqdm(dsmiles4, desc="Standardize test SMILES"):
#     mol = Chem.MolFromSmiles(smiles4)
#     smi = Chem.MolToSmiles(mol)
#     sdsmiles4.append(smi)

# dnb4 = nbonds_list(sdsmiles4)

# # 预测流程
# results = []
# for idx in tqdm(range(len(sdsmiles4)), desc="Predicting"):
#     test_smi = sdsmiles4[idx]
#     test_adduct = dadduct4[idx]
#     test_ccs = dccs4[idx]
#     test_nb = dnb4[idx]
#     fp_test = generate_fingerprint(test_smi)

#     # 计算与58033和8196的RSS
#     tc1 = sorted([tanimoto_coefficient(fp_test, fp) for fp in train_fps1], reverse=True)[:5]
#     rss1 = sum(tc1) / 5
#     tc2 = sorted([tanimoto_coefficient(fp_test, fp) for fp in train_fps2], reverse=True)[:5]
#     rss2 = sum(tc2) / 5

#     # 选择RSS高的模型
#     if rss1 >= rss2:
#         selected_model = ccsmodel  # 58033
#         model_name = "58033"
#         # 构造输入
#         tokens = token_smiles(test_smi, dictionary)
#         input_ids = [dictionary.cls_index] + tokens[:248] + [dictionary.sep_index] + [dictionary.pad_index] * (248 - len(tokens))
#         ion_type_to_index = {'[M+H]+': 0, '[M+Na]+': 1, '[M-H]-': 2}
#         ion_index = ion_type_to_index.get(test_adduct, 0)
#         ion_encoding = torch.eye(3)[ion_index].unsqueeze(0).to(device)
#         combined_fp = generate_combined_fp(test_smi).unsqueeze(0).to(device)
#         nb = torch.tensor([test_nb], dtype=torch.long).to(device)
#         input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
#         with torch.no_grad():
#             pred_ccs = selected_model(input_ids, combined_fp, nb, ion_encoding).item()
#     else:
#         selected_model = ccsmodel2  # 8196
#         model_name = "8196"
#         tokens = token_smiles(test_smi, dictionary)
#         input_ids = [dictionary.cls_index] + tokens[:248] + [dictionary.sep_index] + [dictionary.pad_index] * (248 - len(tokens))
#         ion_type_to_index = {'[M+H]+': 0, '[M+Na]+': 1, '[M-H]-': 2}
#         ion_index = ion_type_to_index.get(test_adduct, 0)
#         ion_encoding = torch.eye(3)[ion_index].unsqueeze(0).to(device)
#         combined_fp = generate_combined_fp(test_smi).unsqueeze(0).to(device)
#         nb = torch.tensor([test_nb], dtype=torch.long).to(device)
#         input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
#         with torch.no_grad():
#             pred_ccs = selected_model(input_ids, combined_fp, nb, ion_encoding).item()

#     results.append({
#         "SMILES": test_smi,
#         "Adduct": test_adduct,
#         "True_CCS": test_ccs,
#         "Pred_CCS": pred_ccs,
#         "RSS_58033": rss1,
#         "RSS_8196": rss2,
#         "Selected_Model": model_name
#     })

# # 输出结果为DataFrame
# results_df = pd.DataFrame(results)
# print(results_df)
# results_df.to_csv("testset_predict_with_rss.csv", index=False)

# # 计算并输出预测指标
# y_true = results_df["True_CCS"].astype(float).values
# y_pred = results_df["Pred_CCS"].astype(float).values
# metrics(y_true, y_pred)
