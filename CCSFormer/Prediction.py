# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm
import dill
from sklearn.metrics import mean_squared_error, r2_score
from numpy import mean, median, abs, sum, cumsum, histogram, sqrt
from CCSFormer.PubChemFP import GetPubChemFPs

# ----------------------------
# Config
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# VOCAB_PATH = "data/vocab.pkl"
# MODEL_F1_PATH = "CCSFormer_base.pth"
# MODEL_F2_PATH = "CCSFormer_precise.pth"
# AD_CACHE_PATH = "data/training_fingerprint.pkl"

# # ----------------------------
# # Load vocab and AD index
# # ----------------------------
# with open(VOCAB_PATH, "rb") as f:
#     dictionary = dill.load(f)

# with open(AD_CACHE_PATH, "rb") as f:
#     ad_index = dill.load(f)

# ----------------------------
# Tokenization
# ----------------------------
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
    return [vocab.stoi.get(token, vocab.unk_index) for token in sentence]

def standardize_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)

def build_model_inputs(test_smi, test_adduct, test_nb, dictionary, device):
    tokens = token_smiles(test_smi, dictionary)
    input_ids = [dictionary.cls_index] + tokens[:248] + [dictionary.sep_index] + \
                [dictionary.pad_index] * (248 - len(tokens))
    ion_type_to_index = {"[M+H]+": 0, "[M+Na]+": 1, "[M-H]-": 2}
    ion_index = ion_type_to_index.get(test_adduct, 0)
    ion_encoding = torch.eye(3, device=device)[ion_index].unsqueeze(0)
    combined_fp = generate_combined_fp(test_smi).unsqueeze(0).to(device)
    nb = torch.tensor([test_nb], dtype=torch.long, device=device)
    input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)
    return input_ids, combined_fp, nb, ion_encoding

# ----------------------------
# Fingerprints
# ----------------------------
def smiles_to_ecfps(smiles, radius=2, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return torch.zeros(nBits, dtype=torch.float)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    fp = gen.GetFingerprintAsNumPy(mol)
    return torch.tensor(fp, dtype=torch.float)

def smiles_to_PubChem(smiles):
    # from PubChemFP import GetPubChemFPs
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return torch.zeros(881, dtype=torch.float)
    mol2 = Chem.AddHs(mol)
    return torch.tensor(GetPubChemFPs(mol2), dtype=torch.float)

def smiles_to_Pharmacophere(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return torch.zeros(441, dtype=torch.float)
    fp = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)
    return torch.tensor(fp, dtype=torch.float)

def generate_combined_fp(smiles, radius=2, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return torch.zeros(nBits + 881 + 441, dtype=torch.float)
    fp_morgan = smiles_to_ecfps(smiles, radius, nBits)
    fp_pubchem = smiles_to_PubChem(smiles)
    fp_pharma = smiles_to_Pharmacophere(smiles)
    return torch.cat([fp_morgan, fp_pubchem, fp_pharma], dim=0)

def nbonds_list(smiles):
    nb = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        nb.append(mol.GetNumBonds())
    return nb

# ----------------------------
# AD fingerprint selection
# ----------------------------
def combined_fp_selection(smiles, radius=2, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    try:
        # from PubChemFP import GetPubChemFPs
        pubchem_fp_arr = GetPubChemFPs(Chem.AddHs(mol))
        fp_pubchem = DataStructs.CreateFromBitString("".join(str(int(x)) for x in pubchem_fp_arr))
    except Exception:
        fp_pubchem = None
    if fp_pubchem is not None:
        arr1 = np.zeros((nBits,), dtype=np.int32)
        arr2 = np.zeros((881,), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fp_morgan, arr1)
        DataStructs.ConvertToNumpyArray(fp_pubchem, arr2)
        arr = np.concatenate([arr1, arr2])
        return DataStructs.CreateFromBitString("".join(str(int(x)) for x in arr))
    return fp_morgan

def bulk_ad_score(fp_query, fp_library, k=5):
    sims = DataStructs.BulkTanimotoSimilarity(fp_query, fp_library)
    if len(sims) == 0:
        return 0.0
    sims = sorted(sims, reverse=True)[:min(k, len(sims))]
    return float(np.mean(sims))

# def build_ad_index(train_smiles_stage1, train_smiles_stage2, k=5, radius=2, nBits=1024):
#     fp_lib_stage1 = [combined_fp_selection(s, radius, nBits) for s in tqdm(train_smiles_stage1, desc="Build FP lib S1")]
#     fp_lib_stage1 = [fp for fp in fp_lib_stage1 if fp is not None]

#     fp_lib_stage2 = [combined_fp_selection(s, radius, nBits) for s in tqdm(train_smiles_stage2, desc="Build FP lib S2")]
#     fp_lib_stage2 = [fp for fp in fp_lib_stage2 if fp is not None]

#     thr1 = 0.724799934396318
#     thr2 = 0.586951719976712

#     return {
#         "k": k, "radius": radius, "nBits": nBits,
#         "stage1": {"fp_lib": fp_lib_stage1, "thr": thr1, "name": "58033"},
#         "stage2": {"fp_lib": fp_lib_stage2, "thr": thr2, "name": "8196"},
#     }

# ----------------------------
# Models (inference only)
# ----------------------------
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, en_layers, ff_dim, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, embed_dim, padding_idx=0)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True
            ),
            num_layers=en_layers
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

class CCSPredictor(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
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
        self.to_query = nn.Linear(128, 128)
        self.to_key = nn.Linear(384, 128)
        self.to_value = nn.Linear(384, 128)
        self.ccs_head = nn.Sequential(
            nn.Linear(128 + 128 + 1 + 3, 384),
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
        cls_out = transformer_out[:, 0, :]
        fp_out = self.fp_processor(combined_fp)
        v = self.to_value(fp_out)
        nb = nb.unsqueeze(-1)
        x = torch.cat((cls_out, v, nb, adducts), dim=-1)
        return self.ccs_head(x)

class CCSPredictor3(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x, combined_fp, nb, adducts):
        seq_len = x.size(1)
        position_ids = torch.arange(0, seq_len).unsqueeze(0).to(x.device)
        embedded = self.base_model.base_model.embedding(x) + self.base_model.base_model.position_embedding(position_ids)
        transformer_out = self.base_model.base_model.encoder(embedded)
        cls_out = transformer_out[:, 0, :]
        fp_out = self.base_model.fp_processor(combined_fp)
        v = self.base_model.to_value(fp_out)
        nb = nb.unsqueeze(-1)
        x = torch.cat((cls_out, v, nb, adducts), dim=-1)
        return self.base_model.ccs_head(x)

def load_model(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model

# # ----------------------------
# # Load models once (CCSFormer_base.pth and CCSFormer_precise.pth)
# # ----------------------------
# base_model_f1 = TransformerModel(
#     vocab_size=14073, embed_dim=128, num_heads=4, en_layers=2, ff_dim=512, max_len=250
# ).to(DEVICE)

# ccs_model_f1 = CCSPredictor(base_model_f1).to(DEVICE)
# ccs_model_f1 = load_model(ccs_model_f1, MODEL_F1_PATH)

# base_model_f2 = TransformerModel(
#     vocab_size=14073, embed_dim=128, num_heads=4, en_layers=2, ff_dim=512, max_len=250
# ).to(DEVICE)

# ccs_model_f2_base = CCSPredictor(base_model_f2).to(DEVICE)
# ccs_model_f2 = CCSPredictor3(ccs_model_f2_base).to(DEVICE)
# ccs_model_f2 = load_model(ccs_model_f2, MODEL_F2_PATH)

# ----------------------------
# MC Dropout
# ----------------------------
def enable_mc_dropout_only(module):
    if isinstance(module, nn.Dropout):
        module.train()
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        module.eval()
    for child in module.children():
        enable_mc_dropout_only(child)

@torch.no_grad()
def mc_dropout_predict(model, input_ids, combined_fp, nb, ion_encoding, T=20):
    model.eval()
    enable_mc_dropout_only(model)
    preds = []
    for _ in range(T):
        pred = model(input_ids, combined_fp, nb, ion_encoding).squeeze().item()
        preds.append(pred)
    mean = float(np.mean(preds))
    std = float(np.std(preds, ddof=1)) if len(preds) > 1 else 0.0
    return mean, std


# ----------------------------
# Prediction functions
# ----------------------------
def metrics(y_true, y_pred):   #r2_score(y_true, y_pred)
    RelativeError = [abs(y_pred[i]-y_true[i])/y_true[i] for i in range(len(y_true))]
    R2_Score = r2_score(y_true,y_pred)
    abs_y_err = [abs(y_pred[i]-y_true[i]) for i in range(len(y_true))]
    mae = mean(abs_y_err)
    mdae = median(abs_y_err)
    MSE = mean_squared_error(y_true, y_pred)
    RMSE = sqrt(mean_squared_error(y_true, y_pred))
    print("R2 Score :", R2_Score)
    print("Mean Absolute Error :", mae)
    print("Median Absolute Error :", mdae)
    print("Median Relative Error :", np.median(RelativeError) * 100, '%')
    print("Mean Relative Error :", np.mean(RelativeError) * 100, '%')
    print("Root Mean Squared Error :", RMSE)
    print("Mean Squared Error :", MSE)
    return R2_Score, np.median(RelativeError) * 100


def predict_external(csv_file, ccs_model_f1, ccs_model_f2, dictionary, ad_index, out_path):
    data = pd.read_csv(csv_file)
    smiles_raw = data["SMILES"].tolist()
    adducts_raw = data["Adduct"].tolist()
    true_ccs_raw = data["True CCS"].tolist()

    items = []
    for smi_raw, ad, tccs in zip(smiles_raw, adducts_raw, true_ccs_raw):
        ssmiles = standardize_smiles(smi_raw)
        if ssmiles is not None:
            items.append((smi_raw, ssmiles, ad, tccs))

    results = []
    for smi_raw, smi_std, ad, tccs in tqdm(items, total=len(items), desc="Predicting"):
        nb = nbonds_list([smi_std])[0]
        input_ids, combined_fp, nb_tensor, ion_encoding = build_model_inputs(
            smi_std, ad, nb, dictionary, DEVICE
        )

        mean1, std1 = mc_dropout_predict(ccs_model_f1, input_ids, combined_fp, nb_tensor, ion_encoding, T=20)
        mean2, std2 = mc_dropout_predict(ccs_model_f2, input_ids, combined_fp, nb_tensor, ion_encoding, T=20)

        fp_q = combined_fp_selection(smi_std, radius=ad_index["radius"], nBits=ad_index["nBits"])
        # print(len(fp_q))
        ad1 = bulk_ad_score(fp_q, ad_index["stage1"]["fp_lib"], k=ad_index["k"]) if fp_q is not None else 0.0
        ad2 = bulk_ad_score(fp_q, ad_index["stage2"]["fp_lib"], k=ad_index["k"]) if fp_q is not None else 0.0
        in1 = ad1 >= ad_index["stage1"]["thr"]
        in2 = ad2 >= ad_index["stage2"]["thr"]

        alpha = 0.5
        eps = 1e-8

        if in2 and not in1:
            pred_mean, pred_std, sel = mean2, std2, "8196"
        elif in1 and not in2:
            pred_mean, pred_std, sel = mean1, std1, "58033"
        elif in1 and in2:
            if std1 <= std2:
                pred_mean, pred_std, sel = mean1, std1, "58033"
            else:
                pred_mean, pred_std, sel = mean2, std2, "8196"
        else:
            ad_sum = ad1 + ad2 + eps
            nad1, nad2 = ad1 / ad_sum, ad2 / ad_sum
            inv1, inv2 = 1.0 / (std1 + eps), 1.0 / (std2 + eps)
            inv_sum = inv1 + inv2 + eps
            ninv1, ninv2 = inv1 / inv_sum, inv2 / inv_sum
            w1 = alpha * nad1 + (1 - alpha) * ninv1
            w2 = alpha * nad2 + (1 - alpha) * ninv2
            wsum = w1 + w2 + eps
            w1, w2 = w1 / wsum, w2 / wsum
            pred_mean = float(w1 * mean1 + w2 * mean2)
            var1, var2 = std1**2, std2**2
            pred_var = float(w1 * var1 + w2 * var2 + w1 * (mean1 - pred_mean) ** 2 + w2 * (mean2 - pred_mean) ** 2)
            pred_std = float(np.sqrt(max(pred_var, 0.0)))
            sel = "weighted_ensemble"

        # results.append({
        #     "SMILES": smi_raw,
        #     "Adduct": ad,
        #     "True_CCS": tccs,
        #     "Pred_CCS": pred_mean,
        #     "Pred_Std": pred_std,
        #     "AD_58033": ad1,
        #     "AD_8196": ad2,
        #     "InDomain_58033": in1,
        #     "InDomain_8196": in2,
        #     "Selected_Model": sel,
        # })
        
        results.append({
        "Predicted CCS": pred_mean,
        "True CCS": tccs,
        "Adduct": ad,
        "SMILES": smi_raw
        })

    out_df = pd.DataFrame(results)
    # out_path = "prediction results.csv"
    out_df.to_csv(out_path, index=False)
    print('## CCS predicted completed')
    re_Metrics = metrics(out_df["True CCS"].values, out_df["Predicted CCS"].values)
    # return re_Metrics