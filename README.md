<img src="logo.png" width:100px>

# CCSFormer

This is the code repo for the paper *A Substructure-Based Pre-trained Transformer for Highly Generalizable Collision Cross Section Prediction*. We present CCSFormer, a substructure-based Transformer encoder with dual-tasks driven pre-training framework for accurate and generalizable CCS prediction.

## Installation
**1.** Install [Anaconda](https://www.anaconda.com) for python 3.12.13.

**2.** Create an environment and install the required packages.

- Open the command line and create an environment.
	```shell
  conda create --name CCSFormer python=3.12.13
  conda activate CCSFormer
	```
- Clone the repository and enter.
	```shell
  git clone https://github.com/yuxuanliao/CCSFormer.git
  cd CCSFormer
	```
- Install the required packages in [`requirements/pip/requirements.txt`](requirements/pip/requirements.txt) with the following commands.
  ```shell
  python -m pip install -r requirements/pip/requirements.txt
	```
- Or install the required packages by using the [`requirements/conda/environment.yaml`](requirements/conda/environment.yaml) with the following commands.
  ```shell
  conda env create -f requirements/conda/environment.yaml
	```

## CCSFormer
The following files are in the [CCSFormer](CCSFormer) folder:
- [DataConstruction.py](SigmaCCS2/DataConstruction.py). for converting SMILES strings to molecular graph and line graph and using one-hot encoding to encode the adduct ion type.
- [ModelTraining.py](SigmaCCS2/ModelTraining.py). for model training.
- [ModelPrediction.py](SigmaCCS2/ModelPrediction.py). for defining the evaluation metrics and predicting the CCS values by SigmaCCS2.
- [Parameter.py](SigmaCCS2/Parameter.py). for saving relevant parameters including atom radius, atom mass, adduct set, and atoms set.

## Model training
Train the model based on your own training dataset with [SigmaCCS2_train](https://github.com/yuxuanliao/SigmaCCS2/blob/main/SigmaCCS2/ModelTraining.py#L65) function.
    
    SigmaCCS2_train(ifilepath, epochs, batchsize, savepath)

*Parameters*
- ifilepath : file path for storing the dataset to train the model
- epochs : the hyperparameter of the model
- batchsize : the hyperparameter of the model
- savepath : file path where the model is saved

## CCS Prediction
The CCS values of molecules are predicted by using the already trained CCSFormer model with [predict_external](https://github.com/yuxuanliao/SigmaCCS2/blob/main/SigmaCCS2/SigmaCCS2_predict.py#L37) function.

    predict_external(ifilepath, ccs_model_f1, ccs_model_f2, dictionary, ad_index, ofilepath)

*Parameters*
- ifilepath : file path for storing the data of smiles and adducts, which needs to predict CCS values
- ccs_model_f1 : file path where the CCSFormer_base model is stored
- ccs_model_f2 : file path where the CCSFormer_precise model is stored
- dictionary : load the vocabulary
- ad_index : The fingerprints in the training sets of the two models
- ofilepath : file path to save the predicted CCS values

## 🧩 Usage
The example code for CCS prediction is included in [CCSFormer_Prediction.ipynb](CCSFormer_Prediction.ipynb). By directly running [CCSFormer_Prediction.ipynb](CCSFormer_Prediction.ipynb), users can use CCSFormer to predict CCS values of molecules.

## Data
The following datasets of CCSFormer are available:
| Datasets | Size | Description |
| --- | --- | --- |
| [`Pretraining dataset.csv`](data/Pretraining%20dataset.csv) | ~300K | PubChem compounds for molecular structure representation learning |
| [`CCSFormer_base.csv`](data/CCSFormer_base.csv) | ~58K | METLIN-CCS for first-stage CCS fine-tuning and prediction |
| [`CCSFormer_precise.csv`](data/CCSFormer_precise.csv) | ~8K | Curated, cross-platform CCS values for second-stage fine-tuning |
| [`ExternalTestSet1.csv`](data/ExternalTestSet1.csv) | 3906 | Environmental contaminants for independent model evaluation and comparison |
| [`ExternalTestSet2.csv`](data/ExternalTestSet2.csv) | 74 | Doping agents for independent model evaluation and comparison |
| [`training_fingerprint.pkl`](data/training_fingerprint.pkl) | 52982 | The fingerprints in the training sets of the two models |
| [`vocab.pkl`](data/vocab.pkl) | 14073 | The vocabulary is constructed by aggregating the unique atomic substructures |

## CCSFormer Models
📥 **Download**: Due to GitHub’s file size limitation (≤25 MB per file), we host the models (Pretraining_model.pth, CCSFormer_base.pth, and CCSFormer_precise.pth) on [GitHub Releases](https://github.com/yuxuanliao/CCSFormer/releases) | [HuggingFace Models](https://huggingface.co/yuxuanliao/CCSFormer-models/tree/main)

The following specialized checkpoints of CCSFormer are available:
| Checkpoint | Task | Specialization |
| --- | --- | --- |
| [`Pretraining_model.pth`](https://huggingface.co/yuxuanliao/CCSFormer-models/blob/main/Pretraining_model.pth) | Regression | Molecular substructure Transformer model pre-trained on diverse unlabeled datasets |
| [`CCSFormer_base.pth`](https://huggingface.co/yuxuanliao/CCSFormer-models/blob/main/CCSFormer_base.pth) | Regression | Base CCS model fine-tuned on METLIN-CCS |
| [`CCSFormer_precise.pth`](https://huggingface.co/yuxuanliao/CCSFormer-models/blob/main/CCSFormer_precise.pth) | Regression | High-precision model obtained by fine-tuning the base CCS model on a curated dataset |

## 🧪 Web Server

We provide an online **CCSFormer Web Server** that integrates the following core capabilities:
- Individual CCS prediction using adduct type and SMILES string inputs.
- Batch CCS prediction by uploading a CSV file containing SMILES and adducts.
- Retrieve top candidate molecules from an in silico reference database using multidimensional filtering on m/z, RT, and CCS in both individual query mode and batch query mode. 
- Visualize molecular structures interactively and download predicted results in CSV format.

- 🌐 **CCSFormer web server**: The application interface allows users to upload a CSV file containing SMILES and adducts, and access results in real time. Visit the app here: [CCSFormer web server](https://huggingface.co/spaces/yuxuanliao/CCSFormer_v1).

---

### 📫 Contact

For questions or collaboration inquiries, please contact:  
📧 **232303012@csu.edu.cn**

---
