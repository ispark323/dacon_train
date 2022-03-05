# 농업 환경 변화에 따른 작물 병해 진단 AI 경진대회
병해 피해를 입은 작물 사진과 작물의 생장 환경 데이터를 이용해 작물의 병해를 진단하는 AI 모델을 개발
<br>[Competition Link](https://dacon.io/competitions/official/235870/overview/description)
* 주최: LG AI Research
* 주관: Dacon
* **Private 9th, Score 0.95255**
* **종합평가 최종 5위 (5/344, 1.5%)**
***

## Structure
Train/Test data folder and sample submission file must be placed under **dataset** folder.
```
repo
  |——dataset
        |——train
                |——10027
                  |——10027.csv
                  |——10027.jpg
                  |——10027.json
                |——....
        |——test
                |——10000
                  |——10000.csv
                  |——10000.jpg
                |——....
        |——train.csv
        |——sample_submission.csv
  |——models
        |——model
        |——runners
  |——data
  |——csv_preprocessing
  |——requirements
  |——utils
```
***
## Development Environment
* Ubuntu 18.04.5
* i9-10900X
* RTX 3090 1EA
* CUDA 11.3
***
## Environment Settings

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-3812/)

### Only CPU
```shell
bash install_cpu_dependency.sh
```

### GPU
```shell
bash install_gpu_dependency.sh
```
***
## Run Solution

### 1. CSV File Preprocessing
Extract columns under missing value threshold (90%), save to new_.csv
```shell
python run_csv_preprocessing.py
```

### Result

```
ex)
train——10027———10027.csv
             |—10027.jpg
             |—10027.json
             |—new_10027.csv
```
***
### 2. Train
5 StratifiedKFold Train using Mixup on every 10 steps / Save plot, model arguments, best model spec, inference, model on every fold.
```shell
bash run_train.sh
```

### Result

```
models
      |——saved_model
            |——1
                  |—acc_f1_score.png
                  |—confusion_matrix.png
                  |—loss.png
                  |—model_spec.json
                  |—model.pt
                  |—result.json
                  |—fold1_submission.csv
            |——...
```
***
### 3. Inference
5 process multi-processing / 5 fold ensemble (soft-voting) inference without tta

```shell
bash run_inference.sh
```

### Result
```
models
      |——saved_model
            |——1
                  |—acc_f1_score.png
                  |—confusion_matrix.png
                  |—loss.png
                  |—model_spec.json
                  |—model.pt
                  |—result.json
                  |—fold1_submission.csv
            |——...
            |——submission.csv
```
***
## Download Best Model
You can download our best model weight [Here](https://drive.google.com/file/d/154x-vbFIAQ5NkCf2J2eQaoSp7oJDWdBf/view?usp=sharing).
***
## Inference With Best Model
Best model must be unzipped under **models** folder

### 1. Unzip Best Model
```shell
apt-get install zip unzip
unzip -d models/best_model
chmod -R 777 models/best_model
```

### 2. Inference
```shell
bash run_best_model_inference.sh
```

### 3. Result
```
models
      |——best_model
            |——1
                  |—acc_f1_score.png
                  |—confusion_matrix.png
                  |—loss.png
                  |—model_spec.json
                  |—model.pt
                  |—result.json
                  |—fold1_submission.csv
            |——...
            |——submission.csv
```