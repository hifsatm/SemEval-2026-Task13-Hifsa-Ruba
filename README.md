# SemEval-2026-Task13-Hifsa-Ruba

Detecting Machine-Generated Code – Subtasks A & B
Overview

This repository contains our system for detecting machine-generated code and identifying the source generator. It includes:

Data loaders
Training scripts
Model definitions
Prediction scripts
Pretrained checkpoints
Final prediction CSVs


Repo Structure:-
semeval-mgcode/
│
├── data/
│   ├── task_a_train.csv
│   ├── task_a_val.csv
│   ├── task_a_test.csv
│   ├── task_b_train.csv
│   ├── task_b_val.csv
│   ├── task_b_test.csv
│   ├── predictions_task_a.csv
│   ├── predictions_task_b.csv
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train_task.py
│   ├── predict_task.py
│
├── checkpoints/
│   ├── best_model_task_a.pt
│   ├── best_model_task_b.pt
│
└── README.md

Installation: pip install torch transformers pandas scikit-learn tqdm

Training Subtask A: 
python src/train_task.py \
  --task_name task_a \
  --train_path data/task_a_train.csv \
  --val_path data/task_a_val.csv \
  --model_name microsoft/codebert-base \
  --epochs 1 \
  --batch_size 8 \
  --out_dir checkpoints

Training Subtask B:
python src/train_task.py \
  --task_name task_b \
  --train_path data/task_b_train.csv \
  --val_path data/task_b_val.csv \
  --model_name microsoft/codebert-base \
  --epochs 1 \
  --batch_size 8 \
  --out_dir checkpoints

Preddictions Subtask A:
python src/predict_task.py \
  --checkpoint checkpoints/best_model_task_a.pt \
  --test_path data/task_a_test.csv \
  --output_path predictions_task_a.csv

Preddictions Subtask B:
python src/predict_task.py \
  --checkpoint checkpoints/best_model_task_b.pt \
  --test_path data/task_b_test.csv \
  --output_path predictions_task_b.csv

Results:
| Task | Accuracy  | Macro-F1  |
| ---- | --------- | --------- |
| A    | 0.979     | 0.979 |
| B    | 0.893     | 0.222 |


