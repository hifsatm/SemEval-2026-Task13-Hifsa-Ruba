# SemEval-2026 Task 13: Detecting Machine-Generated Code  
### CodeBERT-Based System for Subtasks A, B, and C

This repository contains our system for SemEval-2026 Task 13: Detecting Machine-Generated Code with Multiple Programming Languages, Generators, and Application Scenarios.

We participate in all three subtasks:

- **Subtask A:** Binary machine-generated code detection  
  (Human vs Machine)
- **Subtask B:** Multi-class authorship detection  
  (Human + 10 LLM families)
- **Subtask C:** Hybrid code detection  
  (Human / Machine / Hybrid / Adversarial)

Our system is based on:

- `microsoft/codebert-base` pretrained model
- Custom fine-tuning on the official dataset
- No external training data 


## Repository Structure
semeval2026-task13/
│
├── data/
│ ├── task_a_train_small.csv
│ ├── task_a_val_small.csv
│ ├── task_b_train_small.csv
│ ├── task_b_val_small.csv
│ ├── task_c_train_small.csv
│ ├── task_c_val_small.csv
│ ├── task_*_test_set_sample.parquet
│
├── src/
│ ├── dataset.py
│ ├── model.py
│ ├── train_task.py
│ ├── predict_task.py
│
├── checkpoints/
│ ├── best_model_task_a.pt
│ ├── best_model_task_b.pt
│ ├── best_model_task_c.pt
│
├── submission_task_a.csv
├── submission_task_b.csv
├── submission_task_c.csv
└── README.md

## Libraries
- torch – for training/evaluating the model
- transformers – CodeBERT + tokenizer
- pandas – reading parquet/CSV, dataframes
- scikit-learn – accuracy, macro-F1
- tqdm – progress bars
- pyarrow – to load .parquet files

## Training

We fine-tune separate models for each subtask.

Example (Task A):

cd src
python train_task.py \
  --task_name task_a \
  --train_path ../data/task_a_train_small.csv \
  --val_path ../data/task_a_val_small.csv \
  --model_name microsoft/codebert-base \
  --epochs 1 \
  --batch_size 8 \
  --out_dir ../checkpoints


## Prediction
python predict_task.py \
  --checkpoint ../checkpoints/best_model_task_a.pt \
  --test_path ../data/task_a_test_set_sample.parquet \
  --output_path ../predictions_task_a.csv

Repeated this for B and C.


## Submission 

We generate:
submission_task_a.csv
submission_task_b.csv
submission_task_c.csv

Each file follows the required format:
id,label
12345,0
12346,1


## Results 
| Subtask | Description       |   Accuracy |   Macro-F1 |
| ------: | ----------------- | ---------: | ---------: |
|       A | Human vs Machine  |   0.9796   |   0.9796   |
|       B | 11-way Authorship |     0.4079 |     0.3686 |
|       C | Hybrid Detection  |     0.6926 |     0.6944 |

- Subtasks B and C are significantly harder
- Only 20,000 training samples used per task due to compute limits


## Model Description

We fine-tune:
- microsoft/codebert-base

Architecture:
- Transformer encoder
- CLS token classification head
- Dropout 0.1
- CrossEntropy loss

Hyperparameters:
- epochs: 1
- batch size: 8
- max length: 256
- learning rate: 2e-5
- optimizer: AdamW
- scheduler: linear warmup


Ablation Study
To understand the contribution of each component in our CodeBERT-based authorship attribution system, we ran a controlled ablation study across all three SemEval Subtasks (A, B, C). Each experiment modifies one component at a time—encoder, input length, training strategy, or data balancing—while keeping everything else constant.

1. Encoder Variants
Change: CodeBERT → Distilled CodeBERTa-small
Finding:
- Minimal impact on Task A
- Improves performance on multi-class Tasks B and C
- Why: Smaller models generalize better under high class diversity.

2. Input Length
Change: 256 tokens → 128 tokens
Finding:
- Consistent performance drop, worst on Task C
- Why: Hybrid/machine-generated snippets contain patterns beyond 128 tokens.

3. Training Strategy
Change: OneCycle LR → Constant LR
Finding:
- Lower accuracy and macro-F1 across all tasks
- Why: OneCycle stabilizes early updates and improves convergence.

4. Data Balancing
Change: Balanced sampling → No class balancing
Finding:
- Task B breaks: model collapses to majority classes
- Accuracy increases but macro-F1 collapses
- Why: 11-class authorship is highly imbalanced and requires sampling.

Key Takeaways
- Encoder capacity affects generalization on multi-class problems.
- Longer input sequences (256) are crucial for stylistic detection.
- OneCycle LR provides more stable single-epoch fine-tuning.
- Balanced sampling is essential, especially for Task B.

This ablation validates the importance of CodeBERT’s full encoder, long input context, LR scheduling, and balanced sampling in building a robust authorship attribution system.


