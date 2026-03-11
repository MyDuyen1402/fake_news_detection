# Fake News Detection with NLP

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Exploratory Data Analysis](#2-exploratory-data-analysis)
  - [3. Preprocessing](#3-preprocessing)
  - [4. Models](#4-models)
  - [5. Training & Evaluation](#5-training--evaluation)
- [Results](#results)
- [Key Findings](#key-findings)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Authors](#authors)

---

## Overview

Misinformation spreads fast — and distinguishing real news from fake is a genuinely hard problem, even for humans. This project tackles that challenge using a range of NLP models, from hybrid deep learning architectures to large pre-trained transformers, benchmarked on a substantial dataset of nearly 80,000 news articles.

Five models are implemented, trained, and compared head-to-head:

| Model | Type |
|---|---|
| BERT (with Adversarial Training) | Pre-trained Transformer |
| RoBERTa | Pre-trained Transformer |
| TI-CNN (Text-Improved CNN) | Hybrid Deep Learning |
| Capsule Network | Hybrid Deep Learning |
| Bi-LSTM with Attention | Hybrid Deep Learning |

---

## Dataset

The dataset used is the **Misinfo Dataset**, consisting of two separate CSV files:
- `DataSet_Misinfo_FAKE.csv` — fake news articles
- `DataSet_Misinfo_TRUE.csv` — real news articles

These are merged and shuffled into a single combined dataset:

| Statistic | Value |
|---|---|
| Total samples | 78,588 |
| Fake news (label = 0) | ~55% |
| Real news (label = 1) | ~45% |
| Duplicate rows (removed) | 9,984 |
| Missing values | 0 |

The class distribution is slightly imbalanced but not severely so — no resampling was applied to preserve the natural data distribution.

The combined dataset is also hosted on Google Drive and downloaded at runtime via `gdown`.

---

## Project Structure

```
📦 project
 ┣ 📓 [NLP]_Final_Project.ipynb   # Main notebook — full pipeline end to end
 ┣ 📄 DataSet_Misinfo_FAKE.csv    # Raw fake news data
 ┣ 📄 DataSet_Misinfo_TRUE.csv    # Raw real news data
 ┗ 📄 DataSet_Misinfo_Combined.csv  # Merged & shuffled dataset (generated)
```

---

## Pipeline

### 1. Data Preparation

- Loaded two separate CSV files for fake and real news
- Assigned binary labels: `0` for fake, `1` for real
- Concatenated and randomly shuffled into a unified dataframe
- Exported the combined dataset as a `.csv` for reuse

### 2. Exploratory Data Analysis

A full EDA was conducted to understand the data before modeling:

- **Label distribution**: Visualized via bar chart — slight imbalance (~55/45), no correction applied
- **Word clouds**: Generated for both fake and real news. Interestingly, both classes share many common high-frequency words (e.g., *Trump*, *United States*, *year*), suggesting that fake news deliberately mimics the language and topics of real news
- **Top bigrams**: Extracted and visualized the most frequent two-word phrases per class after stopword removal
- **Text length analysis**: Real news articles average significantly more words than fake news, consistent with the intuition that real reporting tends to be more detailed and structured, while fake news is often written to make a quick impression

### 3. Preprocessing

A text cleaning pipeline was applied to all articles:

1. **Deduplication** — removed 9,984 duplicate rows
2. **Lowercasing** — normalized text to lowercase
3. **Special character removal** — stripped all non-alphabetic characters using regex
4. **Tokenization** — split text into individual tokens via NLTK
5. **Stopword removal** — filtered out common English stopwords
6. **Lemmatization** — reduced words to their base form using NLTK's `WordNetLemmatizer`

For Keras-based models, text was further processed with a `Tokenizer` (vocab size: 10,000) and padded to a fixed sequence length of 200 tokens.

For transformer models, tokenization was handled by the respective HuggingFace tokenizers with a max length of 128 tokens.

### 4. Models

#### Transformer Models (HuggingFace)

Both transformer models were fine-tuned for sequence classification using `BertForSequenceClassification` and `RobertaForSequenceClassification`.

Training features include:
- **Mixed precision training** (`torch.cuda.amp`) for faster GPU utilization
- **Gradient accumulation** (2 steps) to effectively double the batch size
- **AdamW optimizer** with weight decay (`lr=2e-5`, `weight_decay=0.01`)
- **Pinned memory & multi-worker DataLoaders** for efficient data throughput

**BERT** additionally uses **adversarial training** via the **Fast Gradient Method (FGM)**, which perturbs the word embeddings during training to improve the model's robustness against subtle text manipulations — a particularly relevant concern for detecting deliberately crafted fake news.

#### Hybrid Deep Learning Models (Keras / TensorFlow)

Three custom architectures were built:

**TI-CNN (Text-Improved CNN)**
- Uses three parallel `Conv1D` branches with kernel sizes 3, 4, and 5 to capture n-gram features at different granularities
- Outputs are max-pooled and concatenated before a dense classification head
- Binary cross-entropy loss, Adam optimizer

**Capsule Network**
- Applies `Conv1D` as a feature extractor, followed by a `PrimaryCapsule` layer (adapted from the original CapsNet for 1D NLP input)
- A `CapsuleLayer` with dynamic routing (3 iterations) preserves spatial relationships between features
- `Length` layer converts capsule vectors to class probabilities
- All custom layers (`CapsuleLayer`, `Length`, `Mask`, `squash`) were re-implemented in-notebook for TF/Keras version compatibility

**Bi-LSTM with Attention**
- Bidirectional LSTM (64 units per direction) captures context from both left and right of each token
- Self-attention is applied over the LSTM outputs to weight the most informative positions
- Global average pooling reduces the attended output, followed by a dense sigmoid classifier

### 5. Training & Evaluation

All models were evaluated on the same 80/20 train-test split (`random_state=42`) using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Training time**

Keras models were trained for **10 epochs** with a batch size of 32.  
Transformer models were fine-tuned for **3 epochs** with a batch size of 16.

---

## Results

| Model | Accuracy | F1-Score | Training Time |
|---|---|---|---|
| **BERT (Adversarial)** | **0.9749** | **0.9750** | ~1447s |
| **RoBERTa** | 0.9746 | 0.9746 | ~1406s |
| Bi-LSTM with Attention | 0.9383 | 0.9378 | ~329s |
| TI-CNN | 0.9468 | 0.9462 | ~108s |
| Capsule Network | 0.9279 | 0.9276 | ~212s |

Performance was also visualized through:
- **Bar charts** comparing accuracy across models
- **Radar charts** showing the Accuracy / F1-Score / (inverted) Training Time trade-off
- **Line charts** tracking training loss per epoch across the three Keras models

---

## Key Findings

- **BERT (Adversarial)** delivered the best overall performance with the highest F1-score (0.9750) and recall (0.9792), making it the most reliable at catching fake news without missing real ones. The adversarial training component contributed meaningfully to its robustness.

- **RoBERTa** was a close second, leading slightly on precision (0.9778) — meaning fewer real news articles were incorrectly flagged as fake.

- **TI-CNN** punches above its weight for a non-pretrained model, achieving an F1 of 0.9462 in only ~108 seconds. It's the go-to choice when compute is a constraint and accuracy requirements are moderate.

- **Bi-LSTM with Attention** outperformed both Capsule Network and sat close to TI-CNN, demonstrating that bidirectional context modeling with attention is a solid strategy even without pretraining.

- **Capsule Network**, while architecturally interesting, underperformed relative to its training cost. Both TI-CNN and Capsule Network showed signs of overfitting when tested on out-of-domain samples, likely due to the absence of pretrained embeddings and limited semantic generalization.

- **Bottom line**: Transformer models (BERT, RoBERTa) are the clear winners for production-grade fake news detection. When computational resources are limited, Bi-LSTM with Attention is the most practical alternative.

---

## Requirements

```
torch
transformers
tensorflow
keras
nltk
scikit-learn
pandas
numpy
matplotlib
seaborn
wordcloud
textblob
gdown
tabulate
tqdm
```

> The Capsule Network layers (`CapsuleLayer`, `Length`, `Mask`) are adapted from [XifengGuo/CapsNet-Keras](https://github.com/XifengGuo/CapsNet-Keras) and re-implemented inline for compatibility with the TensorFlow/Keras version used.

---

## How to Run

1. Clone or download the repository.
2. Install the required packages:
   ```bash
   pip install torch transformers tensorflow keras nltk scikit-learn pandas numpy matplotlib seaborn wordcloud textblob gdown tabulate tqdm
   ```
3. Download NLTK resources (handled automatically in the notebook):
   ```python
   import nltk
   nltk.download('punkt_tab')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```
4. Open `[NLP]_Final_Project.ipynb` in Jupyter or Google Colab.
5. Run all cells sequentially from top to bottom.

> **Note:** Training the transformer models (BERT, RoBERTa) is computationally intensive. A CUDA-enabled GPU is strongly recommended. The notebook automatically detects and uses GPU if available (`torch.device('cuda')`). Google Colab with a T4/A100 runtime is a convenient option.

---

## Authors

| Name | Student ID |
|---|---|
| Ngô Thị Mỹ Duyên | 22280017 |
| Lê Hoàng Uyên Thư | 22280090 |
