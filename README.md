
## 🧠 **English → Tamil Neural Machine Translation**

### *Transformer-based Translation Model with Advanced Training Techniques*

[Transformer Architecture](https://arxiv.org/abs/1706.03762)

> A custom-built Transformer architecture trained from scratch for English → Tamil translation using PyTorch.

---

### 🚀 **Overview**

This project implements a complete **Transformer-based Neural Machine Translation (NMT)** system trained **from scratch** on a large-scale English–Tamil parallel corpus.
The architecture, training loop, and tokenization are all designed manually (no pre-trained models used).

---

### ⚙️ **Key Features and Advanced Techniques**

| Category                       | Technique                           | Description                                                                                                                           |
| ------------------------------ | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Architecture**               | ✨ Custom Transformer                | Encoder–Decoder Transformer (4 layers each, 8 heads, 512-dim) implemented manually using PyTorch.                                     |
| **Tokenization**               | 🧩 Byte-Level BPE                   | Trained custom BPE tokenizers for both English & Tamil using Hugging Face `tokenizers` — handles complex Tamil diacritics & subwords. |
| **Data Handling**              | 📦 Hugging Face Datasets            | Uses `Hemanth-thunder/en_ta` (200k+ sentence pairs) directly from Hugging Face.                                                       |
| **Training Scheduler**         | 🧠 Warmup + Cosine Annealing        | Combines learning rate warmup (stabilizes early training) and cosine annealing decay (for long-term convergence).                     |
| **Loss Function**              | 🎯 Label Smoothing + Ignore Padding | Stabilizes gradients and prevents overconfidence in sequence prediction.                                                              |
| **Gradient Stability**         | 🧮 Gradient Clipping (1.0)          | Prevents exploding gradients during long-sequence training.                                                                           |
| **Model Saving**               | 💾 Auto Checkpointing               | Saves checkpoints every epoch to Google Drive or local system.                                                                        |
| **Decoding**                   | 🌟 Beam Search (optional)           | Significantly improves translation quality and reduces repetition.                                                                    |
| **Mixed Precision (Optional)** | ⚡ AMP Integration                   | Enables faster training with reduced GPU memory (optional).                                                                           |
| **Colab-Ready**                | ☁️ Google Drive Auto-Backup         | Automatically zips and downloads model + tokenizer after training.                                                                    |

---

### 📊 **Training Summary**

| Parameter       | Value                                                                          |
| --------------- | ------------------------------------------------------------------------------ |
| Model Type      | Transformer Encoder–Decoder                                                    |
| Embedding Dim   | 512                                                                            |
| FFN Hidden Dim  | 2048                                                                           |
| Attention Heads | 8                                                                              |
| Encoder Layers  | 4                                                                              |
| Decoder Layers  | 4                                                                              |
| Dropout         | 0.2                                                                            |
| Optimizer       | AdamW                                                                          |
| Learning Rate   | 3e-5 (Warmup + Cosine decay)                                                   |
| Label Smoothing | 0.1                                                                            |
| Batch Size      | 32                                                                             |
| Dataset         | [Hemanth-thunder/en_ta](https://huggingface.co/datasets/Hemanth-thunder/en_ta) |
| Training Time   | ~5 hours (10 epochs on Colab T4 GPU)                                           |

---

### 📦 **Repository Structure**

```
Transformer_mod/
├── model.py                # Transformer architecture (encoder, decoder, attention)
├── train.py                # Training pipeline (scheduler, loss, checkpointing)
├── example.py              # Evaluation & inference script
├── bpe_en_tokenizer.json   # Trained English BPE tokenizer
├── bpe_ta_tokenizer.json   # Trained Tamil BPE tokenizer
├── checkpoint_epoch_*.pth  # Model checkpoints
└── README.md               # Project documentation
```

---

### 🔧 **Setup and Installation**

#### 1️⃣ Clone the repository

```bash
git clone https://github.com/sharaneshwar182007/Transformer_mod.git
cd Transformer_mod
```

#### 2️⃣ Install dependencies

```bash
pip install torch datasets tokenizers tqdm
```

#### 3️⃣ (Optional) Run in Colab with Drive backup

Mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

### 🏋️‍♂️ **Training the Model**

```bash
python train.py
```

This will:

* Load the `Hemanth-thunder/en_ta` dataset
* Train the Transformer for 10 epochs
* Save checkpoints (`/content/drive/MyDrive/EN_TA_Checkpoints/`)
* Automatically download the trained model ZIP file after training

---

### 🧪 **Evaluating and Translating**

After training (or using an existing checkpoint), run:

```bash
python example.py
```

Then enter sentences interactively:

```
Enter the input in English: how are you
Translation in Tamil: நீங்கள் எப்படி இருக்கிறீர்கள்
```

You can exit with `exit`.

---

### 💡 **Advanced Techniques Explained**

#### 🧠 1. Warmup + Cosine Annealing

Smoothly increases LR for first few epochs → avoids unstable gradients.
Then gradually decreases using cosine decay → better long-term convergence.

#### 🧩 2. Byte-Level BPE Tokenization

Tamil has complex script combinations.
Byte-level BPE captures rare and compound characters without splitting words incorrectly.

#### ⚙️ 3. Label Smoothing

Makes training more robust:

```python
nn.CrossEntropyLoss(ignore_index=1, label_smoothing=0.1)
```

#### 🧮 4. Gradient Clipping

Clips gradients to prevent instability:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### 🌈 5. Beam Search Decoding

Improves translation quality by exploring multiple candidate sequences instead of greedy decoding.

---

### 📈 **Expected Training Behavior**

| Epoch | Loss      | Quality                     |
| ----- | --------- | --------------------------- |
| 1–5   | 4.5 → 3.8 | Random Tamil-like tokens    |
| 6–10  | 3.8 → 3.2 | Word fragments appear       |
| 11–15 | 3.2 → 2.8 | Coherent short translations |
| 16–20 | 2.8 → 2.5 | Fluent, meaningful Tamil    |

---

### 🧩 **Future Enhancements**

* ✅ Integrate **BLEU / SacreBLEU** for quantitative evaluation
* ✅ Add **beam search decoding** (top-k or nucleus sampling)
* 🔜 Support **bi-directional translation** (Tamil → English)
* 🔜 Use **mixed-precision** for faster GPU training
* 🔜 Integrate with **Hugging Face Transformers** for deployment

---

### 👨‍💻 **Author**

**G. Sharan Eshwar**
📧 [sharaneshwar182007@gmail.com](mailto:sharaneshwar182007@gmail.com)
🚀 Hobby Researcher in DeepNeural Network,Geospatial AI & Transformer-based Language Models

---

### ⭐ **If you find this project useful**

Please **star 🌟 the repo** on GitHub — it helps others discover it.

---
