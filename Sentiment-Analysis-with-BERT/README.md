# Twitter Sentiment Analysis using BERT

This repository provides an **end‑to‑end notebook** for fine‑tuning a BERT model on a labelled Twitter‑sentiment dataset. Leveraging `🤗 Transformers`, `PyTorch`, and `scikit‑learn`, the notebook walks you through every step—from data cleaning to model evaluation and saving—so you can quickly reproduce or extend a production‑ready sentiment‑analysis pipeline.

> **Notebook**: [`bert_sentiment.ipynb`](bert_sentiment.ipynb)  
> **Dataset Source**: [Kaggle – Twitter Sentiment Analysis](https://www.kaggle.com/datasets)  

---

## 🔧 Key Features

1. **Multi‑class sentiment support** – Predicts **Positive, Neutral, Negative, and Irrelevant** classes out‑of‑the‑box.  
2. **Minimal setup** – Single Jupyter notebook; no extra Python scripts required.  
3. **Hugging Face Trainer API** – Uses `Trainer` & `TrainingArguments` for streamlined fine‑tuning.  
4. **GPU‑ready** – Automatically detects CUDA for faster training on compatible hardware.  
5. **Custom metrics** – Computes accuracy via a pluggable `compute_metrics` callback.  
6. **Model persistence** – Exports both the fine‑tuned model and tokenizer with `save_pretrained()` for later inference.  

---

## 🧭 Pipeline Flow

1. **Data Load** – Reads `twitter_training.csv` (≈​75 k rows).  
2. **Pre‑processing** – Cleans text, maps labels to integers, drops N/A rows.  
3. **Train ⁄ Test Split** – Uses `train_test_split` (default 80 / 20).  
4. **Tokenisation** – BPE tokenisation with `bert‑base‑uncased`.  
5. **Dataset Wrappers** – Converts to `torch.utils.data.Dataset` objects.  
6. **Fine‑tuning** – Optimises for 2–3 epochs with AdamW on a single GPU/CPU.  
7. **Evaluation** – Reports accuracy on hold‑out test set (≈ 90 % on sample run).  
8. **Save Artifacts** – Writes model & tokenizer to `./sentiment‑bert/`.  

---

## 📈 Sample Output

***** Eval results *****   
epoch = 3   
eval_accuracy = 0.9012   
eval_loss = 0.2904   
eval_runtime = 0:00:18.43   
eval_samples_per_second = 815.4   
eval_steps_per_second = 51.0   

---

## 📂 Project Structure
```bash
├── bert_sentiment.ipynb   # End‑to‑end BERT fine‑tuning workflow
├── requirements.txt       # Python package list
├── README.md              # This file
└── sentiment‑bert/        # (Created after training) saved model & tokenizer
```
---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

--- 

## 🔗 **Links & Contact**

- **GitHub Profile:** [Github](https://github.com/pradeep-kumar8/)
- **LinkedIn:** [Likedin](https://linkedin.com/in/pradeep-kumar8)
- **Email:** [gmail](mailto:pradeep.kmr.pro@gmail.com)
