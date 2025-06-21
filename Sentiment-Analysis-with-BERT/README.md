# Twitter Sentiment Analysis using BERT | Distilbert

This repository provides an **end‑to‑end notebook** for fine‑tuning a BERT model on a labelled Twitter‑sentiment dataset. Leveraging `🤗 Transformers`, `PyTorch`, and `scikit‑learn`, the notebook walks you through every step—from data cleaning to model evaluation and saving—so you can quickly reproduce or extend a production‑ready sentiment‑analysis pipeline.

> **Notebook**: [`bert_sentiment_test.ipynb`](bert_sentiment_test.ipynb)   
> **Web App**: FastAPI-based UI with real-time sentiment prediction.   
> **Trained Model**: [Saved Trained Model Link](https://drive.google.com/drive/folders/1umKLrIgb8kWCyOU1oqx9T_R_ueq04qn0?usp=sharing)  
> **Dataset Source**: [Kaggle – Twitter Sentiment Analysis](https://www.kaggle.com/datasets)  

---

## 💻 FastAPI Web Interface
Navigate to http://localhost:8000

Enter a sentence → Predict sentiment in real time

### 🖼️ Sample Screenshots
Input Example
<p align="center">
<img src="https://github.com/user-attachments/assets/241d2037-fc56-4860-a749-ab91ffbc9ed9" width="45%"/>
&nbsp;&nbsp;&nbsp;
<img src="https://github.com/user-attachments/assets/edf3bd8b-db77-4a3b-8d29-d09122445b41" width="45%"/>
</p>
Output Predictions
<p align="center">
<img src="https://github.com/user-attachments/assets/3dc211af-d600-4727-82f8-31387faed1ad" width="45%"/> 
&nbsp;&nbsp;&nbsp;
<img src="https://github.com/user-attachments/assets/c7b9f916-0b80-4072-b824-7793a094b2f2" width="45%"/>
</p>

---
## 🔧 Key Features

1. **Multi‑class sentiment support** – Predicts **Positive, Neutral, Negative, and Irrelevant** classes out‑of‑the‑box.  
2. **Minimal setup** – Single Jupyter notebook; no extra Python scripts required.  
3. **Hugging Face Trainer API** – Uses `Trainer` & `TrainingArguments` for streamlined fine‑tuning.
4. **FastAPI Web Interface** – Lightweight web app for real-time predictions.   
5. **Dockerized Deployment** – Easily build, run, and share using Docker.   
6. **GPU‑ready** – Automatically detects CUDA for faster training on compatible hardware.  
7. **Custom metrics** – Computes accuracy via a pluggable `compute_metrics` callback.  
8. **Model persistence** – Exports both the fine‑tuned model and tokenizer with `save_pretrained()` for later inference.  

---

## 🧭 Pipeline Flow

1. **Data Load** – Reads `twitter_training.csv` (≈​75 k rows).  
2. **Pre‑processing** – Cleans text, maps labels to integers, drops N/A rows.  
3. **Train ⁄ Test Split** – Uses `train_test_split` (default 80 / 20).  
4. **Tokenisation** – BPE tokenisation with `bert‑base‑uncased`.  
5. **Dataset Wrappers** – Converts to `torch.utils.data.Dataset` objects.  
6. **Fine‑tuning** – Optimises for 2–3 epochs with AdamW on a single GPU/CPU.  
7. **Evaluation** – Reports accuracy on hold‑out test set (≈ 90 % on sample run).  
8. **Model Saving** – Writes & tokenizer to `./sentiment‑bert/`.
9. **FastAPI App** – Load model/tokenizer and serve predictions via web form.
10. **Dockerized Deployment** – Build and run with simple Docker commands.   

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
├── Data   # Dataset
    ├── twitter_training.csv  
    └── twitter_validation.csv
├── bert_sentiment.ipynb   # End‑to‑end BERT fine‑tuning workflow
├── distilbert_test.ipynb  # Distilbert test notebook
├── Dockerfile             # Docker File
├── main.py                # App using Fastapi 
├── README.md              # This file
└── sentiment‑bert/        # (Created after training) saved model & tokenizer
```
---

## 📄 License

This project is licensed under the [MIT License](../LICENSE).

--- 

## 🔗 **Links & Contact**

- **GitHub Profile:** [Github](https://github.com/pradeep-kumar8/)
- **LinkedIn:** [Likedin](https://linkedin.com/in/pradeep-kumar8)
- **Email:** [gmail](mailto:pradeep.kmr.pro@gmail.com)
