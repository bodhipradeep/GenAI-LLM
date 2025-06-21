from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from transformers import BertTokenizer, BertForSequenceClassification
from pydantic import BaseModel
import torch

app = FastAPI()

tokenizer = BertTokenizer.from_pretrained('bert_model/saved_bert_tokenizer')
model = BertForSequenceClassification.from_pretrained('bert_model/saved_bert_model')

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits).item()

    label_map = {3: "Positive", 2: "Neutral", 1: "Irrelevant", 0: "Negative"}
    return label_map[predicted_class]

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>Sentiment Analyzer</title>
    </head>
    <body>
        <div class="container">
            <h2>Sentiment Analyzer</h2>
            <form action="/predict_web" method="post">
                <input type="textarea" name="text" placeholder="Enter your sentence" required style="padding: 6px; border-radius: 4px;"/><br/>
                <br/>
                <input type="submit" value="Predict" style="padding: 8px 20px; background-color: #007BFF; color: white; border: none; border-radius: 4px; cursor: pointer;"/>
            </form>
        </div>
    </body>
    </html>
    """

@app.post("/predict_web", response_class=HTMLResponse)
async def predict_web(text: str = Form(...)):
    sentiment = predict_sentiment(text)
    return f"""
    <html>
    <head>
        <title>Sentiment Result</title>
    </head>
    <body>
        <div class="container">
            <h2>Prediction Result</h2>
            <div style="margin-top: 10px; font-size: 1.2em;">Sentiment: <strong>{sentiment}</strong></div>
            <a href="/" style="text-decoration: none; color: #007BFF;">Try another</a>
        </div>
    </body>
    </html>
    """
