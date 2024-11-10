# Required imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import mlflow
import logging
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from flask import Flask, request, jsonify
from flask_caching import Cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from sklearn.metrics import accuracy_score, f1_score
from apscheduler.schedulers.background import BackgroundScheduler
import matplotlib.pyplot as plt
import io
import base64

# Configuration for the model and dataset
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
THRESHOLD = 0.8

# Load the Hugging Face SST-2 Dataset
print("Loading Stanford Sentiment Treebank (SST-2) dataset from Hugging Face...")
ds = load_dataset("stanfordnlp/sst2")

# Select the train and validation splits
train_data = ds["train"]
val_data = ds["validation"]

# Dataset class for sentiment analysis
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten().to(DEVICE),
            'attention_mask': encoding['attention_mask'].flatten().to(DEVICE),
            'label': torch.tensor(label, dtype=torch.long).to(DEVICE)
        }

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Prepare data for training and validation
train_texts = train_data["sentence"]
train_labels = train_data["label"]
val_texts = val_data["sentence"]
val_labels = val_data["label"]

# Initialize train and validation datasets and data loaders
train_dataset = TextDataset(train_texts, train_labels, tokenizer)
val_dataset = TextDataset(val_texts, val_labels, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define the TextClassificationModel
class TextClassificationModel(nn.Module):
    def __init__(self, n_classes=2):
        super(TextClassificationModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=n_classes)

    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).logits

# Training function
def train_model(model, dataloader, optimizer, criterion, num_epochs=NUM_EPOCHS):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_epoch_loss:.4f}")

# Evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return accuracy, f1

# Instantiate and train the model
model = TextClassificationModel(n_classes=2).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

print("Training the model...")
train_model(model, train_dataloader, optimizer, criterion)

# Evaluate model on validation data
print("Evaluating the model on validation data...")
accuracy, f1 = evaluate_model(model, val_dataloader)
print(f"Validation Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

# Inference Pipeline Class
class InferencePipeline:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = DEVICE
        self.model.to(self.device)
        self.model.eval()

    def predict_single(self, text):
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
        return {
            'prediction': int(prediction),
            'probability': float(probabilities.max())
        }

    def predict_batch(self, texts, batch_size=32):
        predictions = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_encodings = self.tokenizer.batch_encode_plus(
                batch_texts,
                add_special_tokens=True,
                max_length=MAX_LENGTH,
                padding='max_length',
                truncation=True,
                return_token_type_ids=False,
                return_attention_mask=True,
                return_tensors='pt',
            )

            input_ids = batch_encodings['input_ids'].to(self.device)
            attention_mask = batch_encodings['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probabilities = torch.softmax(outputs, dim=1)
                batch_predictions = torch.argmax(probabilities, dim=1)

            predictions.extend(batch_predictions.cpu().numpy())
        return predictions

# Initialize the Flask app
app = Flask(__name__)
inference_pipeline = InferencePipeline(model, tokenizer)

# Define Flask routes for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    result = inference_pipeline.predict_single(text)

    if result is None:
        return jsonify({'error': 'Prediction failed'}), 500

    return jsonify(result)

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    data = request.json
    texts = data.get('texts', [])

    if not texts or not isinstance(texts, list):
        return jsonify({'error': 'Texts must be a list of strings'}), 400

    predictions = inference_pipeline.predict_batch(texts)

    if predictions is None:
        return jsonify({'error': 'Batch prediction failed'}), 500

    return jsonify({'predictions': predictions})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
