import torch
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict

# Load datset
df = pd.read_csv('cleaned_resume_data.csv')

# Convert job categories (e.g., "Data Scientist") into numerical labels
label_encoder = LabelEncoder()
df["category"] = label_encoder.fit_transform(df['﻿job_position_name'])

# Split data into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["resume_text"], df["category"], test_size=0.2, random_state=42
)
# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize text
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(list(test_labels), truncation=True, padding=True, max_length=512)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_dict({
    "input_ids": train_encodings['input_ids'], 
    'attention_mask':['attention_mask'], 
    'labels': list(train_labels)
    })

test_dataset = Dataset.from_dict({
    "input_ids": test_encodings['input_ids'], 
    'attention_mask':['attention_mask'], 
    'labels': list(test_labels)
    })

# Save the label encoder to decode predictions later
import joblib
joblib.dump(label_encoder, "label_encoder.pkl")

print("✅ Data Preprocessed Successfully! 🚀")