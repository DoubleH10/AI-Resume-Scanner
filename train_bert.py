import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# Load datset
df = pd.read_csv('cleaned_resume_data.csv')

# Preprocess the data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
labels = {label: i for i, label in enumerate(df["category"].unique())}
df["label"] = df["category"].map(labels)

# Split data into train and test sets
train_texts, val_texts, train_labels, val_labels = train_test_split(df["resume_text"], df["label"], test_size=0.2)

# Custom Dataset class
class ResumeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        inputs = tokenizer(self.texts[idx], padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        return {key: val.squeeze(0) for key, val in inputs.items()}, torch.tensor(self.labels[idx])   

# Create datasets
train_dataset = ResumeDataset(train_texts.tolist(), train_labels.tolist())
val_dataset = ResumeDataset(val_texts.tolist(), val_labels.tolist())

# Load BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(labels))

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()

# Save the model
torch.save(model.state_dict(), "bert_resume_classifier.pth")