import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Load dataset
df = pd.read_csv("data/train.csv")
df.columns = df.columns.str.lower()

# Use correct text column
df['text_'] = df['text_'].astype(str)

# FIX LABELS (based on your dataset)
print("Original labels:", df['label'].unique())

df['label'] = df['label'].map({
    'OR': 1,   # Real
    'CG': 0    # Fake
})

df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

print("After mapping:\n", df['label'].value_counts())

# SPEED OPTIMIZATION (very important)
df = df.sample(2000, random_state=42)   # use smaller dataset

# Prepare data
X = df['text_'].tolist()
y = df['label'].tolist()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=128)

train_encodings = tokenize(X_train)
test_encodings = tokenize(X_test)

# Dataset class
class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ReviewDataset(train_encodings, y_train)
test_dataset = ReviewDataset(test_encodings, y_test)

# Load model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# FAST TRAINING CONFIG
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,              # faster
    per_device_train_batch_size=16,  # faster
    logging_dir="./logs",
    save_strategy="no"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Train
trainer.train()

# Save model
model.save_pretrained("models/bert_model")
tokenizer.save_pretrained("models/bert_model")

print("BERT model trained and saved!")
