# Importing the Essential Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

# Getting the Dataset
training_set = load_dataset('go_emotions')

num_training_samples = 10000

# Initializing the BERTS
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate

# Initializing a list of Models
map_of_models = {"BERT": "bert-base-uncased",
                 "RoBERTa": "FacebookAI/roberta-base"}

# Initializing the tokenizer
model_name = "RoBERTa"
tokenizer = AutoTokenizer.from_pretrained(map_of_models[model_name])
model = AutoModelForSequenceClassification.from_pretrained(map_of_models[model_name], num_labels = 28)
cce = evaluate.load("accuracy")

# Loop to tokenize the data
def tokenize(dataset):
    return tokenizer(dataset['text'], padding='max_length', truncation=True)

# Processing the labels
def return_one_label(dataset):
    dataset['labels'] = dataset['labels'][0]
    return dataset # Only returning the first label of the traub

tokenized_dataset = training_set.map(tokenize, batched=True)
tokenized_dataset = tokenized_dataset.map(return_one_label)

# # Function to compute the metrics
# def compute_metrics(y_pred):
#     logits, labels = y_pred
#     predictions = np.argmax(logits, axis=-1)
#     return cce.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir= f"{model_name}_trained_weights",
    per_device_train_batch_size = 4, 
    per_device_eval_batch_size = 4, 
    num_train_epochs = 2,
    push_to_hub = False
)

# Training the Model
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset= tokenized_dataset['train'].shuffle(seed=14).select(range(num_training_samples)),
    eval_dataset = tokenized_dataset['validation'].shuffle(seed=14).select(range(num_training_samples//2))
)

trainer.train()
print(trainer.evaluate())
