# Importing the Essential Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer

# Initializing a list of Models
map_of_models = {"BERT": "bert-base-uncased"}
map_of_paths = {"BERT": "BERT_trained_weights\checkpoint-5000"}

# Initializing 

# Initializing the tokenizer
model_name = "BERT"

model = pipeline(task = 'sentiment-analysis', model=map_of_paths[model_name], tokenizer=AutoTokenizer.from_pretrained(map_of_models[model_name]))

print(model("This is not good"))