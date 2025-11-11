import numpy as np
import pandas as pd
from transformers import AutoTokenizer, XLNetForMultipleChoice
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("xlnet/xlnet-base-cased")
model = XLNetForMultipleChoice.from_pretrained("xlnet/xlnet-base-cased")

map_of_datasets = {"Finance": r"/Users/nitheeshsundaram/Downloads/NLP datasets/data.csv",
                   "Mental Health": r"/Users/nitheeshsundaram/Downloads/NLP datasets/Combined Data.csv",
                   "Fake News": r"/Users/nitheeshsundaram/Downloads/NLP datasets/Fake News.csv"}


mode = "Mental Health"  # change to "Finance" or "Genius" to test other datasets

dataset = pd.read_csv(map_of_datasets[mode]).sample(10)

if mode == "Mental Health":
    dataset.rename(columns={"statement": "Sentence"}, inplace=True)
elif mode == "Fake_News":
    dataset.rename(columns={"title": "Sentence"}, inplace=True)

# Initializing the choices
choice_0 = "neutral"
choice_1 = "happiness"
choice_2 = "sadness"
choice_3 = "worry"
choice_4 = "love"

num_sentiment_map = {0: "neutral",
                     1: "happiness",
                     2: "sadness",
                     3: "worry",
                     4: "love"}

for _, prompt in dataset.iterrows():
    labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

    # Prompt engineering start every sentence with "The sentiment of the sentence is:"
    encoding = tokenizer([f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}"],
                         [choice_0, choice_1, choice_2, choice_3, choice_4],
                         return_tensors="pt", padding=True)
    outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels)  # batch size is 1

    # the linear classifier still needs to be trained
    loss = outputs.loss
    logits = outputs.logits
    print(f" For the sentence: {prompt['Sentence']}, \n-------->the Sentiment is: {num_sentiment_map[int(torch.argmax(F.softmax(logits, dim=-1)))]}")


