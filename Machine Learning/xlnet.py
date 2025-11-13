import numpy as np
import pandas as pd
from transformers import AutoTokenizer, XLNetForMultipleChoice
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("xlnet/xlnet-base-cased")
model = XLNetForMultipleChoice.from_pretrained("xlnet/xlnet-base-cased")

map_of_datasets = {"Finance": r"/Users/nitheeshsundaram/Downloads/NLP datasets/data.csv",
                   "Mental Health": r"/Users/nitheeshsundaram/Downloads/NLP datasets/Combined Data.csv",
                   "Fake News": r"/Users/nitheeshsundaram/Downloads/NLP datasets/Fake News.csv",
                   "Emotions": r"/Users/nitheeshsundaram/Downloads/NLP datasets/go_emotions_dataset.csv"}


mode = "Mental Health"  # change to "Finance" or "Genius" or "Emotions" to test other datasets

dataset = pd.read_csv(map_of_datasets[mode]).sample(10)

if mode == "Mental Health":
    dataset.rename(columns={"statement": "Sentence"}, inplace=True)
elif mode == "Fake News":
    dataset.rename(columns={"title": "Sentence"}, inplace=True)

# Initializing the choices
choice_0 = "admiration"
choice_1 = "amusement"
choice_2 = "anger"
choice_3 = "annoyance"
choice_4 = "approval"
choice_5 = "caring"
choice_6 = "confusion"

num_sentiment_map = {0: "admiration",
                     1: "amusement",
                     2: "anger",
                     3: "annoyance",
                     4: "approval",
                     5: "caring",
                     6: "confusion"}

for _, prompt in dataset.iterrows():
    labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

    # Prompt engineering start every sentence with "The sentiment of the sentence is:"
    encoding = tokenizer([f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}"],
                         [choice_0, choice_1, choice_2, choice_3, choice_4, choice_5, choice_6],
                         return_tensors="pt", padding=True)
    outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels)  # batch size is 1

    # the linear classifier still needs to be trained
    loss = outputs.loss
    logits = outputs.logits
    print(f" For the sentence: {prompt['Sentence']}, \n-------->the Sentiment is: {num_sentiment_map[int(torch.argmax(F.softmax(logits, dim=-1)))]}")


