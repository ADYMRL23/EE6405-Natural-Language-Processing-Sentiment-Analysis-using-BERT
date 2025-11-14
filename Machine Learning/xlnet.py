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
choice_7 = "curiosity"
choice_8 = "desire"
choice_9 = "disappointment"
choice_10 = "disapproval"
choice_11 = "disgust"
choice_12 = "embarrassment"
choice_13 = "excitement"
choice_14 = "fear"
choice_15 = "gratitude"
choice_16 = "greif"
choice_17 = "joy"
choice_18 = "love"
choice_19 = "nervousness"
choice_20 = "optimism"
choice_21 = "pride"
choice_22 = "realization"
choice_23 = "relief"
choice_24 = "remorse"
choice_25 = "sadness"
choice_26 = "surprise"
choice_27 = "neutral"

num_sentiment_map = {0: "admiration",
                     1: "amusement",
                     2: "anger",
                     3: "annoyance",
                     4: "approval",
                     5: "caring",
                     6: "confusion",
                     7: "curiosity",
                     8: "desire",
                     9: "disappointment",
                     10: "disapproval",
                     11: "disgust",
                     12: "embarrassment",
                     13: "excitement",
                     14: "fear",
                     15: "gratitude",
                     16: "greif",
                     17: "joy",
                     18: "love",
                     19: "nervousness",
                     20: "optimism",
                     21: "pride",
                     22: "realization",
                     23: "relief",
                     24: "remorse",
                     25: "sadness",
                     26: "surprise",
                     27: "neutral"
                     }

for _, prompt in dataset.iterrows():
    labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

    # Prompt engineering start every sentence with "The sentiment of the sentence is:"
    encoding = tokenizer([f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}",
                          f"The closest sentiment for {mode} of the sentence is: {prompt['Sentence']}"],
                         [choice_0, 
                          choice_1, 
                          choice_2, 
                          choice_3, 
                          choice_4, 
                          choice_5, 
                          choice_6,
                          choice_7,
                          choice_8,
                          choice_9,
                          choice_10,
                          choice_11,
                          choice_12,
                          choice_13,
                          choice_14,
                          choice_15,
                          choice_16,
                          choice_17,
                          choice_18,
                          choice_19,
                          choice_20,
                          choice_21,
                          choice_22,
                          choice_23,
                          choice_24,
                          choice_25,
                          choice_26,
                          choice_27],
                         return_tensors="pt", padding=True)
    outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels)  # batch size is 1

    # the linear classifier still needs to be trained
    loss = outputs.loss
    logits = outputs.logits
    print(f" For the sentence: {prompt['Sentence']}, \n-------->the Sentiment is: {num_sentiment_map[int(torch.argmax(F.softmax(logits, dim=-1)))]}")


