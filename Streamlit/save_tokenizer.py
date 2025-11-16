import os, json
from transformers import AutoTokenizer

run_dir = r"artifacts\albert-checkpoint-5000"  

TOKENIZER_NAME = "albert-base-v2"
tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
tok.save_pretrained(run_dir)

print("Tokenizer saved to:", run_dir)