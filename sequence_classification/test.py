from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

tokenizer = AutoTokenizer.from_pretrained("../saved_models/fill/path/here")
model = AutoModelForQuestionAnswering.from_pretrained("../saved_models/fill/path/here")

nlp = pipeline("sentiment-analysis", model=model)

result = nlp("I hate you")[0]
print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

result = nlp("I love you")[0]
print(f"label: {result['label']}, with score: {round(result['score'], 4)}")