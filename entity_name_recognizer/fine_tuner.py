from pathlib import Path
import re

# Conll format: http://noisy-text.github.io/2017/files/wnut17train.conll
def read_wnut(file_path):
    file_path = Path(file_path)

    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split('\n'):
            token, tag = line.split('\t')
            tokens.append(token)
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)

    return token_docs, tag_docs

texts, tags = read_wnut('../dataset/wnut17train.conll')

print(texts[0][10:17], tags[0][10:17], sep='\n')

from sklearn.model_selection import train_test_split
train_texts, val_texts, train_tags, val_tags = train_test_split(texts, tags, test_size=.2)

unique_tags = set(tag for doc in tags for tag in doc)
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}

from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)

import numpy as np

'''
Now we arrive at a common obstacle with using pre-trained models for token-level classification: many of the tokens in the W-NUT corpus are not in DistilBert’s vocabulary. Bert and many models like it use a method called WordPiece Tokenization, meaning that single words are split into multiple tokens such that each token is likely to be in the vocabulary. For example, DistilBert’s tokenizer would split the Twitter handle @huggingface into the tokens ['@', 'hugging', '##face']. This is a problem for us because we have exactly one tag per token. If the tokenizer splits a token into multiple sub-tokens, then we will end up with a mismatch between our tokens and our labels.

One way to handle this is to only train on the tag labels for the first subtoken of a split token. We can do this in 🤗 Transformers by setting the labels we wish to ignore to -100. In the example above, if the label for @HuggingFace is 3 (indexing B-corporation), we would set the labels of ['@', 'hugging', '##face'] to [3, -100, -100].

Let’s write a function to do this. This is where we will use the offset_mapping from the tokenizer as mentioned above. For each sub-token returned by the tokenizer, the offset mapping gives us a tuple indicating the sub-token’s start position and end position relative to the original token it was split from. That means that if the first position in the tuple is anything other than 0, we will set its corresponding label to -100. While we’re at it, we can also set labels to -100 if the second position of the offset mapping is 0, since this means it must be a special token like [PAD] or [CLS].
'''
def encode_tags(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels

train_labels = encode_tags(train_tags, train_encodings)
val_labels = encode_tags(val_tags, val_encodings)

import torch

class WNUTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_encodings.pop("offset_mapping") # we don't want to pass this to the model
val_encodings.pop("offset_mapping")
train_dataset = WNUTDataset(train_encodings, train_labels)
val_dataset = WNUTDataset(val_encodings, val_labels)

from transformers import DistilBertForTokenClassification
model = DistilBertForTokenClassification.from_pretrained('distilbert-base-cased', num_labels=len(unique_tags))