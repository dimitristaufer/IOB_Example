import json
import pandas as pd
from collections import defaultdict
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np

# Function to parse and duplicate the dataset based on different annotation layers
def parse_and_duplicate_dataset(data):
    duplicated_data = []
    for case in data:
        for annotator, annotations in case['annotations'].items():
            new_case = {
                "text": case["text"],
                "annotations": annotations["entity_mentions"]
            }
            duplicated_data.append(new_case)
    return duplicated_data

# Function to convert annotations to IOB format based on identifier_type
def convert_to_iob(data):
    iob_data = []
    for case in data:
        text = case["text"]
        annotations = case["annotations"]

        # Create a list of labels initialized to 'O' (outside)
        labels = ['O'] * len(text)

        for annotation in annotations:
            start = annotation['start_offset']
            end = annotation['end_offset']
            identifier_type = annotation['identifier_type']

            # Label the start of the entity with 'B-identifier_type'
            labels[start] = f'B-{identifier_type}'
            # Label the rest of the entity with 'I-identifier_type'
            for i in range(start + 1, end):
                labels[i] = f'I-{identifier_type}'

        iob_data.append((text, labels))
    #print(iob_data)
    return iob_data

# Function to tokenize the text and align labels manually
def tokenize_and_align_labels(tokenizer, iob_data):
    tokenized_inputs = []

    for text, labels in tqdm(iob_data, desc="Tokenizing and aligning labels"):
        tokenized_input = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=4096,
            is_split_into_words=False,
            return_offsets_mapping=True
            )
        offsets = tokenized_input['offset_mapping']

        aligned_labels = []
        for idx, (start, end) in enumerate(offsets):
            if start == end:
                aligned_labels.append(-100)
            else:
                # Find the correct label for the token span
                token_labels = labels[start:end]
                # Filter out 'O' labels to get the correct entity label
                entity_labels = [label for label in token_labels if label != 'O']

                if len(entity_labels) > 0:
                    if "NO_MASK" in entity_labels[0]:
                        aligned_labels.append('O') # O = no special entity
                    else:
                        if idx == 0 or (offsets[idx - 1][1] != start):
                            aligned_labels.append(entity_labels[0].replace('I-', 'B-')) # replace I- with B-, i.e. mark the beginning
                        else:
                            aligned_labels.append(entity_labels[0])
                else:
                    aligned_labels.append('O') # O = no special entity

        tokenized_inputs.append({
            'input_ids': tokenized_input['input_ids'],
            'attention_mask': tokenized_input['attention_mask'],
            'labels': aligned_labels
        })

    return tokenized_inputs

# Main processing function
def preprocess_data(data, tokenizer):
    duplicated_data = parse_and_duplicate_dataset(data)
    iob_data = convert_to_iob(duplicated_data)
    tokenized_inputs = tokenize_and_align_labels(tokenizer, iob_data)
    return tokenized_inputs

class ECHRDataset(Dataset):
    def __init__(self, tokenized_inputs): # special method reserved by Python
        self.input_ids = [torch.tensor(inputs['input_ids']) for inputs in tokenized_inputs]
        self.attention_mask = [torch.tensor(inputs['attention_mask']) for inputs in tokenized_inputs]
        self.labels = [torch.tensor([self.label_map(str(input)) for input in inputs['labels']]) for inputs in tokenized_inputs]

    def label_map(self, string_label):
        mapping = {
            '-100': -100,  # ignore in training
            'O': 0,        # no special entity
            'B-DIRECT': 1,
            'I-DIRECT': 2,
            'B-QUASI': 3,
            'I-QUASI': 4,
        }
        return mapping.get(string_label, 0)

    def inverse_label_map(self, int_label):
        inverse_mapping = {
            -100: '-100',  # ignore in training
            0: 'O',        # no special entity
            1: 'B-DIRECT',
            2: 'I-DIRECT',
            3: 'B-QUASI',
            4: 'I-QUASI',
        }
        return inverse_mapping.get(int_label, 'O')

    def __len__(self): # special method reserved by Python
        return len(self.input_ids)

    def __getitem__(self, idx): # special method reserved by Python
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

# Example usage
data = [
    {
        "annotations": {
            "annotator1": {
                "entity_mentions": [
                    {
                        "entity_type": "CODE",
                        "entity_mention_id": "001-61807_a1_em1",
                        "start_offset": 54,
                        "end_offset": 62,
                        "span_text": "36110/97",
                        "edit_type": "check",
                        "identifier_type": "DIRECT",
                        "entity_id": "001-61807_a1_e1",
                        "confidential_status": "NOT_CONFIDENTIAL"
                    },
                    {
                        "entity_type": "ORG",
                        "entity_mention_id": "001-61807_a1_em2",
                        "start_offset": 76,
                        "end_offset": 94,
                        "span_text": "Republic of Turkey",
                        "edit_type": "insert",
                        "confidential_status": "NOT_CONFIDENTIAL",
                        "identifier_type": "NO_MASK",
                        "entity_id": "001-61807_a1_e2"
                    }
                ]
            }
        },
        "text": "PROCEDURE\n\nThe case originated in an application (no. 36110/97) against the Republic of Turkey.",
        "task": "Task: Annotate the document to anonymise the following person: Galip Yalman",
        "meta": {
            "year": 2004,
            "legal_branch": "CHAMBER",
            "articles": [
                91,
                34,
                54,
                34,
                44,
                32,
                34,
                52,
                49,
                34,
                93
            ],
            "countries": "TUR",
            "applicant": "Galip Yalman"
        },
        "doc_id": "001-61807",
        "dataset_type": "test"
    }
]


tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

#f = open("../echr_test.json")
#data = json.load(f)
#f.close()

tokenized_inputs = preprocess_data(data, tokenizer)
train_dataset = ECHRDataset(tokenized_inputs)

print("Input IDs:", train_dataset[0]['input_ids'])
print("Int Labels:", train_dataset[0]['labels'])

print("Input IDs as Tokens:", tokenizer.convert_ids_to_tokens(train_dataset[0]['input_ids']))
print("Int Labels as Names:", [train_dataset.inverse_label_map(label.item()) for label in train_dataset[0]['labels']])
