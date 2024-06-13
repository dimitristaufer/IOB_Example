import os
import sys
import json
import logging
import torch
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer, TrainingArguments, DataCollatorForTokenClassification, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import preprocess as pp

# NOTE: peft are adapter only models. they dont have config.json. you need to merge it to base model after training manually or use --merge-adapter argument while training.

# Setting environment variables to manage memory
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

logger = logging.getLogger(__name__)

###################
# Hyper-parameters
###################
training_config = {
    "learning_rate": 2e-4, # 5.0e-06,
    "logging_steps": 5,
    "logging_strategy": "steps",
    "lr_scheduler_type": "cosine",
    "num_train_epochs": 2, # 2 in paper
    "max_steps": -1,
    "output_dir": "./checkpoint_dir",
    "overwrite_output_dir": True,
    "per_device_eval_batch_size": 2,  # Adjusted for memory
    "per_device_train_batch_size": 2,  # Adjusted for memory
    "remove_unused_columns": True,
    "save_steps": 50, # 100
    "save_total_limit": 1,
    "seed": 0,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "gradient_accumulation_steps": 1,  # 8, Accumulate gradients to reduce memory usage, simulating batches
    "warmup_ratio": 0.2,
    "fp16": False,  # Disable mixed precision for compatibility
}

train_conf = TrainingArguments(**training_config)

################
# Model Loading
################
checkpoint_path = "allenai/longformer-base-4096"

labels = ["O", "B-DIRECT", "I-DIRECT", "B-QUASI", "I-QUASI"]

# Model configuration with labels
config = AutoConfig.from_pretrained(checkpoint_path, num_labels=len(labels))
config.id2label = {i: label for i, label in enumerate(labels)}
config.label2id = {label: i for i, label in enumerate(labels)}

print("#####")
print(config)
print("#####")

model = AutoModelForTokenClassification.from_pretrained(checkpoint_path, config=config)

# Applying LoRA configuration for efficient fine-tuning
lora_config = LoraConfig(
    r=16, # rank of the low-rank matrices used in LoRA (capacity), default = 16, increase -> more accurate, but slower and more memory
    lora_alpha=32, # scaling factor for the low-rank matrices, default = 32, increase if pre-trained != fine-tuned and decrease if pretrained = fine-tuned
    lora_dropout=0.05, # regularizing the adaptation and preventing overfitting, default = 0.05, increase -> reduce overfitting risk, decrease -> for very large datasets / underfitting
    bias="none", # how to handle biases in the model, default = none, "all" -> Applies LoRA to all bias terms in the model
    task_type="TOKEN_CLS", # TOKEN_CLS -> Token Classification, SEQ_CLS -> Sequence Classification, QA -> Question Answering, GEN -> Generation
    target_modules=["query", "key", "value", "dense"]  # target modules for Longformer
)
model = get_peft_model(model, lora_config)

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

# Setting pad token to unk token if pad token is not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = 'right'


##################
# Data Processing
##################

f = open("../echr_train.json")
train_split = json.load(f)
f.close()

f = open("../echr_test.json")
test_split = json.load(f)
f.close()

f = open("../echr_dev.json")
dev_split = json.load(f)
f.close()

### For Trainer we need input_ids, attention_mask, and labels. ###
tokenized_inputs = pp.preprocess_data(train_split, tokenizer)
train_dataset = pp.ECHRDataset(tokenized_inputs)

tokenized_inputs = pp.preprocess_data(test_split, tokenizer)
test_dataset = pp.ECHRDataset(tokenized_inputs)

tokenized_inputs = pp.preprocess_data(dev_split, tokenizer)
dev_dataset = pp.ECHRDataset(tokenized_inputs)

print(f"Number of train samples: {len(train_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")
print(f"Number of dev samples: {len(dev_dataset)}")

data_collator = DataCollatorForTokenClassification(tokenizer)

###########
# Training
###########
trainer = Trainer(
    model=model,
    args=train_conf,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

train_result = trainer.train()
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()


#############
# Evaluation
#############
metrics = trainer.evaluate()
metrics["eval_samples"] = len(test_dataset)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)


############
# Save model
############
trainer.save_model(train_conf.output_dir)

exit()


###########
# Testing
###########
def predict(test_sentence):
    tokens = tokenizer(test_sentence, return_tensors="pt", padding=True, truncation=True, max_length=4096)
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        output = model(**tokens)
    predictions = torch.argmax(output.logits, dim=-1)
    return predictions

# Example test sentence
test_sentence = "John was 42 when he sent application (no. 36110/97) to the ECHR against the Republic of Turkey."
predictions = predict(test_sentence)
print(predictions)


'''

1 Epoch, 4 batch size, 1 gradient accum

{'train_runtime': 2528.8731, 'train_samples_per_second': 0.44, 'train_steps_per_second': 0.11, 'train_loss': 0.117647497398819, 'epoch': 1.0}

2 Epochs, 2 batch size, 1 gradient accum

{'train_runtime': 4956.637, 'train_samples_per_second': 0.449, 'train_steps_per_second': 0.224, 'train_loss': 0.07836673611091624, 'epoch': 2.0}



'''