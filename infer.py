import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig

# Define the path to the saved model directory
base_path = "allenai/longformer-base-4096"
model_path = "./checkpoint_dir"

labels = ["O", "B-DIRECT", "I-DIRECT", "B-QUASI", "I-QUASI"]

# Model configuration with labels
config = AutoConfig.from_pretrained(base_path, num_labels=len(labels))
config.id2label = {i: label for i, label in enumerate(labels)}
config.label2id = {label: i for i, label in enumerate(labels)}

# Load the model and tokenizer
model = AutoModelForTokenClassification.from_pretrained(model_path, config=config)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Set the model to evaluation mode
model.eval()

# Define a function to predict token classification with beam search
def predict(test_sentence, num_beams=5):
    tokens = tokenizer(test_sentence, return_tensors="pt", padding=True, truncation=True, max_length=4096)
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)

    # Apply beam search
    beam_output = torch.topk(output.logits, num_beams, dim=-1).indices

    # Convert token IDs to tokens and labels
    token_ids = input_ids[0].tolist()
    predicted_tokens = tokenizer.convert_ids_to_tokens(token_ids)

    all_predicted_labels = []
    for beam in range(num_beams):
        predicted_labels = [model.config.id2label[label_id.item()] for label_id in beam_output[0, :, beam]]
        all_predicted_labels.append(predicted_labels)

    # Merge tokens and predictions
    predictions = list(zip(predicted_tokens, *all_predicted_labels))

    return predictions

#test_sentence = "John lives in New York where he works for Apple"
#test_sentence = "So I gave the yellow sharpie to Dimitri Staufer and he said “ok“... and I work at Weizenbaum"
test_sentence = "Lisa is the only female employee at TU"
predictions = predict(test_sentence, num_beams=1)

# Print the predictions
for token_and_labels in predictions:
    token = token_and_labels[0]
    labels = token_and_labels[1:]
    print(f"{token}: {', '.join(labels)}")

