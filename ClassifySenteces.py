import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

# Path to the saved model
MODEL_CHECKPOINT_PATH = "./model_checkpoint"

# Load the saved model and tokenizer
def load_model_and_tokenizer():
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT_PATH)
    print("Model and tokenizer loaded successfully!")
    return model, tokenizer

# Function to classify multiple sentences
def classify_sentences(sentences):
    """
    Classify a list of sentences using the pre-trained model and tokenizer.

    Args:
        sentences (list): List of input sentences to classify.

    Returns:
        list: Predicted emotion labels as integers.
        Emotion mapping:
        0 = sadness, 1 = anger, 2 = love, 3 = surprise, 4 = fear, 5 = joy
    """
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Tokenize the input sentences
    inputs = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128  # Use the appropriate max length
    )

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits to predicted labels
    predictions = torch.argmax(outputs.logits, axis=1).cpu().numpy()
    return predictions

# Load the label mapping
def load_label_mapping():
    print("Loading label mapping...")
    with open("label_mapping.json", "r") as file:
        label_mapping = json.load(file)
    print("Label mapping loaded successfully!")
    return label_mapping

# Reverse the mapping to map indices back to emotions
def get_index_to_emotion():
    label_mapping = load_label_mapping()
    return {index: emotion for emotion, index in label_mapping.items()}


''' ##Example##
sentences = ["I feel so happy today!", "This is a terrible day."]
predicted_labels = classify_sentences(sentences)
print(predicted_labels)
'''

