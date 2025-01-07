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

# Function to classify multiple sentences and return emotion names
def classify_sentences_with_emotions(sentences):
    """
    Classify a list of sentences and return their corresponding emotion names.

    Args:
        sentences (list): List of input sentences to classify.

    Returns:
        list: Predicted emotion names.
    """
    # Load model, tokenizer, and label mapping
    model, tokenizer = load_model_and_tokenizer()
    label_mapping = load_label_mapping()
    index_to_emotion = {index: emotion for emotion, index in label_mapping.items()}

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

    # Map predictions to emotion names
    emotions = [index_to_emotion[label] for label in predictions]

    return emotions

# Load the label mapping
def load_label_mapping():
    print("Loading label mapping...")
    with open("label_mapping.json", "r") as file:
        label_mapping = json.load(file)
    print("Label mapping loaded successfully!")
    return label_mapping

# # Example usage
# sentences = ["I feel so happy today!", "This is a terrible day.", "I love spending time with my family."]
# predicted_emotions = classify_sentences_with_emotions(sentences)
#
# # Output only the list of emotions
# print(predicted_emotions)
