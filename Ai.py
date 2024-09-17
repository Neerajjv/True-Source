from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai-community/roberta-base-openai-detector")
model = AutoModelForSequenceClassification.from_pretrained("openai-community/roberta-base-openai-detector")


# Function to detect AI-written text and provide the probability
def detect_ai_written_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    # Extract the probability of the text being AI-written
    ai_prob = probabilities[0][1].item()  # Probability for AI-written class

    # Prediction based on highest probability
    prediction = torch.argmax(probabilities, dim=-1).item()

    return prediction, ai_prob


# Example usage
text = "Freedom is a concept that transcends borders, cultures, and epochs, holding a revered place in the hearts of individuals and societies alike. It is often described as the state of being free, particularly from oppressive restrictions or control. However, freedom is more than just the absence of constraints; it is the very essence of human dignity and autonomy."
prediction, ai_prob = detect_ai_written_text(text)
print(f"Prediction: {'AI-written' if prediction == 1 else 'Human-written'}")
print(f"Probability of AI-written: {ai_prob:.4f}")
