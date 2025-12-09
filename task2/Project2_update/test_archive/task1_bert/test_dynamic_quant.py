import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AutoConfig

# Configuration
CHECKPOINT_PATH = './results/checkpoint-645'

print("Testing dynamic quantization...")
print("-" * 50)

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(CHECKPOINT_PATH)
tokenizer.model_max_length = 512
print("✓ Tokenizer loaded")

# Load base model
config = AutoConfig.from_pretrained(CHECKPOINT_PATH)
model = DistilBertForSequenceClassification.from_pretrained(CHECKPOINT_PATH, config=config)
print("✓ Base model loaded")

# Set to CPU
device = torch.device('cpu')
model.to(device)
model.eval()
print(f"✓ Model moved to {device}")

# Apply dynamic quantization
try:
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # Apply quantization to linear layers
        dtype=torch.qint8  # Quantize to int8
    )
    print("✓ Dynamic quantization successful")
    
    # Test prediction
    test_text = "This is a positive news story"
    inputs = tokenizer(test_text, padding='max_length', truncation=True, return_tensors="pt")
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = quantized_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        prediction = torch.argmax(logits, dim=1).cpu().numpy()[0]
    
    print(f"✓ Prediction successful: {prediction}, probabilities: {probabilities}")
    
except Exception as e:
    print(f"✗ Dynamic quantization failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTest completed")
