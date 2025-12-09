import os
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AutoConfig
from transformers.quantization import quantize_dynamic

# Configuration
CHECKPOINT_PATH = './results/checkpoint-645'

def test_model_loading(optimization_method, seq_length):
    """Test loading model with different optimization methods"""
    print(f"\nTesting: {optimization_method}, seq_length={seq_length}")
    print("-" * 50)
    
    try:
        # Load tokenizer with custom sequence length
        tokenizer = DistilBertTokenizer.from_pretrained(CHECKPOINT_PATH)
        tokenizer.model_max_length = seq_length
        print("✓ Tokenizer loaded")
        
        # Load base model
        config = AutoConfig.from_pretrained(CHECKPOINT_PATH)
        config.max_position_embeddings = seq_length
        model = DistilBertForSequenceClassification.from_pretrained(CHECKPOINT_PATH, config=config)
        print("✓ Base model loaded")
        
        # Set to CPU
        device = torch.device('cpu')
        model.to(device)
        model.eval()
        print(f"✓ Model moved to {device}")
        
        # Apply optimization
        if optimization_method == 'torchscript':
            print("Applying TorchScript...")
            # Try both trace and script
            dummy_input = tokenizer("test input", return_tensors="pt", padding='max_length', truncation=True, max_length=seq_length)
            dummy_input = {k: v.to(device) for k, v in dummy_input.items()}
            
            try:
                # First try scripting
                model = torch.jit.script(model)
                print("✓ TorchScript scripting successful")
                model = torch.jit.freeze(model)
                print("✓ TorchScript freezing successful")
            except Exception as e:
                print(f"✗ TorchScript scripting failed: {e}")
                print("Trying tracing instead...")
                try:
                    model = torch.jit.trace(model, (dummy_input['input_ids'], dummy_input['attention_mask']))
                    print("✓ TorchScript tracing successful")
                    model = torch.jit.freeze(model)
                    print("✓ TorchScript freezing successful")
                except Exception as e2:
                    print(f"✗ TorchScript tracing failed: {e2}")
                    return False
        
        elif optimization_method == 'dynamic_quant':
            print("Applying dynamic quantization...")
            try:
                model = quantize_dynamic(
                    model,
                    {torch.nn.Linear},  # Apply quantization to linear layers
                    dtype=torch.qint8  # Quantize to int8
                )
                print("✓ Dynamic quantization successful")
            except Exception as e:
                print(f"✗ Dynamic quantization failed: {e}")
                return False
        
        # Test prediction
        print("Testing prediction...")
        test_text = "This is a test sentence for sentiment prediction"
        
        with torch.no_grad():
            inputs = tokenizer(test_text, padding='max_length', truncation=True, return_tensors="pt", max_length=seq_length)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            if isinstance(model, torch.jit.ScriptModule):
                outputs = model(input_ids, attention_mask)
            else:
                outputs = model(input_ids, attention_mask=attention_mask)
            
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
            prediction = torch.argmax(logits, dim=1).cpu().numpy()[0]
        
        print(f"✓ Prediction successful: {prediction}, probabilities: {probabilities}")
        
        # Check memory usage
        print("Checking memory usage...")
        import psutil
        process = psutil.Process()
        mem_usage = process.memory_info().rss / (1024 * 1024)  # MB
        print(f"✓ Memory usage: {mem_usage:.2f} MB")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Test all configurations
print("Model Loading Debug Script")
print("=" * 50)

# Test baseline models
test_model_loading('baseline', 512)
test_model_loading('baseline', 256)
test_model_loading('baseline', 128)

# Test dynamic quantization
test_model_loading('dynamic_quant', 512)
test_model_loading('dynamic_quant', 256)
test_model_loading('dynamic_quant', 128)

# Test torchscript
test_model_loading('torchscript', 512)
test_model_loading('torchscript', 256)
test_model_loading('torchscript', 128)

print("\n" + "=" * 50)
print("Debug completed")
