import torch
import torch.nn as nn
import torch.nn.functional as F 
from transformers import AutoModelForCausalLM, AutoTokenizer

# Correct model ID for TinyLlama
TINYLLAMA_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def load_model_and_tokenizer(model_id=None, use_flash_attn=True, use_4bit=True):
    """
    Load TinyLlama model and tokenizer.
    
    Args:
        model_id: Model identifier (default: TinyLlama-1.1B-Chat-v1.0)
        use_flash_attn: Whether to use Flash Attention 2
        use_4bit: Whether to use 4-bit quantization
    
    Returns:
        model, tokenizer tuple
    """
    if model_id is None:
        model_id = TINYLLAMA_MODEL_ID
    
    print(f"Loading model: {model_id}")
    
    # Try different loading strategies
    model = None
    strategies = []
    
    # Strategy 1: Flash Attention + 4-bit (best performance)
    if use_flash_attn and use_4bit:
        strategies.append({
            "name": "Flash Attention 2 + 4-bit quantization",
            "kwargs": {
                "device_map": "auto",
                "torch_dtype": torch.float16,
                "trust_remote_code": True,
                "attn_implementation": "flash_attention_2",
                "load_in_4bit": True,
            }
        })
    
    # Strategy 2: 4-bit only
    if use_4bit:
        strategies.append({
            "name": "4-bit quantization",
            "kwargs": {
                "device_map": "auto",
                "torch_dtype": torch.float16,
                "trust_remote_code": True,
                "load_in_4bit": True,
            }
        })
    
    # Strategy 3: Flash Attention + float16
    if use_flash_attn:
        strategies.append({
            "name": "Flash Attention 2 + float16",
            "kwargs": {
                "device_map": "auto",
                "torch_dtype": torch.float16,
                "trust_remote_code": True,
                "attn_implementation": "flash_attention_2",
            }
        })
    
    # Strategy 4: Standard float16 (most compatible)
    strategies.append({
        "name": "Standard float16",
        "kwargs": {
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
        }
    })
    
    # Try each strategy until one works
    for strategy in strategies:
        try:
            print(f"  Trying: {strategy['name']}...")
            model = AutoModelForCausalLM.from_pretrained(model_id, **strategy['kwargs'])
            print(f"  ✓ Success: {strategy['name']}")
            break
        except Exception as e:
            print(f"  ✗ Failed: {str(e)[:80]}...")
            model = None
            continue
    
    if model is None:
        raise RuntimeError("Failed to load model with any strategy!")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"\nModel loaded successfully!")
    print(f"  Layers: {model.config.num_hidden_layers}")
    print(f"  Hidden size: {model.config.hidden_size}")
    
    return model, tokenizer


if __name__ == '__main__':
    model, tokenizer = load_model_and_tokenizer()
    input_ids = tokenizer("Hello, how are you?", return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        outputs = model(input_ids)
        print("[BASE MODEL LOADED]")
