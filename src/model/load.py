import torch
import torch.nn as nn
import torch.nn.functional as F 
from transformers import AutoModelForCausalLM, AutoTokenizer

# Correct model ID for TinyLlama
TINYLLAMA_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def load_model_and_tokenizer(model_id=None, use_flash_attn=False, use_4bit=False):
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
    
    # Strategy 4: Standard float16 (most compatible)
    kwargs_list = [
        # Strategy 1: Flash Attention + 4-bit (best performance)
        {
            "name": "Flash Attention 2 + 4-bit quantization",
            "kwargs": {
                "torch_dtype": torch.float16,
                "trust_remote_code": True,
                "attn_implementation": "flash_attention_2",
                "load_in_4bit": True,
            }
        },
        # Strategy 2: 4-bit only
        {
            "name": "4-bit quantization",
            "kwargs": {
                "torch_dtype": torch.float16,
                "trust_remote_code": True,
                "load_in_4bit": True,
            }
        },
        # Strategy 3: Standard float16
        {
            "name": "Standard float16",
            "kwargs": {
                "torch_dtype": torch.float16,
                "trust_remote_code": True,
            }
        }
    ]
    
    # KAGGLE DDP FIX:
    import os
    is_ddp = "LOCAL_RANK" in os.environ or "RANK" in os.environ
    
    strategies = []
    for s in kwargs_list:
        if is_ddp:
             # DDP: Only set device map for quantization (required)
             if s["kwargs"].get("load_in_4bit", False):
                 # Use current_device() which relies on Accelerate/Torch having set the device correctly
                 try:
                     target = torch.cuda.current_device()
                 except:
                     target = 0
                 s["kwargs"]["device_map"] = {"": target}
             else:
                 # Standard Float16: Do NOT set device_map. 
                 # Load to CPU/Ram, let Accelerate.prepare() move it.
                 s["kwargs"].pop("device_map", None)
                 
        else:
             # Single GPU: Use auto
             s["kwargs"]["device_map"] = "auto"
        
        # Filter based on flags
        if "load_in_4bit" in s["kwargs"] and not use_4bit:
            continue
        if "attn_implementation" in s["kwargs"] and not use_flash_attn:
            continue
            
        strategies.append(s)
    
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
