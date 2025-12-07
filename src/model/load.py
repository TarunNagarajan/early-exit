import torch
import torch.nn as nn
import torch.nn.functional as F 
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
    
    # Configure quantization
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
    else:
        bnb_config = None
    
    # Load model
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
    }
    
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
    
    if use_flash_attn:
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except Exception:
            print("Flash Attention 2 not available, using default attention")
    
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded successfully!")
    print(f"  Layers: {model.config.num_hidden_layers}")
    print(f"  Hidden size: {model.config.hidden_size}")
    
    return model, tokenizer

if __name__ == '__main__':
    model, tokenizer = load_model_and_tokenizer()
    input_ids = tokenizer("Hello, how are you?", return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        outputs = model(input_ids)
        print("[BASE MODEL LOADED]")
