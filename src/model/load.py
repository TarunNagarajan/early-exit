import torch
import torch.nn as nn
import torch.nn.functional as F 
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.float16,
        bnb_4bit_use_double_quant = True
)

model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B",
    quantization_config = bnb_config,
    device_map = "auto",
    torch_dtype = torch.float16,
    attn_implementation = "flash_attention_2"
)

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B")
tokenizer.pad_token = tokenizer.eos_token

input_ids = tokenizer("Hello, how are you?", return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    outputs = model(input_ids)
    print("[BASE MODEL LOADED]")
