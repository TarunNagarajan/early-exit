
import torch
import transformers
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding
import inspect

print(f"Transformers version: {transformers.__version__}")

print("\n=== LlamaRotaryEmbedding.forward signature ===")
try:
    print(inspect.signature(LlamaRotaryEmbedding.forward))
except Exception as e:
    print(f"Could not inspect: {e}")

print("\n=== LlamaAttention.forward signature ===")
try:
    print(inspect.signature(LlamaAttention.forward))
except Exception as e:
    print(f"Could not inspect: {e}")
