#!/usr/bin/env python3
"""
ONE-TIME ENVIRONMENT SETUP FOR COLAB
Run this first, once, then never again
"""

import subprocess
import sys
import os
import torch

def setup_environment():
    print("=" * 80)
    print("🚀 HIERARCHICAL ADAPTIVE TRANSFORMER - ENVIRONMENT SETUP")
    print("=" * 80)
    
    # Check Python version
    print(f"Python: {sys.version}")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.version.cuda}")
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("❌ No CUDA GPU available. This will be SLOW.")
        return False
    
    # Install packages
    print("\n📦 Installing packages...")
    
    packages = [
        "torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        "transformers==4.35.0",
        "accelerate==0.24.1",
        "bitsandbytes==0.41.3",
        "datasets==2.14.5",
        "sentencepiece==0.1.99",
        "wandb==0.16.0",
        "matplotlib==3.7.2",
        "seaborn==0.12.2",
        "einops==0.7.0",
        "flash-attn==2.3.6",
        "tqdm==4.66.1",
        "scikit-learn==1.3.0",
        "pandas==2.1.3",
        "numpy==1.24.3"
    ]
    
    for pkg in packages:
        try:
            print(f"Installing: {pkg.split()[0]}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkg.split())
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Failed to install {pkg}: {e}")
    
    # Create directory structure
    print("\n📁 Creating directory structure...")
    dirs = [
        "src",
        "checkpoints",
        "logs",
        "results",
        "data/wikitext",
        "visualizations"
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  Created: {d}/")
    
    # Test imports
    print("\n🧪 Testing critical imports...")
    try:
        import transformers
        import torch
        import datasets
        from accelerate import Accelerator
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Optimize CUDA settings
    print("\n⚡ Optimizing CUDA settings...")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print("✅ TF32 enabled for faster matmuls")
    print("✅ cuDNN benchmark enabled")
    
    # Final check
    print("\n" + "=" * 80)
    print("🎉 ENVIRONMENT SETUP COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run: python load_model.py --test")
    print("2. Run: python train.py --phase routers --epochs 3")
    print("\nOr for a quick test:")
    print("  python -c 'from src.hierarchical_wrapper import test_integration; test_integration()'")
    
    return True

if __name__ == "__main__":
    success = setup_environment()
    sys.exit(0 if success else 1)