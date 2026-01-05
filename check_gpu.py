import torch

print("=" * 50)
print("GPU DIAGNOSTIC CHECK")
print("=" * 50)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("\nCUDA NOT AVAILABLE!")
    print("Possible reasons:")
    print("1. PyTorch was installed WITHOUT CUDA support")
    print("2. NVIDIA drivers not installed/outdated")
    print("3. CUDA toolkit not installed")
    print("\nTo fix: Install PyTorch with CUDA:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
