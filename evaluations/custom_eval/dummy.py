import torch
   
if torch.backends.mps.is_available():
    print("MPS is available! PyTorch can use the Apple Silicon GPU.")
else:
    print("MPS is not available. Make sure you're on an Apple Silicon Mac and have the correct PyTorch version installed.")
