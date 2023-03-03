import torch

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

