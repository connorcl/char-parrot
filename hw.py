import torch

force_cpu = False

device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")