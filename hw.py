import torch

force_cpu = False

if torch.cuda.is_available() and force_cpu == False:
    use_gpu = True
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
else:
    use_gpu = False
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor