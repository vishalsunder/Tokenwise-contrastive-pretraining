import torch
import torch.nn as nn
import random
import pdb
from models import *
from data import pad

device = torch.device("cuda")
input = [pad(torch.randn(random.choice(list(range(50,100))), 120)).to(device) for _ in range(32)]
model = Listener(120, 2, device, 6, 12, dropout=0.1)
model = model.to(device)

out, mask = model(input)
pdb.set_trace()
