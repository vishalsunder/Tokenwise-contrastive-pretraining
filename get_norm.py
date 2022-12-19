from data import *
from models import *
from util import *
import torch
import pandas as pd
import pdb
import numpy as np
import torchaudio
import argparse
from speechbrain.processing.features import InputNormalization

parser = argparse.ArgumentParser()
parser.add_argument('--lr-text', action='store_true', help='low resource on text')
parser.add_argument('--lr-speech', action='store_true', help='low resource on speech')
parser.add_argument('--pretrain', action='store_true', help='low resource on speech')
parser.add_argument('--nclasses', type=int, default=16, help='')
parser.add_argument('--pyr-layer', type=int, default=2, help='')
parser.add_argument('--slu-data', type=str, default='hvb', help='')

args = parser.parse_args()

data_path = "/homes/3/sunder.9/hvb/data/conv_train.csv"
audio_path = "/data/data24/scratch/sunderv/hvb/"
data = HVB(args, data_path, audio_path, n_mels=80, sample_rate=8000)
norm = InputNormalization(update_until_epoch=3)
collator = CollatorSLU(args)
loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True, num_workers=4, collate_fn=collator)

for epoch in range(1,4):
    for X, lens, lmax, _, _, _ in tqdm(loader):
        lens_norm = [1.*(x/lmax) for x in lens]
        sbatch = norm(X, torch.tensor(lens_norm).float(), epoch=epoch)

save_pick(norm, f'hvb_norm.pkl')
