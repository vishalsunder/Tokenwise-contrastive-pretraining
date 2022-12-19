import torch
import random
import json
import ast
import re
import os
import time
import pandas as pd
import pdb
import numpy as np
import torchaudio
import torchaudio.transforms as AT
from tqdm import tqdm
import copy
#from transformers import Wav2Vec2Processor, Wav2Vec2Model
from transformers import BertTokenizer
from speechbrain.processing.features import STFT, spectral_magnitude, Filterbank, Deltas, InputNormalization, ContextWindow

TOK = BertTokenizer.from_pretrained("bert-base-uncased")#("TODBERT/TOD-BERT-JNT-V1")#("bert-base-uncased")
da2num_hvb_path = '/homes/3/sunder.9/hvb/data/da2num.json'
with open(da2num_hvb_path, 'r') as f:
    da2num_hvb = json.load(f)

MAPPING = {'correct':0, 'incorrect':1, 'prompt':2, 'repeat':3, 'skip':4, 'stutter':5, 'tracking':6}

def merge(lst):
    def merge_sub(i, lst):
        x = [i] 
        while i+1 < len(lst) and lst[i+1][0] == '#':
            x.append(i+1)
            i = i+1
        return x,i+1
    x, j = merge_sub(0, lst)
    idx = [x]
    while j < len(lst):
        x, j = merge_sub(j, lst)
        idx.append(x)
    return idx

def simplify_labels(lst):
    lst2 = []
    for l in lst:
        if l != 'correct':
            lst2.append(1)
        else:
            lst2.append(0)
    return lst2

def map_labels(lst):
    lst2 = []
    for l in lst:
        lst2.append(MAPPING[l])
    return lst2

def mhot_tgt(tlist, ncls):
    tgt = torch.zeros(len(tlist), ncls)
    for i,t in enumerate(tlist):
        tgt[i][t] = 1.
    return tgt

def pad(input, factor=4):
    add_size = input.size(0) % factor
    if add_size != 0:
        rem_size = factor - add_size
        return torch.cat([input, torch.zeros(rem_size, input.size(1))], dim=0)
    else:
        return input

def padding(sbatch):
    dim = sbatch[0].size(2)
    lens = [x.size(1) for x in sbatch]
    lmax = max(lens)
    padded = []
    for x in sbatch:
        pad = torch.zeros(lmax, dim)
        pad[:x.size(1),:] = x
        padded.append(pad.unsqueeze(0))
    X = torch.cat(padded, dim=0)
    return X, lens, lmax

def list_batch(X, lens, lmax):
    idx = list(range(len(lens)))
    random.shuffle(idx)
    sbatch_ = []
    for i, l in enumerate(lens):
        sbatch_.append(X[i,:l,:])
    return sbatch_

def clean_str(text):
    text = re.sub('[^A-Za-z0-9\s]+','',text)
    return text.lower().strip()

def add_delta(features, feat):
    DEL = Deltas(input_size=feat)
    features = features.unsqueeze(0)
    delta1 = DEL(features)
    delta2 = DEL(delta1)
    features = torch.cat([features, delta1, delta2], dim=2)
    return features.squeeze(0)

def crop(signal, length):
    length_adj = signal.shape[1] - length
    if length_adj > 0:
        start = random.randint(0, length_adj) if length_adj > 0 else 0
        return signal[:,start:start + length]
    return signal

class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, audio_path, win_len=25, hop_length=10, n_fft=400, n_mels=40, sample_rate=8000):
        self.df = pd.read_csv(csv_path)
        self.audio_path = audio_path
        self.compute_stft = STFT(sample_rate=sample_rate, win_length=win_len, hop_length=hop_length, n_fft=n_fft)
        self.compute_fbanks = Filterbank(n_mels=n_mels)
        self.sr = sample_rate

    def get_filterbanks(self, signal):
        features = self.compute_stft(signal)
        features = spectral_magnitude(features)
        features = self.compute_fbanks(features)
        return features

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        wav, org_sr = torchaudio.load(row['audio_file'])
        if org_sr > self.sr:
            wav = AT.Resample(org_sr, self.sr)(wav)
        return self.get_filterbanks(wav), clean_str(row['utterance'])

class LibFsh(torch.utils.data.Dataset):
    def __init__(self, args, csv_path_lib, csv_path_fsh, audio_path_fsh, frac=1.0, win_len=25, hop_length=10, n_fft=400, n_mels=40, sample_rate=8000):
        self.df_lib = pd.read_csv(csv_path_lib)
        self.df_fsh = pd.read_csv(csv_path_fsh)
        self.df = pd.concat([self.df_lib, self.df_fsh])
        self.df = self.df.sample(frac=frac).reset_index(drop=True)
        self.audio_path_fsh = audio_path_fsh
        self.compute_stft = STFT(sample_rate=sample_rate, win_length=win_len, hop_length=hop_length, n_fft=n_fft)
        self.compute_fbanks = Filterbank(n_mels=n_mels)
        self.sr = sample_rate

    def get_filterbanks(self, signal):
        features = self.compute_stft(signal)
        features = spectral_magnitude(features)
        features = self.compute_fbanks(features)
        return features
    
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        if row['speaker'] != '[NULL]':
            key = str(row['cnum'])+'_'+str(row['utt_id'])
            audio_path = os.path.join(self.audio_path_fsh,f'{key}.npz')
            wav = torch.from_numpy(np.load(audio_path)['a'])
        else:
            wav, org_sr = torchaudio.load(row['audio_file'])
            if org_sr > self.sr:
                wav = AT.Resample(org_sr, self.sr)(wav)
        return self.get_filterbanks(wav), clean_str(row['utterance'])
    
class LibriSpeech(SpeechDataset):
    def __init__(self, args, csv_path, audio_path, win_len=25, hop_length=10, n_fft=400, n_mels=40, sample_rate=8000):
        super(LibriSpeech, self).__init__(csv_path, audio_path, n_mels=n_mels, sample_rate=sample_rate)
        self.args = args

class SWBD(SpeechDataset):
    def __init__(self, args, csv_path, audio_path, win_len=25, hop_length=10, n_fft=400, n_mels=40, sample_rate=8000):
        super(SWBD, self).__init__(csv_path, audio_path, n_mels=n_mels, sample_rate=sample_rate)
        self.nclasses = args.nclasses
        self.args = args
        lines = open('/homes/3/sunder.9/switchboard2/swda/data/label.json').readlines()
        self.lbl_map = {}
        for line in lines:
            line = json.loads(line)
            self.lbl_map[line['text']] = line['label']

    def __getitem__(self, index):
        row = self.df.iloc[index]
        key = str(row['cnum'])+'_'+str(row['utt_id'])
        audio_path = os.path.join(self.audio_path,f'{key}.npz')
        wav = torch.from_numpy(np.load(audio_path)['a'])
        if self.args.pretrain:
            return self.get_filterbanks(wav), clean_str(row['utterance'])
        else:
            return self.get_filterbanks(wav), clean_str(row['utterance']), self.lbl_map[row['label']], self.nclasses

class SNIPS(SpeechDataset):
    def __init__(self, args, csv_path, audio_path, win_len=25, hop_length=10, n_fft=400, n_mels=40, sample_rate=8000):
        super(SNIPS, self).__init__(csv_path, audio_path, n_mels=n_mels, sample_rate=sample_rate)
        self.nclasses = args.nclasses
        self.args = args

    def __getitem__(self, index):
        row = self.df.iloc[index]
        audio_path = os.path.join(self.audio_path,row['audio_file'])
        #audio_path = audio_path.replace('close', 'far')
        wav, org_sr = torchaudio.load(audio_path)
        if org_sr > self.sr:
            wav = AT.Resample(org_sr, self.sr)(wav)
        if self.args.pretrain:
            return self.get_filterbanks(wav), clean_str(row['utterance'])
        else:
            return self.get_filterbanks(wav), clean_str(row['utterance']), row['label'], None, self.nclasses

class FSC(SpeechDataset):
    def __init__(self, args, csv_path, audio_path, win_len=25, hop_length=10, n_fft=400, n_mels=40, sample_rate=8000):
        super(FSC, self).__init__(csv_path, audio_path, n_mels=n_mels, sample_rate=sample_rate)
        self.nclasses = args.nclasses
        self.args = args

    def __getitem__(self, index):
        row = self.df.iloc[index]
        if 'synth' in row['audio_file']:
            audio_path = os.path.join('/data/data26/scratch/sunderv/slurp_synth',row['audio_file'])
        else:
            audio_path = os.path.join(self.audio_path,row['audio_file'])
        wav, org_sr = torchaudio.load(audio_path)
        if org_sr > self.sr:
            wav = AT.Resample(org_sr, self.sr)(wav)
        if self.args.pretrain:
            return self.get_filterbanks(wav), clean_str(row['utterance'])
        else:
            return self.get_filterbanks(wav), clean_str(row['utterance']), row['label'], self.nclasses

class RRACE(SpeechDataset):
    def __init__(self, args, csv_path, audio_path, win_len=25, hop_length=10, n_fft=400, n_mels=40, sample_rate=8000, train=False, avg_dur=64.0):
        super(RRACE, self).__init__(csv_path, audio_path, n_mels=n_mels, sample_rate=sample_rate)
        self.max_len = int(sample_rate*avg_dur)
        self.nclasses = args.nclasses
        self.args = args
        self.train = train

    def __getitem__(self, index):
        row = self.df.iloc[index]
        wav, org_sr = torchaudio.load(row['audio_file'])
        if org_sr > self.sr:
            wav = AT.Resample(org_sr, self.sr)(wav)
        if self.train:
            wav = crop(wav, self.max_len)
        if self.args.pretrain:
            return self.get_filterbanks(wav), clean_str(row['utterance'])
        else:
            return self.get_filterbanks(wav), clean_str(row['utterance']), map_labels(ast.literal_eval(row['label']))

class SLURP(FSC):
    def __init__(self, args, csv_path, audio_path, win_len=25, hop_length=10, n_fft=400, n_mels=40, sample_rate=8000):
        super(SLURP, self).__init__(args, csv_path, audio_path, n_mels=n_mels, sample_rate=sample_rate)

class HVB(SpeechDataset):
    def __init__(self, args, csv_path, audio_path, win_len=25, hop_length=10, n_fft=400, n_mels=40, sample_rate=8000):
        super(HVB, self).__init__(csv_path, audio_path, n_mels=n_mels, sample_rate=sample_rate)
        self.nclasses = args.nclasses
        self.args = args

    def __getitem__(self, index):
        row = self.df.iloc[index]
        key = str(row['cnum'])+'_'+str(row['utt_id'])
        audio_path = os.path.join(self.audio_path,f'{key}.npz')
        wav = torch.from_numpy(np.load(audio_path)['a'])
        if self.args.pretrain:
            return self.get_filterbanks(wav), clean_str(row['utterance'])
        else:
            label = ast.literal_eval(row['dialog_acts'])
            label = [da2num_hvb[x] for x in label]
            return self.get_filterbanks(wav), clean_str(row['utterance']), label, self.nclasses

class CollatorPT(object):
    def __init__(self, args):
        self.args = args

    def __call__(self, lst):
        #lst = sorted(lst, key=lambda tup: len(tup[1]), reverse=True)

        speech_batch = [pad(x[0].squeeze(0), factor=2**self.args.pyr_layer).unsqueeze(0) for x in lst if x[0].size(1) > 2]
        X, lens, lmax = padding(speech_batch)

        text_raw = [x[1] for x in lst if x[0].size(1) > 2]
        text_batch = TOK(text_raw, return_tensors="pt", padding=True, truncation=True)
        text_batch_unpad = TOK(text_raw).input_ids
        text_batch_raw = [torch.tensor(x).long() for x in text_batch_unpad]

        return X, lens, lmax, text_batch, text_batch_raw

class CollatorSLU(object):
    def __init__(self, args):
        self.args = args

    def __call__(self, lst):
        #lst = sorted(lst, key=lambda tup: len(tup[1]), reverse=True)
        ncls = lst[0][-1]

        speech_batch = [pad(x[0].squeeze(0), factor=2**self.args.pyr_layer).unsqueeze(0) for x in lst if x[0].size(1) > 2]
        X, lens, lmax = padding(speech_batch)

        text_raw = [x[1] for x in lst if x[0].size(1) > 2]
        text_batch = TOK(text_raw, return_tensors="pt", padding=True, truncation=True)
        text_batch_unpad = TOK(text_raw).input_ids
        text_batch_raw = [torch.tensor(x).long() for x in text_batch_unpad]

        label_raw = [x[2] for x in lst if x[0].size(1) > 2]
        if self.args.slu_data == 'hvb':
            label_batch = mhot_tgt(label_raw, ncls)
        else:
            label_batch = torch.tensor(label_raw).long()

        return X, lens, lmax, text_batch, text_batch_raw, label_batch

class CollatorKID(object):
    def __init__(self, args):
        self.args = args

    def __call__(self, lst):
        #lst = sorted(lst, key=lambda tup: len(tup[1]), reverse=True)
        speech_batch = [pad(x[0].squeeze(0), factor=2**self.args.pyr_layer).unsqueeze(0) for x in lst if x[0] is not None and x[0].size(1) > 2]
        X, lens, lmax = padding(speech_batch)

        text_raw = [x[1] for x in lst if x[0] is not None and x[0].size(1) > 2]
        text_batch = TOK(text_raw, return_tensors="pt", padding=True, truncation=True)
        text_batch_unpad = TOK(text_raw).input_ids
        text_batch_raw = [torch.tensor(x[1:-1]).long() for x in text_batch_unpad]
        txt_target = torch.cat([torch.tensor(x[2:]).long() for x in text_batch_unpad])

        merge_idx = [merge(TOK.convert_ids_to_tokens(x[1:-1])) for x in text_batch_unpad]
        cls_target = torch.cat([torch.tensor(x[2]) for x in lst if x[0].size(1) > 2])

        return X, lens, lmax, text_batch, text_batch_raw, merge_idx, cls_target, txt_target
