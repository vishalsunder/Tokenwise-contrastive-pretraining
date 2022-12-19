import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from data import *
from models import *
import pdb

def load2gpu(x, device):
    if x is None:
        return x
    if isinstance(x, dict):
        t2 = {}
        for key, val in x.items():
            t2[key] = val.to(device)
        return t2
    if isinstance(x, list):
        y = []
        for v in x:
            y.append(v.to(device))
        return y
    return x.to(device)

class kNN(object):
    def __init__(self, args, loader, device, norm):
        self.args = args
        self.norm = norm
        self.loader = loader
        self.device = device

    def get_embd(self, model):
        model.eval()
        embd_s = []
        embd_b = []
        for X, lens, lmax, tbatch, tbatch_raw in tqdm(self.loader):
            lens_norm = [1.*(x/lmax) for x in lens]
            sbatch = self.norm(X, torch.tensor(lens_norm).float(), epoch=100)
            sbatch = list_batch(sbatch, lens, lmax)
            #sbatch = [add_delta(x, self.args.nspeech_feat) for x in sbatch]
            sbatch, tbatch, tbatch_raw = load2gpu(sbatch, self.device), load2gpu(tbatch, self.device), load2gpu(tbatch_raw,self.device)
            with torch.no_grad():
                s_rep, b_rep, _ = model(sbatch, tbatch, tbatch_raw)
            embd_s.append(s_rep)
            embd_b.append(b_rep)
        return torch.cat(embd_s, dim=0), torch.cat(embd_b, dim=0)

    def get_embd_mlm(self, model):
        model.eval()
        embd_s = []
        embd_b = []
        for X, lens, lmax, tbatch, tbatch_raw in tqdm(self.loader):
            lens_norm = [1.*(x/lmax) for x in lens]
            sbatch = self.norm(X, torch.tensor(lens_norm).float(), epoch=100)
            sbatch = list_batch(sbatch, lens, lmax)
            #sbatch = [add_delta(x, self.args.nspeech_feat) for x in sbatch]
            sbatch, tbatch, tbatch_raw = load2gpu(sbatch, self.device), load2gpu(tbatch, self.device), load2gpu(tbatch_raw,self.device)
            with torch.no_grad():
                _, _, _, mlm_opt, mlm_tgt, mlm_idx = model.forward_mlm(sbatch, tbatch, tbatch_raw)
            embd_s.append(mlm_opt[mlm_idx])
            embd_b.append(mlm_tgt[mlm_idx])
        return torch.cat(embd_s, dim=0), torch.cat(embd_b, dim=0), mlm_idx

    def run(self, model):
        print(f'Encoding dev set for NN search.')
        s_rep, b_rep = self.get_embd(model)
        print(f'Done.')
        assert s_rep.size(0) == b_rep.size(0), f'No. of speech embeddings ({s_rep.size(0)}) != No. of BERT embeddings ({s_rep.size(0)})'
        N = s_rep.size(0)
        s_rep_norm, b_rep_norm = F.normalize(s_rep, dim=1).cpu(), F.normalize(b_rep, dim=1).cpu()
        y_true = list(range(N))
        align = torch.matmul(s_rep_norm, b_rep_norm.t())
        y_pred = torch.max(align, dim=1)[1].tolist()

        return accuracy_score(y_true, y_pred)

    def run_mlm(self, model):
        print(f'Encoding dev set for NN search.')
        s_rep, b_rep, idx = self.get_embd_mlm(model)
        print(f'Done.')
        N = s_rep.size(0)
        s_rep_norm, b_rep_norm = F.normalize(s_rep, dim=1).cpu(), F.normalize(b_rep, dim=1).cpu()
        y_true = list(range(N))
        align = torch.matmul(s_rep_norm, b_rep_norm.t())
        y_pred = torch.max(align, dim=1)[1].tolist()

        return accuracy_score(y_true, y_pred)
