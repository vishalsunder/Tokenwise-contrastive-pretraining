import torch
import pickle
import pandas as pd
import pdb
from tqdm import tqdm

def get_params(model):
    low = []
    high = []
    for name, parameter in model.named_parameters():
        if 'teacher' in name:
            low.append(parameter)
        else:
            high.append(parameter)
    return low, high

def load_dict(model, dict_path):
    pretrained_dict = torch.load(dict_path)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def save_pick(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pick(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save(model, path):
    torch.save(model.state_dict(), path)

def load(model, path):
    model.load_state_dict(torch.load(path))
