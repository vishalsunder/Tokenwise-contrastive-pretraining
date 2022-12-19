import pdb
import random
import numpy as np
import torch
import torch.nn as nn
import os
import copy
import torch.nn.functional as F
from util import *
from transformers import BertForMaskedLM, BertTokenizer
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from pLSTM import *

NEG = -10000000
TOK_NC = BertTokenizer.from_pretrained("bert-base-uncased")

def seqlen_alter(lst, npyr, unred=False): #lst -> [seq1,80; seq2,80; ...]
    fac = 2**npyr
    lst2 = []
    for x in lst:
        timestep = x.size(0)
        feature_dim = x.size(1)
        if not unred:
            lst2.append(x.contiguous().view(int(timestep/fac), feature_dim*fac))
        else:
            lst2.append(x.contiguous().view(timestep*fac, int(feature_dim/fac)))
    return lst2

def MLM_target(tens, frac=0.10): #tens -> [seq1,768; seq2,768; ...]
    mask_id = []
    target = torch.cat(tens, dim=0)
    offset = 0
    for i, ten in enumerate(tens):
        slen = ten.size(0)
        idx = random.sample(list(range(slen)), int(frac*slen))
        try:
            ten[idx] = 0.
        except:
            pdb.set_trace()
        mask_id.extend([i+offset for i in idx])
        offset += slen
    return tens, target, sorted(mask_id)

def clip_bert(tens, mask):
    out_lens = (1-mask).sum(dim=1).tolist()
    out = []
    for i, ten in enumerate(tens):
        out.append(torch.cat([ten[:out_lens[i]-2], ten[out_lens[i]-1:]], dim=0).unsqueeze(0))
    return torch.cat(out, dim=0)

def merge(tens, merge_idx):
    out = []
    for bat, idx_bat in enumerate(merge_idx):
        for seq, idx_seq in enumerate(idx_bat):
            out.append(tens[bat][idx_seq].mean(dim=0, keepdim=True))
    return torch.cat(out, dim=0)

def merge_keep_seq(tens, merge_idx):
    out = []
    for bat, idx_bat in enumerate(merge_idx):
        seq = []
        for _, idx_seq in enumerate(idx_bat):
            seq.append(tens[bat][idx_seq].mean(dim=0, keepdim=True))
        out.append(torch.cat(seq, dim=0))
    out = pack_sequence(out, enforce_sorted=False)
    out, lens = pad_packed_sequence(out, batch_first=True)
    return out, get_mask(lens)

def extract(tens, mask, offset=0):
    out_lens = (1-mask).sum(dim=1).tolist()
    out = []
    for i, ten in enumerate(tens):
        out.append(ten[:out_lens[i]-offset])
    return torch.cat(out, dim=0)

def seq2list(tens, mask, offset=0):
    out_lens = (1-mask).sum(dim=1).tolist()
    out = []
    for i, ten in enumerate(tens):
        out.append(ten[:out_lens[i]-offset])
    return out

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_mask(lens):
    mask = torch.ones(len(lens), max(lens))
    for i, l in enumerate(lens):
        mask[i][:l] = 0.
    return mask

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, device=None, nlayer=1, dropout=0.1):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, nlayer, bidirectional=True, dropout=dropout)
        self.nlayer = nlayer
        self.device = device

    def forward(self, input):
        pack = pack_sequence(input, enforce_sorted=False)
        output, _ = self.rnn(pack)
        output_padded, lens = pad_packed_sequence(pack)
        mask = get_mask(list(lens))
        return output_padded , mask.to(self.device)

class Attention(nn.Module):
    def __init__(self, input_dim, nhead, dim_feedforward=2048, dropout=0.1):
        super(Attention, self).__init__()
        self.self_attn = nn.MultiheadAttention(input_dim, nhead, dropout=dropout)

        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        self.dropout = nn.Dropout(dropout) 

    def forward(self, Q, K, mask):
        src, attn = self.self_attn(Q, K, K, key_padding_mask=mask)
        ## Add and norm
        src = Q + self.dropout(src)
        src = self.norm1(src)
        ## MLP
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        ## Add and norm
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src, attn

class Listener(nn.Module):
    def __init__(self, input_dim, pyr_layer, device, nlayer, nhead, enc_type='lstm', dropout=0.1):
        super(Listener, self).__init__()
        # Speech feature extractor
        self.pBLSTM = pLSTM(input_dim, pyr_layer, 'LSTM', device=device, dropout=dropout) 
        self.bottle = nn.Linear((2**pyr_layer)*input_dim, 768)
        self.norm = nn.LayerNorm(768)
        # Text feature extractor
        self.tembed = BERTNC()
        self.tencoder = CustomLSTM(768, 1, 'LSTM', device=device, dropout=dropout)
        self.tnorm = nn.LayerNorm(768)

        if enc_type == 'lstm':
            self.encoder = CustomLSTM(768, nlayer, 'LSTM', device=device, dropout=dropout)
            self.self_attn = Attention(768, nhead, dropout=dropout)
        else:
            self.encoder = CustomXMER(768, nlayer, device=device, dropout=dropout)
        self.enc_type = enc_type
        self.dropout = nn.Dropout(dropout)
        self.pyr_layer = pyr_layer

    def forward(self, input, is_train=False, mlm=False):
        #if mlm:
        #    input_red = seqlen_alter(input, self.pyr_layer, unred=False)
        #    input_red, target, mask_id = MLM_target(input_red)
        #    input = seqlen_alter(input_red, self.pyr_layer, unred=True)
        #else:
        #    target, mask_id = None, None
        # pyramid layers
        out_pyr, lens = self.pBLSTM(input)
        # --> 768
        in_enc_pad = self.norm(self.bottle(out_pyr)) #seq, bsz, 768
        # convert to list
        in_enc_list = [in_enc_pad.permute(1,0,2)[i][:l] for i, l in enumerate(list(lens))]
        if mlm:
            in_enc_list, target, mask_id = MLM_target(in_enc_list)
        else:
            target, mask_id = None, None
        # remaining layers
        out_enc, mask = self.encoder(in_enc_list, is_train=is_train)
        if self.enc_type == 'lstm':
            out_enc, _ = self.self_attn(out_enc, out_enc, mask.bool())
        return out_enc, mask, target, mask_id

    def forward_text(self, input, is_train=False, mlm=False):
        embed_out, mask1 = self.tembed(input)
        lens1 = (1-mask1.long()).sum(dim=1).tolist()
        embed_list = [embed_out.permute(1,0,2)[i][:l] for i, l in enumerate(list(lens1))]
        rnn_out, mask2 = self.tencoder(embed_list, is_train=False)
        rnn_out = self.tnorm(rnn_out)
        lens2 = (1-mask2.long()).sum(dim=1).tolist()
        in_enc_list = [rnn_out.permute(1,0,2)[i][:l] for i, l in enumerate(list(lens2))]

        if mlm:
            in_enc_list, target, mask_id = MLM_target(in_enc_list, frac=0.15)
        else:
            target, mask_id = None, None
        # remaining layers
        out_enc, mask = self.encoder(in_enc_list, is_train=is_train)
        if self.enc_type == 'lstm':
            out_enc, _ = self.self_attn(out_enc, out_enc, mask.bool())
        return out_enc, mask, target, mask_id

class SubWReader(nn.Module):
    def __init__(self, vocab_size, device, pos_size=512):
        super(SubWReader, self).__init__()
        self.word_embed = nn.Embedding(vocab_size, 768)
        self.pos_embed = nn.Embedding(pos_size, 768)
        self.norm = nn.LayerNorm(768)
        self.device = device

    def forward(self, input):
        we_list = [self.word_embed(x) for x in input]
        we_pack = pack_sequence(we_list, enforce_sorted=False)
        we_out, lens = pad_packed_sequence(we_pack)
        mask = get_mask(list(lens))
        
        pos_list = [self.pos_embed(torch.arange(len(x)).to(self.device)) for x in input]
        pos_pack = pack_sequence(pos_list, enforce_sorted=False)
        pos_out, _ = pad_packed_sequence(pos_pack)

        out = self.norm(we_out + pos_out)
        
        return out, mask.to(self.device)

class BERTNC(nn.Module):
    def __init__(self):
        super(BERTNC, self).__init__()
        self.encoder = BertForMaskedLM.from_pretrained("bert-base-uncased", output_hidden_states=True).bert.embeddings
        
    def forward(self, inputs):
        return self.encoder(inputs.input_ids).permute(1,0,2), 1. - inputs.attention_mask.float()

class Teacher(nn.Module):
    def __init__(self):
        super(Teacher, self).__init__()
        model = BertForMaskedLM.from_pretrained("bert-base-uncased", output_hidden_states=True)
        self.encoder = model.bert
        
    def forward(self, inputs):
        output = self.encoder(**inputs)
        return output.last_hidden_state.permute(1,0,2), 1. - inputs.attention_mask

    def forward_full(self, inputs):
        output = self.encoder(**inputs)
        return output.hidden_states[-3].permute(1,0,2), output.last_hidden_state.permute(1,0,2), 1. - inputs.attention_mask

class MLP(nn.Module):
    def __init__(self, pyr_layer, input_dim):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(768, 768)
        self.l2 = nn.Linear(768, 768)#input_dim*(2**pyr_layer))
        self.dropout = nn.Dropout(0.1)

    def forward(self, input):
        out1 = F.relu(self.l1(self.dropout(input)))
        out2 = self.l2(self.dropout(out1))
        return out2

class PTModel(nn.Module):
    def __init__(self, config):
        super(PTModel, self).__init__()
        self.listener = Listener(config['input_dim'], config['pyr_layer'], config['device'], config['nlayer'], config['nhead'], enc_type=config['enc_type'], dropout=config['dropout'])
        self.sw_reader = BERTNC()#SubWReader(config['vocab_size'], config['device'])
        self.cross_attn = Attention(768, config['nhead'], dropout=config['dropout'])
        self.back_map = MLP(config['pyr_layer'], config['input_dim'])#nn.Linear(768, 768)

        self.back_map_t = MLP(config['pyr_layer'], config['input_dim'])#nn.Linear(768, 768)

        self.teacher = Teacher()
        self.dropout = nn.Dropout(config['dropout'])

    def forward_st(self, input_s, input_t, input_t_raw, is_train=False):
        speech, mask_s, mlm_tgt_s, mlm_id_s = self.listener(input_s, is_train=is_train, mlm=True)
        text, mask_t, mlm_tgt_t, mlm_id_t = self.listener.forward_text(input_t, is_train=is_train, mlm=True)
        speech_in, text_in = self.back_map(self.dropout(speech)), self.back_map_t(self.dropout(text))
        mlm_opt_s, mlm_opt_t = extract(speech_in.permute(1,0,2), mask_s.long()), extract(text_in.permute(1,0,2), mask_t.long())
        assert mlm_opt_s.size(0) == mlm_tgt_s.size(0)
        assert mlm_opt_t.size(0) == mlm_tgt_t.size(0)

        ncon_word, mask_nc = self.sw_reader(input_t)#self.sw_reader(input_t_raw)

        con_word_s, attn_s = self.cross_attn(ncon_word, speech, mask_s.bool())
        con_word_t, attn_t = self.cross_attn(ncon_word, text, mask_t.bool())
        with torch.no_grad():
            oracle, mask_b = self.teacher(input_t)

        speech_rep = extract(con_word_s.permute(1,0,2), mask_nc.long())
        text_rep = extract(con_word_t.permute(1,0,2), mask_nc.long())
        bert_rep = extract(oracle.permute(1,0,2), mask_nc.long())

        return speech_rep, text_rep, bert_rep, mlm_opt_s, mlm_tgt_s, mlm_id_s, mlm_opt_t, mlm_tgt_t, mlm_id_t

    def forward_mlm(self, input_s, input_t, input_t_raw, is_train=False):
        speech, mask_s, mlm_tgt, mlm_id = self.listener(input_s, is_train=is_train, mlm=True)

        speech_in = self.back_map(self.dropout(speech))
        mlm_opt = extract(speech_in.permute(1,0,2), mask_s.long())
        assert mlm_opt.size(0) == mlm_tgt.size(0)

        ncon_word, mask_t = self.sw_reader(input_t)#self.sw_reader(input_t_raw)

        con_word, attn = self.cross_attn(ncon_word, speech, mask_s.bool())
        with torch.no_grad():
            oracle, mask_t2 = self.teacher(input_t)

        speech_rep = extract(con_word.permute(1,0,2), mask_t.long())
        bert_rep = extract(oracle.permute(1,0,2), mask_t.long())

        return speech_rep, bert_rep, attn, mlm_opt, mlm_tgt, mlm_id

    def forward(self, input_s, input_t, input_t_raw, is_train=False):
        speech, mask_s, _, _ = self.listener(input_s, is_train=is_train)
        ncon_word, mask_t = self.sw_reader(input_t)#self.sw_reader(input_t_raw)

        con_word, attn = self.cross_attn(ncon_word, speech, mask_s.bool())
        with torch.no_grad():
            oracle, mask_t2 = self.teacher(input_t)

        speech_rep = extract(con_word.permute(1,0,2), mask_t.long())
        bert_rep = extract(oracle.permute(1,0,2), mask_t.long())

        return speech_rep, bert_rep, attn

class SLUModel(nn.Module):
    def __init__(self, config):
        super(SLUModel, self).__init__()
        self.listener = Listener(config['input_dim'], config['pyr_layer'], config['device'], config['nlayer'], config['nhead'], enc_type=config['enc_type'], dropout=config['dropout'])
        self.sw_reader = BERTNC()#SubWReader(config['vocab_size'], config['device'])
        self.cross_attn = Attention(768, config['nhead'], dropout=config['dropout'])
        self.cls_layer = nn.Linear(768, config['nclasses'])
        self.dropout = nn.Dropout(config['dropout'])
        self.device = config['device']

    def forward(self, input_s, is_train=False):
        speech, mask_s, _, _ = self.listener(input_s, is_train=is_train)

        bsz = len(input_s)
        #query = [torch.tensor([101]).long().to(self.device) for _ in range(bsz)]
        #query, _ = self.sw_reader(query)
        #
        query = TOK_NC(['[CLS]' for _ in range(bsz)], return_tensors="pt", padding=True, truncation=True).to(self.device)
        query, _ = self.sw_reader(query)
        query = query[0].unsqueeze(0)
        #
        speech_cls, _ = self.cross_attn(query, speech, mask_s.bool())
        return speech_cls.squeeze(0)

    def forward_st(self, input_s, input_t, is_train=False, speech=True, text=True):
        speech_cls = None
        if speech:
            speech, mask_s, mlm_tgt_s, mlm_id_s = self.listener(input_s, is_train=is_train)
            bsz_s = len(input_s)
            query_s = TOK_NC(['[CLS]' for _ in range(bsz_s)], return_tensors="pt", padding=True, truncation=True).to(self.device)
            query_s, _ = self.sw_reader(query_s)
            query_s = query_s[0].unsqueeze(0)
            speech_cls, _ = self.cross_attn(query_s, speech, mask_s.bool())
            speech_cls = speech_cls.squeeze(0)

        text_cls = None
        if text:
            text, mask_t, mlm_tgt_t, mlm_id_t = self.listener.forward_text(input_t)
            bsz_t = len(input_t)
            query_t = TOK_NC(['[CLS]' for _ in range(bsz_t)], return_tensors="pt", padding=True, truncation=True).to(self.device)
            query_t, _ = self.sw_reader(query_t)
            query_t = query_t[0].unsqueeze(0)
            text_cls, _ = self.cross_attn(query_t, text, mask_t.bool())
            text_cls = text_cls.squeeze(0)

        return speech_cls, text_cls

    def classify(self, input_s):
        return self.cls_layer(self.dropout(input_s))

class KidModel(nn.Module):
    def __init__(self, config):
        super(KidModel, self).__init__()
        self.listener = Listener(config['input_dim'], config['pyr_layer'], config['device'], config['nlayer'], config['nhead'], dropout=config['dropout'])
        self.sw_reader = SubWReader(config['vocab_size'], config['device'])
        self.cross_attn = Attention(768, config['nhead'], dropout=config['dropout'])
        self.rnn_cls = nn.LSTM(768, 768, 1, bidirectional=True)
        self.classifier = nn.Linear(2*768, config['nclasses'])
        self.classifier_txt = nn.Linear(768, 30522)
        #self.teacher = Teacher()

    def mfw(self, X, label, gamma=12350.35015119583, mu=429.7652650998209, beta=0.01):
        mapping = {0:35688, 1:325, 2:1118, 3:567, 4:271, 5:151, 6:9}
        lam = torch.from_numpy(np.random.beta(2.0, 2.0, (X.size(0), 1))).float()
        sn = torch.tensor([mapping[x] for x in label.tolist()]).unsqueeze(1).float()
        sn = 0.5*torch.sigmoid((sn-mu)/(beta*gamma))
        lam = sn*lam
        lam = lam.to(X.get_device())
        idx = list(range(X.size(0)))
        random.shuffle(idx)
        X = (1. - lam)*X + lam*X[idx]
        return X

    def forward(self, input_s, input_t, input_t_raw, is_train=False, merge_idx=None, label=None):
        speech, mask_s = self.listener(input_s, is_train=is_train)
        ncon_word, mask_t = self.sw_reader(input_t_raw)

        con_word, attn = self.cross_attn(ncon_word, speech, mask_s.bool())
        out_tok = extract(con_word.permute(1,0,2), mask_t.long())
        out_cls_txt = self.classifier_txt(out_tok)

        #out_feat = merge(con_word.permute(1,0,2), merge_idx)
        out_merge, mask_new = merge_keep_seq(con_word.permute(1,0,2), merge_idx)
        out_merge = out_merge.permute(1,0,2)

        out_feat, _ = self.rnn_cls(out_merge)
        out_feat = extract(out_feat.permute(1,0,2), mask_new.long())

        if label is not None:
            out_feat = self.mfw(out_feat, label)
        out_cls = self.classifier(out_feat)
        return out_cls, out_cls_txt #speech_rep, bert_rep, attn

class CoModel(nn.Module):
    def __init__(self, config):
        super(CoModel, self).__init__()
        self.listener = Listener(config['input_dim'], config['pyr_layer'], config['device'], config['nlayer'], config['nhead'], dropout=config['dropout'])
        self.sw_reader = SubWReader(config['vocab_size'], config['device'])
        self.cross_attn = Attention(768, config['nhead'], dropout=config['dropout'])
        self.cls_layer = nn.Linear(768, config['nclasses'])
        self.teacher = Teacher()
        self.dropout = nn.Dropout(config['dropout'])
        self.device = config['device']

    def forward_full(self, input_s, input_t, input_t_raw, label=None, dist=None, is_train=False, mixup=False):
        speech, mask_s, label_mix = self.listener(input_s, label=label, dist=dist, is_train=is_train, mixup=mixup)
        ncon_word, mask_t = self.sw_reader(input_t_raw)

        con_word, attn = self.cross_attn(ncon_word, speech, mask_s.bool())
        oracle, mask_t2 = self.teacher(input_t)

        speech_rep = extract(con_word.permute(1,0,2), mask_t.long())
        bert_rep = extract(oracle.permute(1,0,2), mask_t.long())
    
        speech_cls = con_word[0]
        bert_cls = oracle[0]

        return speech_rep, bert_rep, speech_cls, bert_cls, label_mix

    def forward(self, input_s, label=None, dist=None, is_train=False, mixup=False):
        speech, mask_s, label_mix = self.listener(input_s, label=label, dist=dist, is_train=is_train, mixup=mixup)

        bsz = len(input_s)
        query = [torch.tensor([101]).long().to(self.device) for _ in range(bsz)]
        query, _ = self.sw_reader(query)
        
        speech_cls, _ = self.cross_attn(query, speech, mask_s.bool())
        return speech_cls.squeeze(0), label_mix

    def classify(self, input_s):
        return self.cls_layer(self.dropout(input_s))

    def classify_t(self, input_s):
        return self.cls_layer_t(self.dropout(input_s))
    
