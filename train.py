from models import *
from util import *
from data import *
from knn import *
from tqdm import tqdm
from copy import deepcopy
from sklearn.metrics import f1_score, accuracy_score
from speechbrain.processing.features import InputNormalization
from speechbrain.lobes.augment import SpecAugment
import numpy as np
import copy
import pdb
import random
import math
import torch
import time
import torch.nn as nn
import torch.nn.functional as F

def euc_loss(r1, r2):
    r1 = F.normalize(r1, dim=1)
    r2 = F.normalize(r2, dim=1)
    return (2. - 2. * ((r1*r2).sum(dim=1))).mean()

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

def mixup_slu_fast(r_t, r_s, targets_t, targets_s, device, beta=2.0):
    M = r_t.size(0)
    N = r_s.size(0)
    S = r_t.size(1)
    C = targets_t.size(1)
    r_t = r_t.unsqueeze(1)
    r_s = r_s.unsqueeze(0)
    tgt_t = targets_t.unsqueeze(1)
    tgt_s = targets_s.unsqueeze(0)
    temp = torch.zeros(M,N,1)
    temp[:] = beta
    beta_list = temp.tolist()
    lam = np.random.beta(beta_list, beta_list)
    lam = torch.from_numpy(np.maximum(lam, 1. - lam)).float().to(device)
    r_mix = (lam) * r_t + (1-lam)*r_s
    tgt_mix = (lam) * tgt_t + (1-lam)*tgt_s
    return r_mix.view(-1, S), tgt_mix.view(-1, C)

class ContrastiveLoss(nn.Module):
    def __init__(self, device, temp=0.07):
        super(ContrastiveLoss, self).__init__()
        self.device = device
        self.temp = temp 

    def forward(self, r1, r2): # bsz, 768
        r1 = F.normalize(r1, dim=1)
        r2 = F.normalize(r2, dim=1)
        assert r1.size(0) == r2.size(0)
        tgt = torch.eye(r1.size(0)).to(self.device)

        align = torch.matmul(r1, r2.t()) / self.temp
        al_0 = torch.log_softmax(align, dim=0)
        al_1 = torch.log_softmax(align, dim=1)

        loss_0 = -1. * self.temp * (al_0 * tgt).sum(dim=0).mean()
        loss_1 = -1. * self.temp * (al_1 * tgt).sum(dim=1).mean()
        loss = 0.5 * (loss_0 + loss_1)
        return loss

    def forward_part(self, r1, r2, idx):
        r1 = r1[idx]
        r1 = F.normalize(r1, dim=1)
        r2 = F.normalize(r2, dim=1)

        tgt = torch.zeros(r1.size(0), r2.size(0))
        for i in range(r1.size(0)):
            tgt[i][idx[i]] = 1
        tgt = tgt.detach().to(self.device)
        tgt_0 = (tgt / tgt.sum(dim=0, keepdim=True)).detach()
        tgt_0[tgt_0 != tgt_0] = 0.
        tgt_1 = (tgt / tgt.sum(dim=1, keepdim=True)).detach()
        tgt_1[tgt_1 != tgt_1] = 0.

        align = torch.matmul(r1, r2.t()) / self.temp
        al_0 = torch.log_softmax(align, dim=0)
        al_1 = torch.log_softmax(align, dim=1)

        loss_0 = -1. * self.temp * (al_0 * tgt_0).sum(dim=0).mean()
        loss_1 = -1. * self.temp * (al_1 * tgt_1).sum(dim=1).mean()
        loss = 0.5 * (loss_0 + loss_1)
        return loss
        
class SupContrastiveLoss(nn.Module):
    def __init__(self, device, temp=0.07):
        super(SupContrastiveLoss, self).__init__()
        self.device = device
        self.temp = temp 

    def forward(self, r, label):
        MASK = torch.eye(label.size(0)).to(self.device)
        PEN = -10000000.

        r = F.normalize(r, dim=1)
        align_r = torch.log_softmax((torch.matmul(r, r.t()) / self.temp) + MASK*PEN, dim=1)

        label = F.normalize(label, dim=1)
        align_label = torch.matmul(label, label.t()) * (1. - MASK)
        align_label = align_label / align_label.sum(dim=1, keepdim=True)
        align_label[align_label != align_label] = 0.
        
        loss = -1. * self.temp * (align_r * align_label).sum(dim=1).mean()
        return loss

class Trainer(object):
    def __init__(self, args, data, device, optimizer, normalizer=None, data_valid=None, data_test=None):
        self.args = args
        self.data = data
        self.device = device
        self.aug = SpecAugment(time_warp=False, freq_mask_width=(0, 15), time_mask_width=(0, 70))
        self.norm = InputNormalization(update_until_epoch=args.norm_epoch)
        if args.normalizer != '':
            self.norm = load_pick(self.args.normalizer)
        self.optimizer = optimizer
        #self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.001, total_steps=args.nsteps, anneal_strategy='linear', pct_start=0.08)
        if args.pretrain:
            collator = CollatorPT(args)
        elif args.slu_cotrain:
            collator = CollatorSLU(args)
            self.cls_loss = nn.CrossEntropyLoss()#nn.KLDivLoss(reduction='batchmean')
            self.kld_loss = nn.KLDivLoss(reduction='batchmean')
            self.l1_loss = torch.nn.L1Loss()
        elif args.slu_data == 'kid':
            collator = CollatorKID(args)
            self.cls_loss = nn.CrossEntropyLoss()
        elif args.slu_data == 'hvb': 
            collator = CollatorSLU(args)
            self.cls_loss = nn.BCEWithLogitsLoss()
        else:
            collator = CollatorSLU(args)
            if args.mixup:
                self.cls_loss = nn.KLDivLoss(reduction='batchmean')
            else:
                self.cls_loss = nn.CrossEntropyLoss()
        self.con_loss = ContrastiveLoss(device)
        self.supcon_loss = SupContrastiveLoss(device)
        self.loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collator)
        self.loader_va = torch.utils.data.DataLoader(data_valid, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collator)
        self.loader_te = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collator)
        self.loader_map = torch.utils.data.DataLoader(data_valid, batch_size=1, shuffle=True, num_workers=8, collate_fn=collator)
        self.knn_moniter = kNN(args, self.loader_va, device, self.norm)

    def opt_step(self, model, loss):
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
        self.optimizer.step()

    def get_attn_map(self, steps, model, loader, path):
        model.eval()
        y_pred = []
        y_true = []
        for X, lens, lmax, tbatch, tbatch_raw in loader:
            lens_norm = [1.*(x/lmax) for x in lens]
            sbatch = self.norm(X, torch.tensor(lens_norm).float(), epoch=100)
            sbatch = list_batch(sbatch, lens, lmax)
            #sbatch = [add_delta(x, self.args.nspeech_feat) for x in sbatch]
            sbatch, tbatch, tbatch_raw = load2gpu(sbatch, self.device), load2gpu(tbatch, self.device), load2gpu(tbatch_raw,self.device)
            with torch.no_grad():
                _, _, attn = model(sbatch, tbatch, tbatch_raw)
            pdb.set_trace()
            attn = attn[0].detach().cpu().numpy()
            np.save(f'{path}_last.npy', attn)
            break

    def evaluate(self, model, loader, test=False):
        model.eval()
        y_pred = []
        y_true = []
        sents = []
        for X, lens, lmax, tbatch, tbatch_raw, label in tqdm(loader):
            lens_norm = [1.*(x/lmax) for x in lens]
            sbatch = self.norm(X, torch.tensor(lens_norm).float(), epoch=100)
            sbatch = list_batch(sbatch, lens, lmax)
            sbatch, label = load2gpu(sbatch, self.device), load2gpu(label, self.device)
            with torch.no_grad():
                r_s = model(sbatch)
                pred = model.classify(r_s)
            if self.args.slu_data == 'hvb':
                y_pred.extend(torch.round(torch.sigmoid(pred)).long().cpu().tolist())
                y_true.extend(label.long().cpu().tolist())
            else:
                y_pred.extend(torch.max(pred, dim=1)[1].cpu().tolist())
                y_true.extend(label.cpu().tolist())
        if self.args.slu_data == 'hvb':
            score = f1_score(y_true, y_pred, average='macro')
        else:
            score = accuracy_score(y_true, y_pred)
        return score

    def evaluate_kid(self, model, loader):
        model.eval()
        y_pred = []
        y_true = []
        sents = []
        for X, lens, lmax, tbatch, tbatch_raw, merge_idx, label, _ in tqdm(loader):
            lens_norm = [1.*(x/lmax) for x in lens]
            sbatch = self.norm(X, torch.tensor(lens_norm).float(), epoch=100)
            sbatch = list_batch(sbatch, lens, lmax)
            sbatch, tbatch, tbatch_raw, label = load2gpu(sbatch, self.device), load2gpu(tbatch, self.device), load2gpu(tbatch_raw,self.device), load2gpu(label, self.device)
            with torch.no_grad():
                pred, _ = model(sbatch, tbatch, tbatch_raw, merge_idx=merge_idx)
            y_pred.extend(torch.max(pred, dim=1)[1].cpu().tolist())
            y_true.extend(label.cpu().tolist())
        score = f1_score(y_true, y_pred, average='macro')
        return score

    def pretrain_st(self, model, logger):
        steps = self.args.steps_done
        best_score = None
        best_model = None
        epoch = 0
        loss_mlm_list = []
        loss_aln_list = []
        loss_list = []
        scores_val = []
        rpat = self.args.patience
        while rpat > 0:
            epoch = epoch+1
            print(f'Running epoch {epoch}.')
            for X, lens, lmax, tbatch, tbatch_raw in tqdm(self.loader):
                model.train()
                steps += 1
                lens_norm = [1.*(x/lmax) for x in lens]
                sbatch = self.norm(X, torch.tensor(lens_norm).float(), epoch=epoch-1)
                if self.args.specaug:
                    sbatch = self.aug(sbatch)# TODO adding specaugment
                sbatch = list_batch(sbatch, lens, lmax)
                #sbatch = [add_delta(x, self.args.nspeech_feat) for x in sbatch]
                sbatch, tbatch, tbatch_raw = load2gpu(sbatch, self.device), load2gpu(tbatch, self.device), load2gpu(tbatch_raw,self.device)
                speech_rep, text_rep, bert_rep, mlm_opt_s, mlm_tgt_s, mlm_id_s, mlm_opt_t, mlm_tgt_t, mlm_id_t = model.forward_st(sbatch, tbatch, tbatch_raw, is_train=self.args.sni)
                loss_aln = 0.75*self.con_loss(speech_rep, bert_rep.detach()) + 0.25*self.con_loss(text_rep, bert_rep.detach())
                loss_mlm = 0.75*self.con_loss.forward_part(mlm_opt_s, mlm_tgt_s.detach(), mlm_id_s) + 0.25*self.con_loss.forward_part(mlm_opt_t, mlm_tgt_t.detach(), mlm_id_t)
                loss = loss_aln + loss_mlm
                self.opt_step(model, loss)
                #self.scheduler.step()
                loss_aln_list.append(loss_aln.item())
                loss_mlm_list.append(loss_mlm.item())
                loss_list.append(loss.item())
                if steps % self.args.save_after == 0:
                    save(model, f'{self.args.save_path}_steps_{steps}.pt')
                if steps % self.args.val_after == 0:
                    print(f'Running validation.') 
                    score_val = self.knn_moniter.run(model)
                    #score_val = self.knn_moniter.run_mlm(model)
                    scores_val.append(score_val)
                    np.save(f'knn_scores/{self.args.logging_file[5:-4]}.npy', np.array(scores_val))
                    if best_score is None or best_score < score_val:
                        best_model = copy.deepcopy(model)
                        best_score = score_val
                        rpat = self.args.patience
                    else:
                        rpat -= 1
                    log = f'| steps = {steps} | dev_acc = {score_val} |'
                    logger.info(log)
                    #self.get_attn_map(steps, model, self.loader_map, os.path.join('attn_maps',self.args.logging_file[5:-4]))
                if steps % self.args.log_after == 0:
                    log = f'| steps = {steps} | loss_aln = {np.mean(loss_aln_list)} | loss_mlm = {np.mean(loss_mlm_list)} | patience = {rpat} |'
                    logger.info(log)
                    loss_mlm_list = []
                    loss_aln_list = []
                    loss_list = []
        if self.args.save_model:
            save(best_model, self.args.save_path+'_best.pt')
            #save(model, self.args.save_path+'_last.pt')
    

    def pretrain_mlm(self, model, logger):
        steps = self.args.steps_done
        best_score = None
        best_model = None
        epoch = 0
        loss_mlm_list = []
        loss_aln_list = []
        loss_list = []
        scores_val = []
        rpat = self.args.patience
        while rpat > 0:
            epoch = epoch+1
            print(f'Running epoch {epoch}.')
            for X, lens, lmax, tbatch, tbatch_raw in tqdm(self.loader):
                model.train()
                steps += 1
                lens_norm = [1.*(x/lmax) for x in lens]
                sbatch = self.norm(X, torch.tensor(lens_norm).float(), epoch=epoch-1)
                if self.args.specaug:
                    sbatch = self.aug(sbatch)# TODO adding specaugment
                sbatch = list_batch(sbatch, lens, lmax)
                #sbatch = [add_delta(x, self.args.nspeech_feat) for x in sbatch]
                sbatch, tbatch, tbatch_raw = load2gpu(sbatch, self.device), load2gpu(tbatch, self.device), load2gpu(tbatch_raw,self.device)
                r_s, r_b, _, mlm_opt, mlm_tgt, mlm_idx = model.forward_mlm(sbatch, tbatch, tbatch_raw, is_train=self.args.sni)
                loss_aln = self.con_loss(r_s, r_b.detach())
                loss_mlm = self.con_loss.forward_part(mlm_opt, mlm_tgt.detach(), mlm_idx)
                loss = loss_aln + loss_mlm
                self.opt_step(model, loss)
                #self.scheduler.step()
                loss_aln_list.append(loss_aln.item())
                loss_mlm_list.append(loss_mlm.item())
                loss_list.append(loss.item())
                if steps % self.args.save_after == 0:
                    save(model, f'{self.args.save_path}_steps_{steps}.pt')
                if steps % self.args.val_after == 0:
                    print(f'Running validation.') 
                    score_val = self.knn_moniter.run(model)
                    #score_val = self.knn_moniter.run_mlm(model)
                    scores_val.append(score_val)
                    np.save(f'knn_scores/{self.args.logging_file[5:-4]}.npy', np.array(scores_val))
                    if best_score is None or best_score < score_val:
                        best_model = copy.deepcopy(model)
                        best_score = score_val
                        rpat = self.args.patience
                    else:
                        rpat -= 1
                    log = f'| steps = {steps} | dev_acc = {score_val} |'
                    logger.info(log)
                    #self.get_attn_map(steps, model, self.loader_map, os.path.join('attn_maps',self.args.logging_file[5:-4]))
                if steps % self.args.log_after == 0:
                    log = f'| steps = {steps} | loss_aln = {np.mean(loss_aln_list)} | loss_mlm = {np.mean(loss_mlm_list)} | patience = {rpat} |'
                    logger.info(log)
                    loss_mlm_list = []
                    loss_aln_list = []
                    loss_list = []
        if self.args.save_model:
            save(best_model, self.args.save_path+'_best.pt')
            #save(model, self.args.save_path+'_last.pt')

    def pretrain(self, model, logger):
        steps = self.args.steps_done
        best_score = None
        best_model = None
        epoch = 0
        loss_list = []
        scores_val = []
        rpat = self.args.patience
        while rpat > 0:
            epoch = epoch+1
            print(f'Running epoch {epoch}.')
            for X, lens, lmax, tbatch, tbatch_raw in tqdm(self.loader):
                model.train()
                steps += 1
                lens_norm = [1.*(x/lmax) for x in lens]
                sbatch = self.norm(X, torch.tensor(lens_norm).float(), epoch=epoch-1)
                if self.args.specaug:
                    sbatch = self.aug(sbatch)# TODO adding specaugment
                sbatch = list_batch(sbatch, lens, lmax)
                #sbatch = [add_delta(x, self.args.nspeech_feat) for x in sbatch]
                sbatch, tbatch, tbatch_raw = load2gpu(sbatch, self.device), load2gpu(tbatch, self.device), load2gpu(tbatch_raw,self.device)
                r_s, r_b, _ = model(sbatch, tbatch, tbatch_raw, is_train=self.args.sni)
                loss = self.con_loss(r_s, r_b.detach())
                self.opt_step(model, loss)
                #self.scheduler.step()
                loss_list.append(loss.item())
                if steps % self.args.save_after == 0:
                    save(model, f'{self.args.save_path}_steps_{steps}.pt')
                if steps % self.args.val_after == 0:
                    print(f'Running validation.') 
                    score_val = self.knn_moniter.run(model)
                    scores_val.append(score_val)
                    np.save(f'knn_scores/{self.args.logging_file[5:-4]}.npy', np.array(scores_val))
                    if best_score is None or best_score < score_val:
                        best_model = copy.deepcopy(model)
                        best_score = score_val
                        rpat = self.args.patience
                    else:
                        rpat -= 1
                    log = f'| steps = {steps} | dev_acc = {score_val} |'
                    logger.info(log)
                    #self.get_attn_map(steps, model, self.loader_map, os.path.join('attn_maps',self.args.logging_file[5:-4]))
                if steps % self.args.log_after == 0:
                    log = f'| steps = {steps} | loss = {np.mean(loss_list)} | patience = {rpat} |'
                    logger.info(log)
                    loss_list = []
        if self.args.save_model:
            save(best_model, self.args.save_path+'_best.pt')
            #save(model, self.args.save_path+'_last.pt')

    def slu_speech(self, model, logger):
        best_score = None
        best_model = None
        epoch = 0
        loss_list = []
        rpat = self.args.patience
        while rpat > 0:
            epoch = epoch+1
            print(f'Running epoch {epoch}.')
            for X, lens, lmax, tbatch, tbatch_raw, label in tqdm(self.loader):
                model.train()
                lens_norm = [1.*(x/lmax) for x in lens]
                sbatch = self.norm(X, torch.tensor(lens_norm).float(), epoch=epoch-1)
                if self.args.specaug:
                    sbatch = self.aug(sbatch)# TODO adding specaugment
                sbatch = list_batch(sbatch, lens, lmax)
                sbatch, label = load2gpu(sbatch, self.device), load2gpu(label, self.device)
                r_s = model(sbatch, is_train=self.args.sni)
                pred = model.classify(r_s)
                loss = self.cls_loss(pred, label)
                self.opt_step(model, loss)
                loss_list.append(loss.item())
            print(f'Running validation.') 
            score_val = self.evaluate(model, self.loader_va)
            if best_score is None or best_score < score_val:
                best_model = copy.deepcopy(model)
                best_score = score_val
                rpat = self.args.patience
            else:
                rpat -= 1
            log = f'| epoch = {epoch} | loss = {np.mean(loss_list)} | dev_score = {score_val} |'
            logger.info(log)
            print(log)
            loss_list = []
        print(f'Running test.') 
        score_test = self.evaluate(best_model, self.loader_te, test=True)
        print(f'| test score = {score_test} |')
        logger.info(f'| test score = {score_test} |')

    def slu_st(self, model, logger):
        best_score = None
        best_model = None
        epoch = 0
        loss_list = []
        rpat = self.args.patience
        while rpat > 0:
            epoch = epoch+1
            print(f'Running epoch {epoch}.')
            for X, lens, lmax, tbatch, tbatch_raw, label in tqdm(self.loader):
                model.train()
                lens_norm = [1.*(x/lmax) for x in lens]
                sbatch = self.norm(X, torch.tensor(lens_norm).float(), epoch=epoch-1)
                if self.args.specaug:
                    sbatch = self.aug(sbatch)# TODO adding specaugment
                sbatch = list_batch(sbatch, lens, lmax)
                sbatch, label = load2gpu(sbatch, self.device), load2gpu(label, self.device)
                sbatch, tbatch, label = load2gpu(sbatch, self.device), load2gpu(tbatch, self.device), load2gpu(label,self.device)
                r_s, r_t = model.forward_st(sbatch, tbatch, is_train=self.args.sni, speech=self.args.speech, text=self.args.text)
                if r_s is None:
                    pred = model.classify(r_t)
                elif r_t is None:
                    pred = model.classify(r_s)
                else:
                    pred_s = model.classify(r_s)
                    pred_t = model.classify(r_t)
                    pred = torch.cat([pred_s, pred_t], dim=0)
                    label = torch.cat([label, label], dim=0)
                loss = self.cls_loss(pred, label)
                self.opt_step(model, loss)
                loss_list.append(loss.item())
            print(f'Running validation.') 
            score_val = self.evaluate(model, self.loader_va)
            if best_score is None or best_score < score_val:
                best_model = copy.deepcopy(model)
                best_score = score_val
                rpat = self.args.patience
            else:
                rpat -= 1
            log = f'| epoch = {epoch} | loss = {np.mean(loss_list)} | dev_score = {score_val} |'
            logger.info(log)
            print(log)
            loss_list = []
        print(f'Running test.') 
        score_test = self.evaluate(best_model, self.loader_te, test=True)
        print(f'| test score = {score_test} |')
        logger.info(f'| test score = {score_test} |')

    def kid_speech(self, model, logger):
        best_score = None
        best_model = None
        epoch = 0
        loss_list = []
        rpat = self.args.patience
        while rpat > 0:
            epoch = epoch+1
            print(f'Running epoch {epoch}.')
            for X, lens, lmax, tbatch, tbatch_raw, merge_idx, label, label_txt in tqdm(self.loader):
                model.train()
                lens_norm = [1.*(x/lmax) for x in lens]
                sbatch = self.norm(X, torch.tensor(lens_norm).float(), epoch=epoch-1)
                #sbatch = self.aug(sbatch)# TODO adding specaugment
                sbatch = list_batch(sbatch, lens, lmax)
                sbatch, tbatch, tbatch_raw, label, label_txt = load2gpu(sbatch, self.device), load2gpu(tbatch, self.device), load2gpu(tbatch_raw, self.device), load2gpu(label, self.device), load2gpu(label_txt, self.device)
                pred, pred_txt = model(sbatch, tbatch, tbatch_raw, merge_idx=merge_idx, label=label.cpu())
                loss = self.cls_loss(pred, label) + self.cls_loss(pred_txt, label_txt)
                self.opt_step(model, loss)
                loss_list.append(loss.item())
            print(f'Running validation.') 
            score_val = self.evaluate_kid(model, self.loader_va)
            if best_score is None or best_score < score_val:
                best_model = copy.deepcopy(model)
                best_score = score_val
                rpat = self.args.patience
            else:
                rpat -= 1
            log = f'| epoch = {epoch} | loss = {np.mean(loss_list)} | dev_score = {score_val} |'
            logger.info(log)
            print(log)
            loss_list = []
        print(f'Running test.') 
        score_test = self.evaluate_kid(best_model, self.loader_te)
        print(f'| test score = {score_test} |')
        logger.info(f'| test score = {score_test} |')

    def slu_cotrain(self, model, logger):
        best_score = None
        best_model = None
        epoch = 0
        loss_cont_list = []
        loss_pred_list = []
        rpat = self.args.patience
        while rpat > 0:
            epoch = epoch+1
            print(f'Running epoch {epoch}.')
            for X, lens, lmax, tbatch, tbatch_raw, _, label, label_oh in tqdm(self.loader):
                model.train()
                lens_norm = [1.*(x/lmax) for x in lens]
                sbatch = self.norm(X, torch.tensor(lens_norm).float(), epoch=epoch-1)
                if self.args.specaug:
                    sbatch = self.aug(sbatch)# TODO adding specaugment
                sbatch = list_batch(sbatch, lens, lmax)
                #sbatch = [add_delta(x, self.args.nspeech_feat) for x in sbatch]
                sbatch, tbatch, tbatch_raw, label, label_oh = load2gpu(sbatch, self.device), load2gpu(tbatch, self.device), load2gpu(tbatch_raw,self.device), load2gpu(label, self.device), load2gpu(label_oh, self.device)
                s_rep, b_rep, s_cls, b_cls, _ = model.forward_full(sbatch, tbatch, tbatch_raw, is_train=self.args.sni)
                #pred_t = model.classify_t(b_cls)#torch.cat([s_cls, b_cls], dim=0))
                pred_s = model.classify(s_cls)#torch.cat([s_cls, b_cls], dim=0))
                pred_t = model.classify(b_cls)
                loss_pred = self.cls_loss(pred_s, label) + self.cls_loss(pred_t, label) #torch.cat([label, label], dim=0))
                #soft_pred = torch.log_softmax(pred_s/1.0, dim=1)
                #soft_target = torch.softmax(pred_t/1.0, dim=1)
                loss_cont = self.l1_loss(pred_s, pred_t.detach())#self.con_loss(s_cls, b_cls.detach())#self.kld_loss(soft_pred, soft_target.detach())
                loss = loss_cont + loss_pred
                self.opt_step(model, loss)
                loss_cont_list.append(loss_cont.item())
                loss_pred_list.append(loss_pred.item())
            print(f'Running validation.') 
            score_val = self.evaluate(model, self.loader_va)
            if best_score is None or best_score < score_val:
                best_model = copy.deepcopy(model)
                best_score = score_val
                rpat = self.args.patience
            else:
                rpat -= 1
            log = f'| epoch = {epoch} | loss_pred = {np.mean(loss_pred_list)} | loss_cont = {np.mean(loss_cont_list)} | dev_score = {score_val} |'
            logger.info(log)
            print(log)
            loss_cont_list = []
            loss_pred_list = []
        print(f'Running test.') 
        score_test = self.evaluate(best_model, self.loader_te)
        print(f'| test score = {score_test} |')
        logger.info(f'| test score = {score_test} |')
        if self.args.save_model:
            save(best_model, self.args.save_path+'_best.pt')
