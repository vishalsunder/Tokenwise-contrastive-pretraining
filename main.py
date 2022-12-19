from util import *
from models import *
from train import *
from data import *
from logging.handlers import RotatingFileHandler
from tokenizers import Tokenizer
import torch.optim as optim
import torch
import pdb
import logging
import copy
import argparse
import time
import random
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--pretrain', action='store_true',
                        help='')
    parser.add_argument('--mlm', action='store_true',
                        help='')
    parser.add_argument('--st', action='store_true',
                        help='')
    parser.add_argument('--multi-modal', action='store_true',
                        help='')
    parser.add_argument('--speech', action='store_true',
                        help='')
    parser.add_argument('--text', action='store_true',
                        help='')
    parser.add_argument('--slu-cotrain', action='store_true',
                        help='')
    parser.add_argument('--save-model', action='store_true',
                        help='')
    parser.add_argument('--discr', action='store_true',
                        help='')
    parser.add_argument('--specaug', action='store_true',
                        help='')
    parser.add_argument('--sni', action='store_true',
                        help='')
    parser.add_argument('--mixup', action='store_true',
                        help='')
    parser.add_argument('--supcon', action='store_true',
                        help='')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--slu-data', type=str, default='hvb',
                        help='')
    parser.add_argument('--enc-type', type=str, default='lstm',
                        help='')
    parser.add_argument('--normalizer', type=str, default='',
                        help='')
    parser.add_argument('--logging-file', type=str, default='',
                        help='log file')
    parser.add_argument('--train-path', type=str, default='',
                        help='training file')
    parser.add_argument('--train-path2', type=str, default='',
                        help='training file')
    parser.add_argument('--valid-path', type=str, default='',
                        help='validation file')
    parser.add_argument('--valid-path2', type=str, default='',
                        help='validation file')
    parser.add_argument('--test-path', type=str, default='',
                        help='testing file')
    parser.add_argument('--test-path2', type=str, default='',
                        help='testing file')
    parser.add_argument('--audio-path', type=str, default='',
                        help='where speech files are saved')
    parser.add_argument('--audio-path2', type=str, default='',
                        help='where speech files are saved')
    parser.add_argument('--save-path', type=str, default='',
                        help='')
    parser.add_argument('--dict-path', type=str, default='',
                        help='')
    parser.add_argument('--nspeech-feat', type=int, default=40,
                        help='')
    parser.add_argument('--pyr-layer', type=int, default=2,
                        help='')
    parser.add_argument('--nlayer', type=int, default=6,
                        help='')
    parser.add_argument('--nhead', type=int, default=12,
                        help='')
    parser.add_argument('--nsteps', type=int, default=100000,
                        help='')
    parser.add_argument('--steps-done', type=int, default=0,
                        help='')
    parser.add_argument('--log-after', type=int, default=100,
                        help='')
    parser.add_argument('--val-after', type=int, default=500,
                        help='')
    parser.add_argument('--save-after', type=int, default=100000,
                        help='')
    parser.add_argument('--sample-rate', type=int, default=8000,
                        help='')
    parser.add_argument('--nclasses', type=int, default=16,
                        help='')
    parser.add_argument('--patience', type=int, default=10,
                        help='')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='')
    parser.add_argument('--norm-epoch', type=int, default=3,
                        help='')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='')
    parser.add_argument('--cls-wt', type=float, default=0.5,
                        help='')

    args = parser.parse_args()
    device = torch.device("cpu")
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            device = torch.device("cuda")

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed) # ignored if not --cuda
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    rfh = RotatingFileHandler(args.logging_file, maxBytes=100000, backupCount=10, encoding="UTF-8")
    logger.addHandler(rfh)

    vocab_size = len(TOK)
    config = {'input_dim':1*args.nspeech_feat, 'pyr_layer':args.pyr_layer, 'device':device, 'nlayer':args.nlayer, 'nhead':args.nhead, 'dropout':args.dropout, 'vocab_size':vocab_size, 'nclasses':args.nclasses, 'enc_type':args.enc_type}

    print(f'Loading model.')
    if args.pretrain:
        model = PTModel(config)
    elif args.slu_cotrain:
        model = CoModel(config)
    else:
        if args.slu_data == 'kid':
            model = KidModel(config)
        else:
            model = SLUModel(config)
    if args.dict_path != '':
        load_dict(model, args.dict_path) 
    model = model.to(device)
    print(f'Done.')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)
    if args.discr:
        low_param, high_param = get_params(model)
        optimizer = optim.Adam([{'params':low_param, 'lr':0.00002}, {'params':high_param}], lr=0.0001, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)
    if args.pretrain:
        #data_train = RRACE(args, args.train_path, args.audio_path, n_mels=args.nspeech_feat, sample_rate=args.sample_rate, train=True)
        #data_valid = RRACE(args, args.valid_path, args.audio_path, n_mels=args.nspeech_feat, sample_rate=args.sample_rate)
        #data_test = RRACE(args, args.test_path, args.audio_path, n_mels=args.nspeech_feat, sample_rate=args.sample_rate)
        data_train = SLURP(args, args.train_path, args.audio_path, n_mels=args.nspeech_feat, sample_rate=args.sample_rate)
        data_valid = SLURP(args, args.valid_path, args.audio_path, n_mels=args.nspeech_feat, sample_rate=args.sample_rate)
        data_test = SLURP(args, args.test_path, args.audio_path, n_mels=args.nspeech_feat, sample_rate=args.sample_rate)
        #data_train = LibriSpeech(args, args.train_path, args.audio_path, n_mels=args.nspeech_feat, sample_rate=args.sample_rate)
        #data_valid = LibriSpeech(args, args.valid_path, args.audio_path, n_mels=args.nspeech_feat, sample_rate=args.sample_rate)
        #data_test = LibriSpeech(args, args.test_path, args.audio_path, n_mels=args.nspeech_feat, sample_rate=args.sample_rate)
        #data_train = LibFsh(args, args.train_path, args.train_path2, args.audio_path, n_mels=args.nspeech_feat, sample_rate=args.sample_rate)
        #data_valid = LibFsh(args, args.valid_path, args.valid_path2, args.audio_path2, frac=0.60, n_mels=args.nspeech_feat, sample_rate=args.sample_rate)
        #data_test = LibFsh(args, args.test_path, args.test_path2, args.audio_path2, frac=0.60, n_mels=args.nspeech_feat, sample_rate=args.sample_rate)
    elif args.slu_data == 'hvb':
        data_train = HVB(args, args.train_path, args.audio_path, n_mels=args.nspeech_feat, sample_rate=args.sample_rate)
        data_valid = HVB(args, args.valid_path, args.audio_path, n_mels=args.nspeech_feat, sample_rate=args.sample_rate)
        data_test = HVB(args, args.test_path, args.audio_path, n_mels=args.nspeech_feat, sample_rate=args.sample_rate)
    elif args.slu_data == 'snips':
        data_train = SNIPS(args, args.train_path, args.audio_path, n_mels=args.nspeech_feat, sample_rate=args.sample_rate)
        data_valid = SNIPS(args, args.valid_path, args.audio_path, n_mels=args.nspeech_feat, sample_rate=args.sample_rate)
        data_test = SNIPS(args, args.test_path, args.audio_path, n_mels=args.nspeech_feat, sample_rate=args.sample_rate)
    elif args.slu_data == 'slurp':
        data_train = SLURP(args, args.train_path, args.audio_path, n_mels=args.nspeech_feat, sample_rate=args.sample_rate)
        data_valid = SLURP(args, args.valid_path, args.audio_path, n_mels=args.nspeech_feat, sample_rate=args.sample_rate)
        data_test = SLURP(args, args.test_path, args.audio_path, n_mels=args.nspeech_feat, sample_rate=args.sample_rate)
    elif args.slu_data == 'fsc':
        data_train = FSC(args, args.train_path, args.audio_path, n_mels=args.nspeech_feat, sample_rate=args.sample_rate)
        data_valid = FSC(args, args.valid_path, args.audio_path, n_mels=args.nspeech_feat, sample_rate=args.sample_rate)
        data_test = FSC(args, args.test_path, args.audio_path, n_mels=args.nspeech_feat, sample_rate=args.sample_rate)
    elif args.slu_data == 'kid':
        data_train = RRACE(args, args.train_path, args.audio_path, n_mels=args.nspeech_feat, sample_rate=args.sample_rate, train=True)
        data_valid = RRACE(args, args.valid_path, args.audio_path, n_mels=args.nspeech_feat, sample_rate=args.sample_rate)
        data_test = RRACE(args, args.test_path, args.audio_path, n_mels=args.nspeech_feat, sample_rate=args.sample_rate)
    trainer = Trainer(args, data_train, device, optimizer, data_valid=data_valid, data_test=data_test)
    if args.pretrain:
        if args.mlm:
            trainer.pretrain_mlm(model, logger)
        elif args.st:
            trainer.pretrain_st(model, logger)
        else:
            trainer.pretrain(model, logger)
        #score_val = trainer.knn_moniter.run(model)
        #print(score_val)
        #trainer.get_attn_map('base', model, trainer.loader_map, os.path.join('attn_maps',args.logging_file[5:-4]))
    elif args.slu_cotrain:
        trainer.slu_cotrain(model, logger)
    else:
        if args.slu_data == 'kid':
            trainer.kid_speech(model, logger)
        elif args.multi_modal:
            trainer.slu_st(model, logger)
        else:
            trainer.slu_speech(model, logger)
