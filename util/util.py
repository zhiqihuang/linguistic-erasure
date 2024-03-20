#!/usr/bin/env python
# coding: utf-8
import subprocess
import torch
import numpy as np
import torch
import wandb
import random
import os
from contextlib import contextmanager

def check_cuda():
    is_available = torch.cuda.is_available()
    if is_available:
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"cuda is available, GPU name {device_name}")
    return is_available

class NullContextManager(object):
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource
    def __enter__(self):
        return self.dummy_resource
    def __exit__(self, *args):
        pass

class MixedPrecisionManager():
    def __init__(self, activated):

        self.activated = activated

        if self.activated:
            self.scaler = torch.cuda.amp.GradScaler()

    def context(self):
        return torch.cuda.amp.autocast() if self.activated else NullContextManager()

    def backward(self, loss):
        if self.activated:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, model, optimizer):
        if self.activated:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            optimizer.zero_grad()

def setup_wandb(args):
    wandb.login(key = '20a758ad1186a647e85dc9cbba110dc0972d29c3')
    if hasattr( args, 'is_master'):
        if args.is_master:
            run = wandb.init(project='m2mdpr', entity='zhiqi', config=args, settings=wandb.Settings(code_dir="."))
            run.name = args.job_name
        else:
            run = None
    else:
        run = wandb.init(project='m2mdpr', entity='zhiqi', config=args, settings=wandb.Settings(code_dir="."))
        run.name = args.job_name
    return run

def set_seed(seed):
    '''
    some cudnn methods can be random even after fixing the seed
    unless you tell it to be deterministic
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def trec_eval(qrelf, runf, metric, trec_eval_f):
    output = subprocess.check_output([trec_eval_f, '-m', metric, qrelf, runf]).decode().rstrip()
    output = output.replace('\t', ' ').split('\n')
    assert len(output) == 1
    return float(output[0].split()[2])

def test_trec_eval(qrelf, runf, metrics, trec_eval_f):
    output = subprocess.check_output([trec_eval_f, qrelf, runf]).decode().rstrip()
    output = output.split('\n')
    eval_out = []
    for line in output:
        linetoks = line.split('\t')
        if linetoks[0].strip() in metrics:
            eval_out.append(line)
    return eval_out

def test_trec_eval(qrelf, runf, metrics, trec_eval_f):
    measure = []
    for m in metrics.split(','):
        measure.append('-m')
        measure.append(m.strip())
    output = subprocess.check_output([trec_eval_f] + measure + [qrelf, runf]).decode().rstrip()
    output = output.split('\n')
    eval_out = []
    for line in output:
        eval_out.append(line)
    return eval_out

def query_tokenizer(qtxt, args, tokenizer, padding='longest'):
    toks = tokenizer(
        list(qtxt),
        padding=padding,
        return_tensors='pt',
        max_length = args.query_maxlen,
        truncation=True
    )
    ids, mask = toks['input_ids'], toks['attention_mask']
    return ids.to(args.device), mask.to(args.device)

def doc_tokenizer(dtxt, args, tokenizer, padding='longest'):
    toks = tokenizer(
        list(dtxt),
        padding=padding,
        return_tensors='pt',
        max_length = args.doc_maxlen,
        truncation=True
    )
    ids, mask = toks['input_ids'], toks['attention_mask']
    return ids.to(args.device), mask.to(args.device)

def get_module(model):
    return model.module if hasattr(model, "module") else model
