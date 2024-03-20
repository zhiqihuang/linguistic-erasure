#!/usr/bin/env python
# coding: utf-8
import torch
import os
from models.leace import get_score
from models.leace_layer import mDPRLayerScrubber, BertModelLeace
from transformers import AutoTokenizer
from arguments import get_train_parser
from concept_erasure import LeaceFitter
from tqdm import tqdm
from util.dataset import get_train_tevatrono, read_multi_parallel, read_multi_collections, MultilingualParallelDataset
from util.util import MixedPrecisionManager, doc_tokenizer, query_tokenizer, set_seed, setup_wandb
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def get_domain_datasets(args):
    languages = sorted(args.langs)
    assert 'english' not in languages
    # _, multi_parallel = read_multi_parallel(args.parallel_collection, languages, args.nproc_per_node, args.num_train)
    multi_parallel = read_multi_collections(args.parallel_collection, languages)
    
    train_data = []
    valid_data = []
    train_size = args.num_train
    valid_size = 5000

    for lang in languages:
        train_data.extend(multi_parallel[lang][:train_size])
        valid_data.extend(multi_parallel[lang][train_size:valid_size+train_size])

    parallel_train = MultilingualParallelDataset(train_data)
    parallel_valid = MultilingualParallelDataset(valid_data)
    print("parallel_train", len(parallel_train), "parallel_valid", len(parallel_valid))
    train_loader = torch.utils.data.DataLoader(
            parallel_train,
            drop_last=True,
            batch_size=args.batch_size*125,
            pin_memory=True, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
            parallel_valid,
            drop_last=False,
            batch_size=args.batch_size*125,
            pin_memory=True, shuffle=False)
    return train_loader, valid_loader

def get_ir_datasets(args):
    dataset_train = get_train_tevatrono(args)
    train_loader = torch.utils.data.DataLoader(
            dataset_train,
            drop_last=True,
            batch_size=args.batch_size,
            shuffle=True, pin_memory=True)
    return train_loader

def update_scrubber(model, tokenizer, args, dc_loader_iter, dc_train_loader, dc_valid_loader, steps):
    model.eval()
    amp = MixedPrecisionManager(args.fp16)
    dev_res = {}
    run_dev = (steps and steps % args.logging_steps == 0)
    train_tokenized_batches = []
    for _ in range(20):
        try:
            batch = next(dc_loader_iter)
        except StopIteration:
            dc_loader_iter = iter(dc_train_loader)
            batch = next(dc_loader_iter)
        sents, labels = batch
        sents = sum(list(map(list, zip(*sents))), [])
        labels = sum(list(map(list, zip(*labels))), [])
        lingual_ids, lingual_mask = doc_tokenizer(sents, args, tokenizer)
        lingual_labels = torch.stack(labels, dim=0).long().to(args.device)
        train_tokenized_batches.append([lingual_ids, lingual_mask, lingual_labels])

    leace_fitter = []
    with torch.no_grad():
        with amp.context():
            for layer in tqdm(range(1, 14), desc='leace fitter layer'):
                fitter = LeaceFitter(768, 1, dtype=torch.float64, device=args.device)
                for item in train_tokenized_batches:
                    lingual_ids, lingual_mask, lingual_labels = item
                    features = model.feature(input_ids=lingual_ids, attention_mask=lingual_mask, encode_depth=layer, leace_fitter=leace_fitter)
                    fitter.update(features, lingual_labels)
                leace_fitter.append(fitter)
    
    if run_dev:
        X_train, y_train = [], []
        with torch.no_grad():
            for item in train_tokenized_batches:
                lingual_ids, lingual_mask, lingual_labels = item
                features = model.feature(input_ids=lingual_ids, attention_mask=lingual_mask, encode_depth=0, leace_fitter=leace_fitter).detach().cpu()
                X_train.append(features)
                y_train.append(lingual_labels)
            X_train = torch.cat(X_train).cpu().numpy()
            y_train = torch.cat(y_train).cpu().numpy().astype(int)
            
            X_dev = []
            y_dev = []
            for batch in dc_valid_loader:
                sents, labels = batch
                sents = sum(list(map(list, zip(*sents))), [])
                labels = sum(list(map(list, zip(*labels))), [])
                lingual_ids, lingual_mask = doc_tokenizer(sents, args, tokenizer)
                lingual_labels = torch.stack(labels, dim=0).long().to(args.device)
                features = model.feature(input_ids=lingual_ids, attention_mask=lingual_mask, encode_depth=0, leace_fitter=leace_fitter).detach().cpu()
                X_dev.append(features)
                y_dev.append(lingual_labels.cpu())
            X_dev = torch.cat(X_dev).numpy()
            y_dev = torch.cat(y_dev).numpy().astype(int)
        loss_val, train_score, dev_score = get_score(X_train, y_train, X_dev, y_dev)
        dev_res['loss_val'] = loss_val
        dev_res['train_score'] = train_score
        dev_res['dev_score'] = dev_score
    model.train()
    return leace_fitter, dev_res

def train(model, tokenizer, args, ir_train_loader, dc_train_loader, dc_valid_loader, wandb_run=None):
    
    wandb_run.watch(model)
    model.train()
    steps = 0
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    amp = MixedPrecisionManager(args.fp16)
    train_rank_loss = torch.tensor(0.0).to(args.device)
    dc_loader_iter = iter(dc_train_loader)
    
    leace_fitter, dev_res = update_scrubber(model, tokenizer, args, dc_loader_iter, dc_train_loader, dc_valid_loader, steps)
    
    for epoch in range(args.num_train_epochs):
        for item in tqdm(ir_train_loader, desc=f'train at epoch {epoch}'):
            queries, pos_docs, neg_docs = item
            queries, pos_docs, neg_docs = list(map(list, zip(*queries))), list(map(list, zip(*pos_docs))) ,list(map(list, zip(*neg_docs)))
            queries = sum(queries, [])
            pos_docs = sum(pos_docs, [])
            neg_docs = sum(neg_docs, [])
            q_ids, q_mask = query_tokenizer(queries, args, tokenizer)
            d_ids, d_mask = doc_tokenizer(pos_docs + neg_docs, args, tokenizer)

            steps += 1
            if steps % args.gradient_accumulation_steps == 0:
                with amp.context():
                    rank_loss = model(q_ids, q_mask, d_ids, d_mask, encode_depth=0, leace_fitter=leace_fitter)
                amp.backward(rank_loss)
                amp.step(model, optim)

                # update scrubber
                leace_fitter, dev_res = update_scrubber(model, tokenizer, args, dc_loader_iter, dc_train_loader, dc_valid_loader, steps)
            else:
                with amp.context():
                    rank_loss = model(q_ids, q_mask, d_ids, d_mask, encode_depth=0, leace_fitter=leace_fitter)
                amp.backward(rank_loss)
            
            train_rank_loss += rank_loss.item()
            
            if steps % args.logging_steps == 0 and steps % args.gradient_accumulation_steps == 0:
                train_rank_loss = train_rank_loss.item() / (args.logging_steps)
                wandb_run.log({
                    'train_rank_loss': train_rank_loss,
                    'recover_loss': dev_res['loss_val'],
                    'dev_recover_accuracy': dev_res['dev_score'] * 100,
                    'train_recover_accuracy': dev_res['train_score'] * 100,
                })
                train_rank_loss = torch.tensor(0.0).to(args.device)

        # save checkpoint after training finish:
        state = {'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optim.state_dict()}
        torch.save(state, os.path.join(args.output_dir, f'checkpoint_at_{epoch+1}.pth'))
        for idx, fitter in enumerate(leace_fitter):
            torch.save(fitter.__dict__, os.path.join(args.output_dir, f'layer_{idx+1}_fitter_at_{epoch+1}.pth'))

def main(args):
    set_seed(args.seed)
    args.device = torch.cuda.current_device()
    wandb_obj = setup_wandb(args)
    os.makedirs(args.output_dir, exist_ok=True)

    args.num_langs = len(args.langs)
    
    # init model
    if args.use_pooler:
        base_encoder = BertModelLeace.from_pretrained(args.base_model_name, add_pooling_layer=True)
    else:
        base_encoder = BertModelLeace.from_pretrained(args.base_model_name, add_pooling_layer=False)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    except:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, from_slow=True)

    assert tokenizer.is_fast
    
    model = mDPRLayerScrubber(base_encoder, args)
    model.to(args.device)

    # load checkpoint
    if args.checkpoint:
        model.load(args.checkpoint)
    
    # train
    ir_train_loader = get_ir_datasets(args)
    dc_train_loader, dc_valid_loader = get_domain_datasets(args)

    train(model, tokenizer, args, ir_train_loader, dc_train_loader, dc_valid_loader, wandb_obj)

if __name__ == '__main__':
    parser = get_train_parser()
    args = parser.parse_args()
    main(args)