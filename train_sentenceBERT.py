import os
import argparse
import inspect
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_scheduler
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device

from src.data_utils.data_reader import load_wow_episodes

INF = 1e8


class TextDataset(Dataset):
    def __init__(self, split):
        self.episodes = load_wow_episodes('./data', split, history_in_context=True, max_episode_length=1)
        self.history = []
        self.hisres = []
        for episode in self.episodes:
            self.history.append(' '.join(episode['context']))
            episode['context'].append(episode['response'])
            tmp = ' '.join(episode['context'])
            self.hisres.append(tmp)
    def __len__(self):
        return len(self.history)
    def __getitem__(self, index):
        return self.history[index], self.hisres[index]

def main(args):
    # random seed
    torch.manual_seed(777)

    # set device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    bert_ref = SentenceTransformer("sentence-transformers/stsb-roberta-base-v2")
    bert = SentenceTransformer("sentence-transformers/stsb-roberta-base-v2")

    bert.to(device)
    bert_ref.to(device)

    dataloaders = {}
    if args.do_train:
        train_dataset = TextDataset(split='train')
        dataloaders["train"] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_dataset = TextDataset(split='valid')
        dataloaders["valid"] = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
        valid_unseen_dataset = TextDataset(split='valid_unseen')
        dataloaders["valid_unseen"] = DataLoader(valid_unseen_dataset, batch_size=args.batch_size, shuffle=False)
    if args.do_eval:
        test_dataset = TextDataset(split='test')
        dataloaders["test"] = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        test_unseen_dataset = TextDataset(split='test_unseen')
        dataloaders["test_unseen"] = DataLoader(test_unseen_dataset, batch_size=args.batch_size, shuffle=False)

    if args.do_train:
        # optimizer = torch.optim.Adam(bert.parameters(), lr=args.lr, weight_decay=args.wd)
        optimizer = AdamW(bert.parameters(), lr=args.lr, weight_decay=args.wd)
        num_training_steps = args.epoch * len(dataloaders["train"])
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
        loss_fn = torch.nn.MSELoss() # torch.nn.KLDivLoss('batchmean')
        best_seen_acc, best_unseen_acc = INF, INF
        patience, best_epoch = 0, -1
        for e in range(args.epoch):
            # training
            bert.train()
            bert_ref.eval()
            loss, acc = 0, 0
            for iteration, batch in enumerate(tqdm(dataloaders["train"])):
                optimizer.zero_grad()
                with torch.set_grad_enabled(False):
                    gold = bert_ref.encode(batch[1], show_progress_bar=False, batch_size=args.batch_size, convert_to_tensor=True)
                with torch.set_grad_enabled(True):
                    features = bert.tokenize(batch[0])
                    features = batch_to_device(features, device)
                    out_features = bert.forward(features)
                    pred = out_features['sentence_embedding']

                    _loss = loss_fn(input=pred, target=gold)
                    loss += _loss.item()
                    _acc = torch.dist(pred, gold, 2) / pred.shape[0]
                    acc += _acc.item()
                    _loss.backward()
                    optimizer.step()
                    lr_scheduler.step()


            loss /= (iteration + 1)
            acc /= (iteration + 1)
            print(f'Epoch {e}: loss = {loss:.6f} acc = {acc:.4f}')
            
            # evaluation
            bert.eval()
            bert_ref.eval()
            seen_acc, unseen_acc = 0, 0
            for iteration, batch in enumerate(tqdm(dataloaders["valid"])):
                with torch.set_grad_enabled(False):
                    gold = bert_ref.encode(batch[1], show_progress_bar=False, batch_size=args.batch_size, convert_to_tensor=True)
                    pred = bert.encode(batch[0], show_progress_bar=False, batch_size=args.batch_size, convert_to_tensor=True)
                    _acc = torch.dist(pred, gold, 2) / pred.shape[0]
                    seen_acc += _acc.item()
            seen_acc /= (iteration + 1)
            for iteration, batch in enumerate(tqdm(dataloaders["valid_unseen"])):
                with torch.set_grad_enabled(False):
                    gold = bert_ref.encode(batch[1], show_progress_bar=False, batch_size=args.batch_size, convert_to_tensor=True)
                    pred = bert.encode(batch[0], show_progress_bar=False, batch_size=args.batch_size, convert_to_tensor=True)
                    _acc = torch.dist(pred, gold, 2) / pred.shape[0]
                    unseen_acc += _acc.item()

            unseen_acc /= (iteration + 1)
            print(f'Epoch {e}: seen acc = {seen_acc:.4f} unseen acc = {unseen_acc:.4f}')
            if (best_seen_acc + best_unseen_acc) > (seen_acc + unseen_acc):
                best_seen_acc = seen_acc
                best_unseen_acc = unseen_acc
                best_epoch = e
                patience = 0
                bert.save(args.output_dir)
            else:
                patience += 1
            
            if patience == args.patience:
                print('out of patience!')
                print(f'Best Epoch {best_epoch}: seen acc = {best_seen_acc:.4f} unseen acc = {best_unseen_acc:.4f}')
                break
    
    if args.do_eval:
        # testing
        bert = SentenceTransformer(args.output_dir)
        bert.eval()
        bert_ref.eval()
        seen_acc, unseen_acc = 0, 0
        for iteration, batch in enumerate(tqdm(dataloaders["test"])):
            with torch.set_grad_enabled(False):
                gold = bert_ref.encode(batch[1], show_progress_bar=False, batch_size=args.batch_size, convert_to_tensor=True)
                pred = bert.encode(batch[0], show_progress_bar=False, batch_size=args.batch_size, convert_to_tensor=True)
                _acc = torch.dist(pred, gold, 2) / pred.shape[0]
                seen_acc += _acc.item()
        seen_acc /= (iteration + 1)
        for iteration, batch in enumerate(tqdm(dataloaders["test_unseen"])):
            with torch.set_grad_enabled(False):
                gold = bert_ref.encode(batch[1], show_progress_bar=False, batch_size=args.batch_size, convert_to_tensor=True)
                pred = bert.encode(batch[0], show_progress_bar=False, batch_size=args.batch_size, convert_to_tensor=True)
                _acc = torch.dist(pred, gold, 2) / pred.shape[0]
                unseen_acc += _acc.item()
        unseen_acc /= (iteration + 1)
        print(f'Test: seen acc = {seen_acc:.4f} unseen acc = {unseen_acc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SentenceTransformer with KLDiv')
    parser.add_argument('-cu', '--cuda', help='CUDA', type=str, required=False, default='1')
    # Training hyper-parameters
    parser.add_argument('-bs', '--batch_size', help='Batch size', type=int, required=False, default=8)
    parser.add_argument('--lr', help='Learning rate', type=float, required=False, default=1e-3)
    parser.add_argument('--wd', help='Weight decay', type=float, required=False, default=0)
    parser.add_argument('-ep', '--epoch',help='Epoch', type=int, required=False, default=100)
    parser.add_argument('-pa', '--patience', help='Patience to stop training', type=int, required=False, default=5)
    parser.add_argument('--output_dir', type=str, default='save/models/sentence_bert')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    
    args = parser.parse_args()
    main(args)