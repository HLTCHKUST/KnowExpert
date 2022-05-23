import os
import json
import logging
import time
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
import inspect
import shutil

import torch
import transformers

from src.data_utils.utils import build_input_for_seq2seq_model

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

class Trainer():
    def __init__(self, tokenizer, model, dataloader, criterion, optimizer, scheduler, device, args, get_data_from_batch_fn):
        self.tokenizer = tokenizer
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.args = args

        self.model_input_names = list(inspect.getargspec(self.model.forward))[0][1:]
        self.get_data_from_batch = get_data_from_batch_fn

        # Evaluation results
        self.results = {
            "train_loss": 1000,
            "valid_loss": 1000,
            "valid_unseen_loss": 1000,
            "best_val_seen_loss": 1000,  
            "best_val_loss": 1000, 
            "kn_loss": 1000,
            "kn_valid_loss": 1000,
            "train_ppl": 1e5,
            "valid_ppl": 1e5,
            "valid_unseen_ppl": 1e5,
            "kn_ppl": 1e5,
            "kn_valid_ppl": 1e5,
        }
        self.ckpt_list = []

        # Early stop
        self.best_val_seen_loss = 1000
        self.best_val_loss = 1000
        self.patience = args.patience
        self.patience_iter = 0


    def Train(self, eval_splits=["valid"]):
        tb_writer = SummaryWriter(log_dir=f'runs/{self.args.exp}')
        for epoch in range(self.args.epoch):
            logger.info(f'============= Epoch {epoch} / {self.args.epoch} =============')
            self.train_one_epoch(epoch)

            if self.scheduler is not None:
                tb_writer.add_scalar("lr", self.scheduler.get_last_lr()[0], epoch)
            tb_writer.add_scalar("train_loss", self.results["train_loss"], epoch)
            tb_writer.add_scalar("train_ppl", self.results["train_ppl"], epoch)

            for split in eval_splits:
                self.eval_one_epoch(epoch, split)

                tb_writer.add_scalar(f"{split}_loss", self.results[f"{split}_loss"], epoch)
                tb_writer.add_scalar(f"{split}_ppl", self.results[f"{split}_ppl"], epoch)

            if self.patience_iter >= self.patience:
                logger.info("Out of patience! Early stop!")
                break
        tb_writer.close()
    
    def test(self, prefix, do_valid, do_test):
        if prefix == 'seen' or prefix is None:
            prefix = ''

        if do_valid:
            self.eval_one_epoch(self.args.checkpoint, 'valid_'+prefix if len(prefix)>0 else 'valid')
        if do_test:
            self.eval_one_epoch(self.args.checkpoint, 'test_'+prefix if len(prefix)>0 else 'test')
    

    def prepare_inputs(self, batch):
        input_ids, masks, kn_sent, kn_mask, topic, topic_mask, labels, label_masks, response_masks, \
            label_starts, label_idxs, token_type_ids = self.get_data_from_batch(batch, model_type=self.args.model_type)

        input_ids = input_ids.to(self.device)
        masks = masks.to(self.device)
        labels = labels.to(self.device)
        label_masks = label_masks.to(self.device)
        label_starts = label_starts.to(self.device)
        label_idxs = label_idxs.to(self.device)                
        
        inputs = {
            "input_ids": input_ids,
            "attention_mask": masks,
        }

        if "experts" in self.model_input_names:
            topic_mask = topic_mask.to(self.device)
            inputs.update({"experts": topic_mask})
        
        if self.args.model_type == "seq2seq":
            decoder_inputs, decoder_masks, labels, label_masks = build_input_for_seq2seq_model(labels, label_masks)
            decoder_inputs = decoder_inputs.to(self.device)
            decoder_masks = decoder_masks.to(self.device)
            labels = labels.to(self.device)
            label_masks = label_masks.to(self.device)

            seq2seq_inputs = {
                "decoder_input_ids": decoder_inputs,
                "decoder_attention_mask": decoder_masks,
                "use_cache": False,
            }
            inputs.update(seq2seq_inputs)
        elif self.args.model_type == "decoder_only":
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)
                inputs.update({"token_type_ids": token_type_ids})
        
        labels[~label_masks.bool()] = -100  # We only compute loss on masked tokens
        if self.args.model_type == "decoder_only":
            response_masks = response_masks.to(self.device)
            labels[response_masks.bool()] = -100


        return inputs, labels
    

    def train_one_epoch(self, epoch):
        self.model.train()
            
        running_loss = 0.0
        iters = 0
        since = time.time()
        loader = self.dataloader["train"]
        for step, batch in tqdm(enumerate(loader), desc=f'Training', total=len(loader), ncols=100):
            inputs, labels = self.prepare_inputs(batch)

            iters += 1
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):

                outputs = self.model(**inputs)
                logits = outputs[0]

                # compute loss
                loss = self.criterion(logits.contiguous().view(-1, logits.size(-1)), labels.contiguous().view(-1))
                running_loss += loss.item()

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                loss.backward()
                
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()

        # compute ppl
        tr_loss = running_loss / iters
        ppl = np.exp(tr_loss)

        logger.info(f"Epoch: train loss = {tr_loss:.4f}, train ppl = {ppl:.2f}")
        self.results["train_loss"] = tr_loss
        self.results["train_ppl"] = ppl

        time_elapsed = time.time() - since
        logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    

    def eval_one_epoch(self, epoch, split):
        self.model.eval()
        running_loss = 0.0
        iters = 0
        since = time.time()
        loader = self.dataloader[split]
        for batch in tqdm(loader, desc=f'Evaluating', total=len(loader), ncols=100):
            inputs, labels = self.prepare_inputs(batch)

            iters += 1
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(False):

                outputs = self.model(**inputs)
                logits = outputs[0]

                # compute loss
                loss = self.criterion(logits.contiguous().view(-1, logits.size(-1)), labels.contiguous().view(-1))
                running_loss += loss.item()

        # compute ppl
        val_loss = running_loss / iters
        val_ppl = np.exp(val_loss)

        logger.info(f"Epoch {epoch}: {split} loss = {val_loss:.4f}, {split} ppl = {val_ppl:.2f}")
        self.results[f"{split}_loss"] = val_loss
        self.results[f"{split}_ppl"] = val_ppl

        if self.best_val_seen_loss > val_loss and split == "valid":
            self.best_val_seen_loss = val_loss
            # save model
            self.save_model("best_seen")
            # self.save_model(epoch)
            self.patience_iter = 0
        elif split == "valid":
            self.patience_iter += 1
        
        if split == "valid_unseen":
            val_sum_loss = val_loss + self.results["valid_loss"]
            if self.best_val_loss > val_sum_loss:
                self.best_val_loss= val_sum_loss
                # save model
                self.save_model("best")

        time_elapsed = time.time() - since
        logger.info('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    

    def save_model(self, epoch):
        logger.info("Saving model...")
        saving_path = f'{self.args.save_path}/models/{self.args.exp}/'

        os.makedirs(saving_path, exist_ok=True)
        # save args
        # if not os.path.exists(os.path.join(saving_path, "args")): # resave args everytime
        args_data = json.dumps(vars(self.args), indent=4)
        with open(os.path.join(saving_path, "args"), "w") as f:
            f.write(args_data)

        checkpoint_prefix = "model"
        # Save model checkpoint
        output_dir = os.path.join(saving_path, "{}-{}".format(checkpoint_prefix, epoch))
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", output_dir)


        if epoch != "best" and epoch != "best_seen":
            self.ckpt_list.append(output_dir)
        
        self.check_saved_checkpoints()
    
    def load_best_model(self, ckpt="best"):
        best_model_path = f'{self.args.save_path}/models/{self.args.exp}/model-{ckpt}'
        logger.info("Loading best model from %s", best_model_path)
        self.model = self.model.from_pretrained(best_model_path)

    
    def check_saved_checkpoints(self):
        """
        For saving space, we just keep the last PATIENCE+1 checkpoints
        """
        while len(self.ckpt_list) > 3:
            del_path = self.ckpt_list.pop(0)
            if os.path.exists(del_path):
                logger.info(f"Delete checkpoint {del_path} because of the limited number of the saved checkpoints.")
                shutil.rmtree(del_path)
    
