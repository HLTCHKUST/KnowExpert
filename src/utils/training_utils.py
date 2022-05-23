import os	
import sys	
import json	
import pickle	
import random	
import logging	
import time	
import argparse	
import numpy as np	
from tqdm import tqdm, trange	
from datetime import datetime

import torch	
from torch.utils.data import Dataset, DataLoader	
import transformers	
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM, get_linear_schedule_with_warmup	

from src.data_utils.moe_reader_st import get_data_from_batch as moe_st_batcher
from src.data_utils.moe_reader import get_data_from_batch as moe_batcher
from src.data_utils.dialog_reader_st import get_data_from_batch as dialog_st_batcher
from src.data_utils.dialog_reader import get_data_from_batch as dialog_batcher

logger = logging.getLogger(__name__)	


def save_adapter(args, tokenizer, model, optimizer, scheduler, global_step):
    if args.model_type == "seq2seq":
        save_seq2seq_adapter(args, tokenizer, model, optimizer, scheduler, global_step)
    elif args.model_type == "decoder_only":
        if args.dual_kadapter:
            save_decoder_only_dual_adapter(args, tokenizer, model, optimizer, scheduler, global_step)
        else:
            save_decoder_only_adapter(args, tokenizer, model, optimizer, scheduler, global_step)
    else:
        raise ValueError("Invalid model type for saving adapters!")

def save_decoder_only_adapter(args, tokenizer, model, optimizer, scheduler, global_step):	
    logger.info("Saving model...")	
    saving_path = f'{args.save_path}/models/{args.exp}/kadapter-{args.index}'	

    os.makedirs(saving_path, exist_ok=True)	
    # save args	
    if not os.path.exists(os.path.join(saving_path, "args")):	
        args_data = json.dumps(vars(args), indent=4)	
        with open(os.path.join(saving_path, "args"), "w") as f:	
            f.write(args_data)	

    checkpoint_prefix = "checkpoint"	
    # Save model checkpoint	
    output_dir = os.path.join(saving_path, "{}-{}".format(checkpoint_prefix, global_step))	
    os.makedirs(output_dir, exist_ok=True)	

    # Instead of saving the whole model, to save space, here we only save the adapters    
    # TODO double check how to save and load part of the model
    model_to_save = {}
    for l in range(len(model.transformer.h)):
        if args.task_adapter:
            model_to_save[f"topic_adapter{l}"] = model.transformer.h[l].topic_adapter.state_dict()
            torch.save(model_to_save, os.path.join(output_dir, f"task_adapter.pt"))
        elif args.kadapter:
            model_to_save[f"kadapter{l}"] = model.transformer.h[l].kadapter.state_dict()
            torch.save(model_to_save, os.path.join(output_dir, f"kadapter.pt"))
        else:
            raise ValueError(f"Invalid model with no task adapter and kadapter")

    # Here we don't save pre-trained tokenizer and training args to save space
    logger.info("Saving model checkpoint to %s", output_dir)

def save_decoder_only_dual_adapter(args, tokenizer, model, optimizer, scheduler, global_step):	
    logger.info("Saving model...")	
    saving_path = f'{args.save_path}/models/{args.exp}/kadapter-{args.index}'	

    os.makedirs(saving_path, exist_ok=True)	
    # save args	
    if not os.path.exists(os.path.join(saving_path, "args")):	
        args_data = json.dumps(vars(args), indent=4)	
        with open(os.path.join(saving_path, "args"), "w") as f:	
            f.write(args_data)	

    checkpoint_prefix = "checkpoint"	
    # Save model checkpoint	
    output_dir = os.path.join(saving_path, "{}-{}".format(checkpoint_prefix, global_step))	
    os.makedirs(output_dir, exist_ok=True)	

    # Instead of saving the whole model, to save space, here we only save the adapters    
    # TODO double check how to save and load part of the model
    model_to_save = {}
    for l in range(len(model.transformer.h)):
        if args.task_adapter:
            model_to_save[f"topic_adapter{l}"] = model.transformer.h[l].topic_adapter.state_dict()
            torch.save(model_to_save, os.path.join(output_dir, f"task_adapter.pt"))
        elif args.kadapter:
            model_to_save[f"pre_kadapter{l}"] = model.transformer.h[l].pre_kadapter.state_dict()
            model_to_save[f"post_kadapter{l}"] = model.transformer.h[l].post_kadapter.state_dict()
            torch.save(model_to_save, os.path.join(output_dir, f"kadapter.pt"))
        else:
            raise ValueError(f"Invalid model with no task adapter and kadapter")

    # Here we don't save pre-trained tokenizer and training args to save space
    logger.info("Saving model checkpoint to %s", output_dir)	


def save_seq2seq_adapter(args, tokenizer, model, optimizer, scheduler, global_step):	
    logger.info("Saving model...")	
    saving_path = f'{args.save_path}/models/{args.exp}/kadapter-{args.index}'	

    os.makedirs(saving_path, exist_ok=True)	
    # save args	
    if not os.path.exists(os.path.join(saving_path, "args")):	
        args_data = json.dumps(vars(args), indent=4)	
        with open(os.path.join(saving_path, "args"), "w") as f:	
            f.write(args_data)	

    checkpoint_prefix = "checkpoint"	
    # Save model checkpoint	
    output_dir = os.path.join(saving_path, "{}-{}".format(checkpoint_prefix, global_step))	
    os.makedirs(output_dir, exist_ok=True)	
    
    # TODO double check how to save and load part of the model
    model_to_save = {}
    for l in range(len(model.model.encoder.layers)):
        if args.task_adapter:
            model_to_save[f"topic_adapter_encoder{l}"] = model.model.encoder.layers[l].topic_adapter.state_dict()
            torch.save(model_to_save, os.path.join(output_dir, f"task_adapter.pt"))
        elif args.kadapter:
            model_to_save[f"kadapter_encoder{l}"] = model.model.encoder.layers[l].kadapter.state_dict()
            torch.save(model_to_save, os.path.join(output_dir, f"kadapter.pt"))
        else:
            raise ValueError(f"Invalid model with no task adapter and kadapter")
    
    for l in range(len(model.model.decoder.layers)):
        if args.task_adapter:
            model_to_save[f"topic_adapter_decoder{l}"] = model.model.decoder.layers[l].topic_adapter.state_dict()
            torch.save(model_to_save, os.path.join(output_dir, f"task_adapter.pt"))
        elif args.kadapter:
            model_to_save[f"kadapter_decoder{l}"] = model.model.decoder.layers[l].kadapter.state_dict()
            torch.save(model_to_save, os.path.join(output_dir, f"kadapter.pt"))
        else:
            raise ValueError(f"Invalid model with no task adapter and kadapter")

    logger.info("Saving model checkpoint to %s", output_dir)	


def save_model(args, tokenizer, model, optimizer, scheduler, global_step):	
    logger.info("Saving model...")	
    saving_path = f'{args.save_path}/models/{args.exp}/'	

    os.makedirs(saving_path, exist_ok=True)	
    # save args	
    if not os.path.exists(os.path.join(saving_path, "args")):	
        args_data = json.dumps(vars(args), indent=4)	
        with open(os.path.join(saving_path, "args"), "w") as f:	
            f.write(args_data)	

    checkpoint_prefix = "checkpoint"	
    # Save model checkpoint	
    output_dir = os.path.join(saving_path, "{}-{}".format(checkpoint_prefix, global_step))	
    os.makedirs(output_dir, exist_ok=True)	
    model_to_save = (	
        model.module if hasattr(model, "module") else model	
    )  # Take care of distributed/parallel training	
    model_to_save.save_pretrained(output_dir)	
    tokenizer.save_pretrained(output_dir)	

    torch.save(args, os.path.join(output_dir, "training_args.bin"))	
    logger.info("Saving model checkpoint to %s", output_dir)	

    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))	
    if scheduler:	
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))	
    logger.info("Saving optimizer and scheduler states to %s", output_dir)

def check_saving_path(path):
    if os.path.exists(path):
        return True
    else:
        try: 
            os.makedirs(path, exist_ok=True)
        except:
            logger.info("The saving path {path} cannot be used, please double check!")
            raise 

def mask_tokens(inputs, masks, tokenizer, args):	
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """	

    if tokenizer.mask_token is None:	
        raise ValueError(	
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."	
        )	

    labels = inputs.clone()	
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)	
    probability_matrix = torch.full(labels.shape, args.mlm_probability)	
    special_tokens_mask = [	
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()	
    ]	
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)	
    if masks is not None:	
        padding_mask = masks.eq(0)	
        probability_matrix.masked_fill_(padding_mask, value=0.0)	
    masked_indices = torch.bernoulli(probability_matrix).bool()	
    labels[~masked_indices] = -100  # We only compute loss on masked tokens	

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])	
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices	
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)	

    # 10% of the time, we replace masked input tokens with random word	
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced	
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)	
    inputs[indices_random] = random_words[indices_random]	

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged	
    return inputs, padding_mask, labels	