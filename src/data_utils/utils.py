# This file is for all utility functions

import os
import pickle
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def build_input_for_seq2seq_model(labels, label_masks):
    """
    Build decoder inputs and labels
    """
    seqlen = labels.size(1)
    decoder_inputs = labels.narrow(1, 0, seqlen - 1).clone()
    decoder_masks = label_masks.narrow(1, 0, seqlen - 1).clone()
    decoder_labels = labels.narrow(1, 1, seqlen-1).clone()
    decoder_label_masks = label_masks.narrow(1, 1, seqlen - 1).clone()
    return decoder_inputs, decoder_masks, decoder_labels, decoder_label_masks


def save(toBeSaved, filename, mode='wb'):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    file = open(filename, mode)
    pickle.dump(toBeSaved, file, protocol=4) # protocol 4 allows large size object, it's the default since python 3.8
    file.close()

def load(filename, mode='rb'):
    file = open(filename, mode)
    loaded = pickle.load(file)
    file.close()
    return loaded

def pad_sents(sents, pad_token=0, max_len=512):
    sents_padded = []
    lens = get_lens(sents)
    max_len = min(max(lens), max_len)
    sents_padded = []
    new_len = []
    for i, l in enumerate(lens):
        if l > max_len:
            l = max_len
        new_len.append(l)
        sents_padded.append(sents[i][:l] + [pad_token] * (max_len - l))
    return sents_padded, new_len

def sort_sents(sents, reverse=True):
    sents.sort(key=(lambda s: len(s)), reverse=reverse)
    return sents

def get_mask(sents, unmask_idx=1, mask_idx=0, max_len=512):
    lens = get_lens(sents)
    max_len = min(max(lens), max_len)
    mask = []
    for l in lens:
        if l > max_len:
            l = max_len
        mask.append([unmask_idx] * l + [mask_idx] * (max_len - l))
    return mask

def get_lens(sents):
    return [len(sent) for sent in sents]

def get_max_len(sents):
    max_len = max([len(sent) for sent in sents])
    return max_len

def truncate_sents(self, sents, length):
    sents = [sent[:length] for sent in sents]
    return sents

def pad_list_of_sents(list_of_sents, pad_token=0, max_len=512):
    list_of_sents_padded = []
    lens = []
    max_lens = []
    for sents in list_of_sents:
        sents_len = get_lens(sents)
        lens.append(sents_len)
        max_lens.append(max(sents_len))
    max_len = min(max(max_lens), max_len)
    new_lens = []
    for sents, sents_len in zip(list_of_sents, lens):
        sents_padded = []
        new_len = []
        for i, l in enumerate(sents_len):
            if l > max_len:
                l = max_len
            new_len.append(l)
            sents_padded.append(sents[i][:l] + [pad_token] * (max_len - l))
        list_of_sents_padded.append(sents_padded)
        new_lens.append(new_len)
    return list_of_sents_padded, new_lens

def get_list_of_mask(list_of_sents, unmask_idx=1, mask_idx=0, max_len=512):
    list_of_mask = []
    lens = []
    max_lens = []
    for sents in list_of_sents:
        sents_len = get_lens(sents)
        lens.append(sents_len)
        max_lens.append(max(sents_len))
    max_len = min(max(max_lens), max_len)
    for sents_len in lens:
        mask = []
        for l in sents_len:
            if l > max_len:
                l = max_len
            mask.append([unmask_idx] * l + [mask_idx] * (max_len - l))
        list_of_mask.append(mask)
    return list_of_mask

def get_input_from_batch(batch, device):
    src_tokens = batch["context"].to(device)
    # src_tokens_mask = batch["context_mask"]

    know_tokens = batch["knowledge_sentences"].to(device) # DELETED ([batch, max_num_kn, seqlen] -> [batch, seqlen, max_num_kn])
    ck_mask = batch["knowledge_sentences_mask"].to(device)

    cs_ids = batch["chosen_sentence_id"].to(device)

    return src_tokens, know_tokens, ck_mask, cs_ids

def get_output_from_batch(batch, device):
    tgt_tokens = batch["response"].to(device)
    # tgt_tokens_mask = batch["response_mask"]
    # cs_ids = batch["chosen_sentence_id"].to(device)
    return tgt_tokens

    
def convert_one_hot(d, dim):
    # 1d -> 2d
    length = d.shape[0]
    dd = np.zeros((length, dim))
    for i in range(dim):
        mask = (d==i)
        dd[mask, i] = 1
    mask = (d==-1)
    dd[mask] = 1/dim
    return dd