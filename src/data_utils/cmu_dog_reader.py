import os
import json
import pickle
import random
import pprint
import logging
import numpy as np
from tqdm import tqdm
from operator import itemgetter
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator

from parlai.core.dict import DictionaryAgent
from parlai.core.worlds import create_task

import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AutoTokenizer

from src.data_utils.utils import pad_sents, get_mask, pad_list_of_sents, get_list_of_mask
from src.data_utils.data_reader import getDataLoader
from src.data_utils.dialog_reader import DialogReader

from pprint import PrettyPrinter

pp = PrettyPrinter(indent=4, width=200)

class CMUDoGDialogReader(DialogReader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._task = "cmu_dog"

    def _load_and_preprocess_all(self, mode: str):
        """
        Load the preprocessed data from {data_folder}/datasets-CMU_DoG
        """
        episodes = []
        src_file = os.path.join(self._data_dir, f"datasets-CMU_DoG/src-{mode}-tokenized.txt")
        tgt_file = os.path.join(self._data_dir, f"datasets-CMU_DoG/tgt-{mode}-tokenized.txt")
        kn_file  = os.path.join(self._data_dir, f"datasets-CMU_DoG/knl-{mode}-tokenized.txt")
        
        with open(src_file, "r") as f:
            src_data = f.readlines()
        with open(tgt_file, "r") as f:
            tgt_data = f.readlines()
        with open(kn_file, "r") as f:
            kn_data  = f.readlines()

        for c, r, k in tqdm(zip(src_data, tgt_data, kn_data), total=len(src_data), desc="Preprocess dataset", ncols=100):
            episode = {'context': [],
                       'response': [],
                       'knowledge': []}

            context = c.replace("&apos;","'").replace("&quot;",'"').strip().split("&lt; SEP &gt;")
            assert len(context) == 3
            knowledge = k.replace("&apos;","'").replace("&quot;",'"').strip().split("&lt; SEP &gt;")
            episode["context"] = [text.strip() for text in context]
            episode["response"] = r.replace("&apos;","'").replace("&quot;",'"').strip()
            episode["knowledge"] = [text.strip() for text in knowledge]
            episodes.append(episode)
        return episodes
    
    def read(self, mode: str, history_in_context: bool, kn_in_context: bool):
        def _gen(episodes, sos_token, eos_token, train_kld):
            """
            Convert example into samples for training and testing
            1. truncate the knowledge
            2. split the episode into training samples
            """
            samples = {
                "context": [],             # list
                "response": [],            # list
                "token_type": [],          # list
                "title": [],               # list
                "topic": [],               # list
                "chosen_sentence": [],     # list
                "sample_id": [],           # list
                "episode_id": [],          # list
                "knowledge": [],           # list of list
            }

            if train_kld:
                samples["kn"] = []                  # list
                samples["kn_token_type"] = []       # list

            for _id, episode in enumerate(tqdm(episodes, desc="Generate samples", ncols=100)):
                contexts = episode["context"]                          # list
                response = episode["response"]                        # list
                knowledges = episode["knowledge"]

                # get history
                history_head = self._tokenizer.encode(sos_token)   # list
                history_kn_head = self._tokenizer.encode(sos_token)
                token_type_ids_head = [1] * len(history_head)
                token_type_ids_kn_head = [0] * len(history_kn_head)

                current_kn = ""
                if kn_in_context:
                    if self._kn_mode == "retrieval":
                        current_kn = []
                        inds = self._kn_rank_list[_id][:self._top_kn_num]
                        for ind in inds:
                            current_kn.append(knowledge[ind])
                        current_kn = f" {eos_token} ".join(current_kn)
                    else:
                        raise ValueError("The knowledge mode is not supported.")

                kn = self._tokenizer.encode(current_kn + " " + eos_token)
                history_kn_head += kn
                token_type_ids_kn_head += [0] * (len(kn) - 1) + [1]

                history = []
                token_type_ids = []

                if history_in_context:
                    for num in range(len(contexts)):
                        tempc = self._tokenizer.encode(contexts[num].strip() + " " + eos_token)
                        history += tempc
                        if num % 2 == 0:
                            token_type_ids += [1] * (len(tempc) - 1) + [0]
                        else:
                            token_type_ids += [0] * (len(tempc) - 1) + [1]
                    if kn_in_context:
                        context = history_kn_head + history
                        token_type_ids = token_type_ids_kn_head + token_type_ids
                    else:
                        context = history_head + history
                        token_type_ids = token_type_ids_head + token_type_ids
                else:
                    context =  self._tokenizer.encode(sos_token + " " + contexts[-1].strip() + " " + eos_token)
                    token_type_ids = [1] * len(context)

                if self._model_type == "seq2seq":
                    samples["context"].append(context)
                    samples["response"].append(self._tokenizer.encode(sos_token + " " + response.strip() + " " + eos_token))
                elif self._model_type == "decoder_only":
                    response = self._tokenizer.encode(response.strip() + " " + eos_token)
                    token_type_ids = token_type_ids[:-1] + [0] * (len(response) + 1)
                    response_mask = [1] * len(context) + [0] * len(response)
                    context += response
                    samples["context"].append(context)
                    samples["response"].append(response_mask)
                else:
                    raise ValueError(f"Model type '{self._model_type}' is invalid.")

                assert len(token_type_ids) == len(samples["context"][-1])
                samples["token_type"].append(token_type_ids)
                samples["title"].append(self._tokenizer.encode("no title"))
                samples["topic"].append(self._tokenizer.encode("no topic"))
                samples["chosen_sentence"].append(self._tokenizer.encode("no_sentence_provided"))
                samples["sample_id"].append(_id)
                samples["episode_id"].append(_id)
                samples["knowledge"].append(knowledges)
                # if self._debug:
                #     break # for debug
            return samples

        def _uniform(samples, history_in_context):
            """
            1. pad the sents in the same sample to the maximum length
            2. get the length of the inputs
            3. get the mask of the inputs
            """
            uniformed_samples = {}
            for key in samples:
                if "id" in key or "knowledge" in key:
                    uniformed_samples[key] = samples[key]
                    continue

                # Get sample data, check the type
                sample = samples[key]
                if key == "checked_sentence":
                    mask = get_mask(sample, max_len=self._max_kn_length)
                    padded_sample, _ = pad_sents(sample, pad_token=0, max_len=self._max_kn_length)
                elif key == "context" or key == "token_type":
                    mask = get_mask(sample, max_len=self._max_context_length)
                    padded_sample, _ = pad_sents(sample, pad_token=0, max_len=self._max_context_length)
                elif "kn" in key:
                    mask = get_mask(sample, max_len=1024)
                    padded_sample, _ = pad_sents(sample, pad_token=0, max_len=1024)
                else:
                    mask = get_mask(sample, max_len=self._max_length)
                    padded_sample, _ = pad_sents(sample, pad_token=0, max_len=self._max_length)
                uniformed_samples[key] = padded_sample
                uniformed_samples[f"{key}_mask"] = mask
            return uniformed_samples

        # read the datasets and do preprocessing (tokenize, set up KN)
        episodes = self._load_and_preprocess_all(mode)
        # formulate the samples
        sos_token = self._tokenizer.cls_token if self._tokenizer.cls_token is not None else self._tokenizer.bos_token
        eos_token = self._tokenizer.sep_token if self._tokenizer.sep_token is not None else self._tokenizer.eos_token

        samples = _gen(episodes, sos_token, eos_token, self._train_kld)
        uniformed_samples = _uniform(samples, history_in_context)
        return uniformed_samples

def get_cmu_dog_dataloader(args, tokenizer, train=True):
    if train:
        train_reader = CMUDoGDialogReader(
            tokenizer,
            mode="train",
            max_length = args.max_length,
            max_context_length = args.max_context_length,
            max_kn_length = args.max_kn_length,
            data_dir = args.data_dir,
            history_in_context = args.history_in_context,
            model_type = args.model_type,
            debug = args.debug if hasattr(args, 'debug') else False,
            inference = args.inference if hasattr(args, 'inference') else False,
        )
        train_loader = getDataLoader(train_reader, args.bsz, test=False)
    else:
        train_loader = None

    valid_reader = CMUDoGDialogReader(
        tokenizer,
        mode="valid",
        max_length = args.max_length,
        max_context_length = args.max_context_length,
        max_kn_length = args.max_kn_length,
        data_dir = args.data_dir,
        history_in_context = args.history_in_context,
        model_type = args.model_type,
        debug = args.debug if hasattr(args, 'debug') else False,
        inference = args.inference if hasattr(args, 'inference') else False,
    )
    valid_loader = getDataLoader(valid_reader, args.eval_bsz, test=True)

    test_reader = CMUDoGDialogReader(
        tokenizer,
        mode="test",
        max_length = args.max_length,
        max_context_length = args.max_context_length,
        max_kn_length = args.max_kn_length,
        data_dir = args.data_dir,
        history_in_context = args.history_in_context,
        model_type = args.model_type,
        debug = args.debug if hasattr(args, 'debug') else False,
        inference = args.inference if hasattr(args, 'inference') else False,
    )
    test_loader = getDataLoader(test_reader, args.eval_bsz, test=True)

    dataloaders = {
        "train": train_loader,
        "valid": valid_loader,
        "test": test_loader,
    }
    return dataloaders

def load_cmu_episodes(data_dir, mode):
    """
    Load the preprocessed data from {data_folder}/datasets-CMU_DoG
    """
    episodes = []
    src_file = os.path.join(data_dir, f"datasets-CMU_DoG/src-{mode}-tokenized.txt")
    tgt_file = os.path.join(data_dir, f"datasets-CMU_DoG/tgt-{mode}-tokenized.txt")
    kn_file  = os.path.join(data_dir, f"datasets-CMU_DoG/knl-{mode}-tokenized.txt")
      
    with open(src_file, "r") as f:
        src_data = f.readlines()
    with open(tgt_file, "r") as f:
        tgt_data = f.readlines()
    with open(kn_file, "r") as f:
        kn_data  = f.readlines()

    for c, r, k in tqdm(zip(src_data, tgt_data, kn_data), total=len(src_data), desc="Preprocess dataset", ncols=100):
        episode = {'text': [],
                   'response': [],
                    'knowledge': []}

        context = c.replace("&apos;","'").replace("&quot;",'"').strip().split("&lt; SEP &gt;")
        assert len(context) == 3
        knowledge = k.replace("&apos;","'").replace("&quot;",'"').strip().split("&lt; SEP &gt;")
        episode["text"] = [text.strip() for text in context]
        episode["response"] = r.replace("&apos;","'").replace("&quot;",'"').strip()
        episode["knowledge"] = [text.strip() for text in knowledge]
        episodes.append(episode)
    return episodes   
    

if __name__ == "__main__":
    args = {
        "max_length" : 128,  
        "max_context_length" : 128,   # 256
        "max_kn_length" : 128,
        "data_dir" : "./data",
        "model_type" : "gpt2",
        "bsz" : 3,
        "history_in_context" : True,
    }
    tokenizer = AutoTokenizer.from_pretrained(args["model_type"])
    train_reader = CMUDoGDialogReader(
        tokenizer,
        mode="valid",
        max_length = args["max_length"],
        max_kn_length = args["max_kn_length"],
        data_dir = args["data_dir"],
        history_in_context = args["history_in_context"]
    )