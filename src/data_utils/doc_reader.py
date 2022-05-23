import os
import json
import math
import pickle
import random
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator
from nltk import sent_tokenize

import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AutoTokenizer, AutoModel

from src.data_utils.utils import pad_sents, get_mask, pad_list_of_sents, get_list_of_mask
from src.data_utils.masking import split_span_masking

import matplotlib.pyplot as plt

class DocReader(Dataset):
    def __init__(self,
                 data,
                 tokenizer,
                 dataset: str = "wow",
                 mode: str = "full",
                 max_length: int = 512,
                 data_dir: str = "./data",
                 model_type: str = "gpt2",
                 **kwargs
                 ):
        self._mode = mode   # 
        self._max_length = max_length
        self._data_dir = data_dir
        self._model_type = model_type
        self._tokenizer = tokenizer

        self._task = dataset
        assert self._task == "wow" or self._task == "cmu_dog"

        # read the datasets and do preprocessing
        pad_token_id = self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None else 0
        if data is None:
            data = self.load_doc()
        preproc_data = self.preproc_doc(data)
        self.mask = get_mask(preproc_data, max_len=self._max_length)
        self.data, _ = pad_sents(preproc_data, pad_token=pad_token_id, max_len=self._max_length)
        self.label = self.data.copy()
        self.label_mask = self.mask.copy()

        assert self.data is not None

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long), torch.tensor(self.mask[idx], dtype=torch.long), torch.tensor(self.label[idx], dtype=torch.long), torch.tensor(self.label_mask[idx], dtype=torch.long)

    def __len__(self):
        return len(self.data)
    
    @property
    def vocab(self):
        return self._tokenizer.vocab
    
    def load_doc(self):
        if self._task == "wow":
            data = self._load_wow_doc()
        elif self._task == "cmu_dog":
            data = self._load_cmu_doc()
        else:
            raise NotImplementedError
        return data
    
    def preproc_doc(self, raw_data):
        preproc_data = []

        for text in tqdm(raw_data,desc="Tokenize",total=len(raw_data)):
            tokenized_text = self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(text.lower()))   # TODO test lower case
            iter_num = len(tokenized_text) // self._max_length
            for i in range(0, iter_num):  # Truncate in block of max length
                preproc_data.append(self._tokenizer.build_inputs_with_special_tokens(tokenized_text[i*self._max_length : (i + 1) * self._max_length]))
            if len(tokenized_text) - iter_num * self._max_length > 0:
                n = iter_num
                preproc_data.append(self._tokenizer.build_inputs_with_special_tokens(tokenized_text[n * self._max_length:]))
        return preproc_data
    
    def _load_wow_doc(self):
        raw_data = []
        topics = []

        if self._mode == "full":
            with open(os.path.join(self._data_dir, "data.json"), 'r') as f:
                data = json.load(f)
        elif self._mode == "seen":
            with open(os.path.join(self._data_dir, "train.json"), 'r') as f:
                train = json.load(f)
            with open(os.path.join(self._data_dir, "valid_random_split.json"), 'r') as f:
                valid = json.load(f)
            with open(os.path.join(self._data_dir, "test_random_split.json"), 'r') as f:
                test = json.load(f)
            data = train + valid + test
        elif self._mode == "unseen":
            with open(os.path.join(self._data_dir, "valid_topic_split.json"), 'r') as f:
                valid = json.load(f)
            with open(os.path.join(self._data_dir, "test_topic_split.json"), 'r') as f:
                test = json.load(f)
            data = valid + test
        else:
            raise ValueError("Invalid document type!")
        for item in tqdm(data,desc="Read",total=len(data)):
            for i in range(len(item["dialog"])):
                dial = item["dialog"][i]
                if "wizard" in dial["speaker"].lower():
                    for j in range(len(dial["retrieved_passages"])):
                        for key in dial["retrieved_passages"][j].keys():
                            if key.replace("amp;", "").replace("&quot;",'\"') in dial["retrieved_topics"]:
                                # print(key, dial["retrieved_topics"])
                                # print(dial["checked_sentence"])
                                # input()
                                # if " ".join(dial["retrieved_passages"][j][key]) not in raw_data:
                                #     topics.append(key.replace("amp;", "").replace("&quot;",'\"'))
                                raw_data.append(" ".join(dial["retrieved_passages"][j][key]))
                                
        # print(len(topics))
        # print(len(list(set(topics))))
        # print(len(raw_data))
        # print(len(list(set(raw_data))))
        # print(raw_data)
        # topics = list(set(topics))
        raw_data = list(set(raw_data))
        return raw_data
    
    def _load_cmu_doc(self):
        raw_data = []
        topics = []

        filenames = os.listdir(self._data_dir+"/WikiData") 
        for name in filenames:   
            with open(os.path.join(self._data_dir+"/WikiData", name), 'r') as f:
                data = json.load(f)
            
            for key, value in data.items():
                if type(value) == str:
                    raw_data.append(value)
                elif type(value) == dict:
                    for k,v in value.items():
                        if k not in ['critical_response', 'movieName']:
                            if k == 'cast':
                                raw_data.append(" ".join([value['movieName'], k, v if type(v) == str else ", ".join(v)]))
                            else:
                                raw_data.append(" ".join([value['movieName'], k, v if type(v) == str else " ".join(v)]))
                        elif k == 'critical_response':
                            raw_data.append(" ".join(v))
        return raw_data


class MaskedDocReader(DocReader):
    def __init__(self,
                 data,
                 tokenizer,
                 dataset: str = "wow",
                 mode: str = "full",
                 max_length: int = 512,
                 data_dir: str = "./data",
                 model_type: str = "gpt2",
                 entity_spans: list = [],
                 time_spans: list = [],
                 random_masking: bool = True,
                 random_only: bool = False,
                 mask_ratio: float = 0.3,
                 scale: int = 10,
                 percent: float = 0.8,
                 **kwargs
                 ):
        self._mode = mode   # 
        self._max_length = max_length
        self._data_dir = data_dir
        self._model_type = model_type
        self._tokenizer = tokenizer

        self._task = dataset
        assert self._task == "wow" or self._task == "cmu_dog"

        self._entity_spans = entity_spans
        self._time_spans = time_spans
        self._random_masking = random_masking
        self._random_only = random_only
        self._mask_ratio = mask_ratio
        self._scale = scale
        self._percent = percent

        # read the datasets and do preprocessing
        pad_token_id = self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None else 0
        if data is None:
            data = self.load_doc()
        masked_data, label = self.preproc_doc(data)
        self.mask = get_mask(masked_data)
        self.data, _ = pad_sents(masked_data, pad_token=pad_token_id)
        self.label_mask = get_mask(label)
        self.label, _ = pad_sents(label, pad_token=pad_token_id)
        assert self.data is not None and self.label is not None
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long), torch.tensor(self.mask[idx], dtype=torch.long), torch.tensor(self.label[idx], dtype=torch.long), torch.tensor(self.label_mask[idx], dtype=torch.long)

    def preproc_doc(self, raw_data):
        masked_data, label = [], []
        intervals = self._split_input(raw_data)

        assert len(intervals) == len(raw_data)

        for ind, (text, interval) in tqdm(enumerate(zip(raw_data, intervals)), desc="Masking",total=len(intervals), ncols=100):
            text = text.encode("ascii", "ignore")
            text = text.decode()

            masked_lines, orig_lines = self._span_masking(text, interval, ind)

            tokenized_masked_lines = []
            tokenized_lines = []
            for line, orig_line in zip(masked_lines, orig_lines):
                tokenized_masked_lines.append(self._tokenizer.encode(line.lower()))
                tokenized_lines.append(self._tokenizer.encode(orig_line.lower()))

            masked_data.extend(tokenized_masked_lines)
            label.extend(tokenized_lines)
        return masked_data, label
    
    def _split_input(self, raw_data):
        """
        split the documents by the length after tokenization
        return ```split_data```: a list of [list of splited sentences]
        """
        intervals = []
        split_data = [] 

        diff = []
        for text in tqdm(raw_data,desc="Spliting",total=len(raw_data), ncols=100):
            text = text.encode("ascii", "ignore")
            text = text.decode()

            split_line = []
            tokenized_text = self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(text.lower()))   # TODO test lower case
            iter_num = len(tokenized_text) // (self._max_length-2)

            for i in range(0, iter_num):  # Truncate in block of max length
                split_line.append(self._tokenizer.decode(tokenized_text[i*(self._max_length-2) : (i + 1) * (self._max_length-2)]))
            if len(tokenized_text) - iter_num * (self._max_length-2) > 0:
                n = iter_num
                split_line.append(self._tokenizer.decode(tokenized_text[n * (self._max_length-2):]))

            split_data.append(split_line)

        for text, split_line in zip(raw_data, split_data):
            new_line = " ".join(split_line)
            orig_len = len(text)
            new_len = len(new_line)

            interval = []
            for ind, line in enumerate(split_line):
                span_len = int(len(line)/new_len*orig_len)
                if ind == 0:
                    start = 0
                else:
                    start = int((len(" ".join(split_line[:ind]))+1)/new_len*orig_len)
                end = start + span_len + 1
                interval.append([start, end])
                    
            intervals.append(interval)
        return intervals
    
    def _span_masking(self, doc, spans, ind):
        mask_token = self._tokenizer.mask_token
        masked_lines, orig_lines = split_span_masking(doc, spans, self._entity_spans[ind], self._time_spans[ind], mask_token=self._tokenizer.mask_token, random_masking=self._random_masking, \
            random_only=self._random_only, mask_ratio=self._mask_ratio, scale=self._scale, percent=self._percent)
        return masked_lines, orig_lines


class DialOrientDocReader(Dataset):
    def __init__(self,
                 data,
                 topics,
                 tokenizer,
                 dataset: str = "wow",
                 mode: str = "full",
                 max_length: int = 512,
                 data_dir: str = "./data",
                 model_type: str = "gpt2",
                 perm_times: int = 10,
                 **kwargs
                 ):
        self._mode = mode   # 
        self._max_length = max_length
        self._data_dir = data_dir
        self._model_type = model_type
        self._tokenizer = tokenizer
        self._eos_token = self._tokenizer.eos_token
        self._perm_times = perm_times

        self._task = dataset
        assert self._task == "wow" or self._task == "cmu_dog"

        # read the datasets and do preprocessing
        pad_token_id = self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None else 0  # for GPT2 model, it will just be 0
        if data is None:
            data = self.load_doc()
        preproc_data, token_type_ids, label_masks = self.preproc_doc(data, topics)
        self.mask = get_mask(preproc_data, max_len=self._max_length)
        self.data, _ = pad_sents(preproc_data, pad_token=pad_token_id, max_len=self._max_length)
        self.token_type_ids, _ = pad_sents(token_type_ids, pad_token=pad_token_id, max_len=self._max_length)
        self.label = self.data.copy()
        self.label_mask, _ = pad_sents(label_masks, pad_token=0, max_len=self._max_length)  # also don't compute the loss for the padding part

        assert self.data is not None

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long), torch.tensor(self.mask[idx], dtype=torch.long),  torch.tensor(self.token_type_ids[idx], dtype=torch.long), torch.tensor(self.label[idx], dtype=torch.long), torch.tensor(self.label_mask[idx], dtype=torch.long)

    def __len__(self):
        return len(self.data)
    
    @property
    def vocab(self):
        return self._tokenizer.vocab
    
    def load_doc(self):
        if self._task == "wow":
            data = self._load_wow_doc()
        elif self._task == "cmu_dog":
            data = self._load_cmu_doc()
        else:
            raise NotImplementedError
        return data
    
    def _permutation(self, sents):
        indexs = list(range(len(sents)))
        num_perm = round(len(sents) / 5)
        if num_perm < 1:
            return sents
        
        # When we can do at least one permutation
        perm_idxs = sorted(random.sample(indexs, 2 * num_perm))
        for i in range(num_perm):
            indexs[perm_idxs[2*i]], indexs[perm_idxs[2*i+1]] = indexs[perm_idxs[2*i+1]], indexs[perm_idxs[2*i]]

        permute_sents = [sents[i] for i in indexs]
        return permute_sents
    
    def _make_samples(self, permute_sents, tokenized_text=None, token_type_id=None, label_mask=None, include_topic=False):
        data_samples, token_type_ids, label_masks = [], [], []
        pre_add = 1 if include_topic else 0

        for idx, sent in enumerate(permute_sents):
            tokenized_sent = self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(sent.lower()))   # +" "+self._eos_token
            if len(tokenized_sent)> 150:  # remove the sentences that are hard to learn, emperical
                continue
            elif len(tokenized_sent) > 81: # max length of the response sequence in the training set of WoW
                tokenized_sent = tokenized_sent[:81]

            tokenized_sent += self._tokenizer.encode(" "+self._eos_token) # add eos token later in case it's truncated

            tokenized_text.append(tokenized_sent)
            token_type_id.append([(idx+1+pre_add) % 2] * len(tokenized_sent))
            label_mask.append([(idx+pre_add) % 2] * len(tokenized_sent))
        
        if len(tokenized_text) == 0:
            return [], [], []

        elif len(tokenized_text) == 1:
            data_samples.append(tokenized_text[0])
            token_type_ids.append(token_type_id[0])
            label_masks.append(label_mask[0])
        else:
            data_samples.append(tokenized_text[0]+tokenized_text[1])
            token_type_ids.append(token_type_id[0]+token_type_id[1])
            label_masks.append(label_mask[0]+label_mask[1])

            num_sample = math.floor(len(tokenized_text)/2-1)
            for idx in range(num_sample):
                i = idx * 2
                data_samples.append(tokenized_text[i]+tokenized_text[i+1]+tokenized_text[i+2]+tokenized_text[i+3])
                token_type_ids.append(token_type_id[i]+token_type_id[i+1]+token_type_id[i+2]+token_type_id[i+3])
                label_masks.append(label_mask[i]+label_mask[i+1]+label_mask[i+2]+label_mask[i+3])

        return data_samples, token_type_ids, label_masks
    
    def preproc_doc(self, raw_data, topics):
        preproc_data = []
        token_type_ids = []
        label_masks = []

        for topic, text in tqdm(zip(topics, raw_data), desc="Tokenize", total=len(raw_data)):
            sents = sent_tokenize(text)
            
            # every sents will be permuted for multiple times 
            # to make the model less dependent on the permutation of the knowledge sentences
            for _ in range(self._perm_times):
                # sentence permutation
                permute_sents = self._permutation(sents)

                sample_data, token_type_id, label_mask = self._make_samples(permute_sents, tokenized_text=[], token_type_id=[], label_mask=[], include_topic=False)
                preproc_data.extend(sample_data)
                token_type_ids.extend(token_type_id)
                label_masks.extend(label_mask)

                # add topic ahead of time
                tokenized_text = []
                token_type_id = []
                label_mask = []
                tokenized_topics = self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(topic.lower()+" "+self._eos_token))
                tokenized_text.append(tokenized_topics)
                token_type_id.append([1] * len(tokenized_topics))
                label_mask.append([0] * len(tokenized_topics))
                r_sample_data, r_token_type_id, r_label_mask = self._make_samples(permute_sents, tokenized_text=tokenized_text, token_type_id=token_type_id, label_mask=label_mask, include_topic=True)
                preproc_data.extend(r_sample_data)
                token_type_ids.extend(r_token_type_id)
                label_masks.extend(r_label_mask)

        return preproc_data, token_type_ids, label_masks
    
    def _load_wow_doc(self):
        raw_data = []
        topics = []

        if self._mode == "full":
            with open(os.path.join(self._data_dir, "data.json"), 'r') as f:
                data = json.load(f)
        elif self._mode == "seen":
            with open(os.path.join(self._data_dir, "train.json"), 'r') as f:
                train = json.load(f)
            with open(os.path.join(self._data_dir, "valid_random_split.json"), 'r') as f:
                valid = json.load(f)
            with open(os.path.join(self._data_dir, "test_random_split.json"), 'r') as f:
                test = json.load(f)
            data = train + valid + test
        elif self._mode == "unseen":
            with open(os.path.join(self._data_dir, "valid_topic_split.json"), 'r') as f:
                valid = json.load(f)
            with open(os.path.join(self._data_dir, "test_topic_split.json"), 'r') as f:
                test = json.load(f)
            data = valid + test
        else:
            raise ValueError("Invalid document type!")
        for item in tqdm(data,desc="Read",total=len(data)):
            for i in range(len(item["dialog"])):
                dial = item["dialog"][i]
                if "wizard" in dial["speaker"].lower():
                    for j in range(len(dial["retrieved_passages"])):
                        for key in dial["retrieved_passages"][j].keys():
                            if key.replace("amp;", "").replace("&quot;",'\"') in dial["retrieved_topics"]:
                                raw_data.append(" ".join(dial["retrieved_passages"][j][key]))
                                
        raw_data = list(set(raw_data))
        return raw_data
    
    def _load_cmu_doc(self):
        raw_data = []
        topics = []

        filenames = os.listdir(self._data_dir+"/WikiData") 
        for name in filenames:   
            with open(os.path.join(self._data_dir+"/WikiData", name), 'r') as f:
                data = json.load(f)
            
            for key, value in data.items():
                if type(value) == str:
                    raw_data.append(value)
                elif type(value) == dict:
                    for k,v in value.items():
                        if k not in ['critical_response', 'movieName']:
                            if k == 'cast':
                                raw_data.append(" ".join([value['movieName'], k, v if type(v) == str else ", ".join(v)]))
                            else:
                                raw_data.append(" ".join([value['movieName'], k, v if type(v) == str else " ".join(v)]))
                        elif k == 'critical_response':
                            raw_data.append(" ".join(v))
        return raw_data


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    with open("data/wiki_articles.txt", "r") as f:
        data = f.readlines()
    with open("data/wiki_topics.json", "r") as f:
        topics = json.load(f)
    # dataset = DocReader(None, tokenizer, mode="full", max_length=256, data_dir="./data/wizard_of_wikipedia")
    dataset = DialOrientDocReader(data, topics, tokenizer, mode="full", max_length=256, data_dir="./data/wizard_of_wikipedia", perm_times=1)
    # prepare dataloader
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=bsz,
                                            shuffle=True)

    # import pickle
    # from src.data_utils.masking import getClusters
    # with open("./data/wiki_entity/ctm_8_entity_0.pkl", "rb") as f:
    #     wiki_spans = pickle.load(f)

    # with open("./data/wiki_time/sutime_0_ctm8.pkl", "rb") as f:
    #     wiki_time = pickle.load(f)
    
    # wiki_spans = wiki_spans[:len(wiki_time)]

    # data = getClusters("./data/wiki_articles.txt", "./save/results/topics/ctm_topics_8.npy", 0)

    # dataset = MaskedDocReader(
    #     data,
    #     tokenizer, 
    #     max_length=128, 
    #     data_dir="./data",
    #     entity_spans=wiki_spans,
    #     time_spans=wiki_time,
    #     random_masking=True,
    #     random_only=False,
    #     mask_ratio=0.15,
    #     scale=0,
    #     )
    
    # loader = torch.utils.data.DataLoader(dataset=dataset,
    #                                      batch_size=1,
    #                                      shuffle=False)
    
    # for batch in loader:
    #     input_ids, input_masks, labels, label_masks = batch
    #     print("\ninput_ids", input_ids)
    #     print("input_masks", input_masks)
    #     print("labels", labels) 
    #     print("label_masks", label_masks) 
    #     print("input text", tokenizer.decode(input_ids[0]))
    #     print("label text", tokenizer.decode(labels[0]))
    #     input()
