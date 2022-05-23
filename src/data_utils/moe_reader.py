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

from src.data_utils.utils import pad_sents, get_mask, pad_list_of_sents, get_list_of_mask, convert_one_hot
from src.data_utils.data_reader import getDataLoader, load_wow_episodes
from src.data_utils.cmu_dog_reader import load_cmu_episodes

class MoeDialogReader(Dataset):
    def __init__(self,
                 episodes,
                 tokenizer,
                 expert_labels,
                 task: str = "wow",
                 mode: str = "train",
                 n_experts: int = 4,
                 max_length: int = 128,
                 max_context_length: int = 128,
                 max_kn_length: int = 128,
                 max_episode_length: int = 1,
                 data_dir: str = "./data",
                 history_in_context: bool = False,
                 kn_in_context: bool = False,
                 model_type: str = "decoder_only",
                 debug: bool = False,
                 ):
        self._max_length = max_length
        self._max_context_length = max_context_length
        self._max_kn_length = max_kn_length
        self._max_episode_length = max_episode_length
        self._data_dir = data_dir
        self._model_type = model_type
        self._task = task
        self._tokenizer = tokenizer
        self._debug = debug

        assert len(expert_labels)>0
        assert expert_labels.shape[1] == n_experts
        self._n_experts = n_experts
        self._expert_labels = expert_labels

        self.data = self.read(episodes, mode, history_in_context, kn_in_context)

    def __getitem__(self, idx):
        """Returns one data pair (source and target)."""
        item = {}
        item["episode_id"] = self.data["episode_id"][idx]
        item["sample_id"] = self.data["sample_id"][idx]
        for key in ["context", "response", "chosen_sentence", "title"]:
            item[key] = torch.LongTensor(self.data[key][idx])
            item[f"{key}_mask"] = torch.LongTensor(self.data[f"{key}_mask"][idx])

        for key in ["token_type"]:
            item[key] = torch.LongTensor(self.data[key][idx])
        
        item["knowledge"] = self.data["knowledge"][idx]
        item["topic"] = self.data["topic"][idx]
        item["topic_map"] = torch.FloatTensor(self.data["topic_map"][idx])
        
        return item

    def __len__(self):
        return len(self.data["episode_id"])

    def read(self, episodes, mode: str, history_in_context: bool, kn_in_context: bool):
        def _gen(episodes, sos_token, eos_token):
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
                "topic_map": [],           # list of list
                "chosen_sentence": [],     # list
                "sample_id": [],           # list
                "episode_id": [],          # list
                "knowledge": [],           # list of list
            }

            for _id, episode in enumerate(tqdm(episodes, desc="Generate samples", ncols=100)):
                contexts = episode["context"]                          # list
                response = episode["response"]                        # list
                title = episode["title"]                # list
                topic = episode["topic"]                # list
                checked_sentence = episode["checked_sentence"]
                knowledge = episode["knowledge"]
                topic_map = episode['topic_map']
                sample_id = episode["sample_id"] 
                episode_num = episode["episode_num"]                   # int
                episode_length = len(episode["context"])

                # get history
                history_head = self._tokenizer.encode(sos_token, add_special_tokens=False)   # list
                history_kn_head = self._tokenizer.encode(sos_token, add_special_tokens=False)
                token_type_ids_head = [1] * len(history_head)
                token_type_ids_kn_head = [0] * len(history_kn_head)

                # get history
                kn = self._tokenizer.encode(checked_sentence.strip() + " " + eos_token, add_special_tokens=False)
                history_kn_head += kn
                token_type_ids_kn_head += [0] * (len(kn) - 1) + [(self._max_episode_length+1) % 2]
                
                history = []
                token_type_ids = []

                if history_in_context:
                    pre_num = 1 if self._max_episode_length%2 == 0 else 0
                    for num in range(len(contexts)):
                        tempc = self._tokenizer.encode(contexts[num].strip() + " " + eos_token, add_special_tokens=False)
                        history += tempc
                        token_type_ids += [(num + pre_num) % 2] * (len(tempc) - 1) + [num % 2]

                    if kn_in_context:
                        context = history_kn_head + history
                        token_type_ids = token_type_ids_kn_head + token_type_ids
                    else:
                        context = history_head + history
                        token_type_ids = token_type_ids_head + token_type_ids
                else:
                    context =  self._tokenizer.encode(sos_token + " " + contexts[-1].strip() + " " + eos_token, add_special_tokens=False)
                    token_type_ids = [1] * len(context)

                if self._model_type == "seq2seq":
                    samples["context"].append(context)
                    samples["response"].append(self._tokenizer.encode(sos_token + " " + response.strip() + " " + eos_token, add_special_tokens=False))
                elif self._model_type == "decoder_only":
                    response = self._tokenizer.encode(response.strip() + " " + eos_token, add_special_tokens=False)
                    token_type_ids = token_type_ids[:-1] + [0] * (len(response) + 1)
                    response_mask = [1] * len(context) + [0] * len(response)
                    context += response
                    samples["context"].append(context)
                    samples["response"].append(response_mask)
                else:
                    raise ValueError(f"Model type '{self._model_type}' is invalid.")

                assert len(token_type_ids) == len(samples["context"][-1])
                samples["token_type"].append(token_type_ids)
                samples["title"].append(self._tokenizer.encode(title.strip()))
                samples["topic"].append(topic)
                samples["topic_map"].append(topic_map)
                samples["chosen_sentence"].append(self._tokenizer.encode(checked_sentence.strip()))
                samples["sample_id"].append(sample_id)
                samples["episode_id"].append(episode_num)
                samples["knowledge"].append(knowledge)
                
                if self._debug:
                    if len(samples["context"]) >= 100:
                        break
            return samples

        def _uniform(samples, history_in_context):
            """
            1. pad the sents in the same sample to the maximum length
            2. get the length of the inputs
            3. get the mask of the inputs
            """
            uniformed_samples = {}
            for key in samples:
                if "id" in key or "knowledge" in key or "topic" in key:
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
        episodes = self._preprocess_episodes(episodes, mode) if self._task == 'wow' else self._preprocess_episodes_cmu(episodes, mode)
        # formulate the samples
        sos_token = self._tokenizer.bos_token
        eos_token = self._tokenizer.eos_token

        samples = _gen(episodes, sos_token, eos_token)
        uniformed_samples = _uniform(samples, history_in_context)
        return uniformed_samples

    @property
    def vocab(self):
        return self._tokenizer.vocab

    def _preprocess_episodes(self, episodes, mode):
        """
        Add the topic and topic map according to the settings
        """
        for episode_num, episode in enumerate(tqdm(episodes, desc="Preprocess dataset", ncols=100)):
            topic = np.argmax(self._expert_labels[episode_num])  # TODO need to double check whether it's correct or not
            
            episode['topic'] = topic
            episode['topic_map'] = self._expert_labels[episode_num].tolist()
            if self._debug:
                if episode_num >= 100:
                    break

        return episodes

    def _preprocess_episodes_cmu(self, episodes, mode):
        """
        Tokenize all the fields in CMU-DoG.
        Return List[Dict[samples](episodes)]

        Output Example:
        [
            { # one episode
                'context': [], # in episode length
                'response': [],
                'title': [],
                'episode_num': int,
                'episode_length': int,
            }
            ...
            {
                # another episode
            }
        ]
        """
        new_episodes = []
        for episode_num, episode in enumerate(tqdm(episodes, desc="Preprocess dataset", ncols=100)):
            new_examples = {'context': [],
                            'response': '',
                            'title': '',
                            'topic': '',
                            'topic_map': [],
                            'checked_sentence': '',
                            'knowledge': [],
                            'sample_id': episode_num,
                            'episode_num': episode_num}

            # Tokenize inputs and convert to tokens
            context = [line.lower() for line in episode['text']]
            response = episode['response']
            title = 'no title' # example['title']
            topic = np.argmax(self._expert_labels[episode_num])  # TODO need to double check whether it's correct or not
            knowledges = [kn.lower().rstrip().split('\n') for kn in episode['knowledge']]

            new_examples['context'] = context
            new_examples['response'] = response.lower()
            new_examples['title'] = title.lower()
            new_examples['topic'] = topic
            new_examples['checked_sentence'] = 'no_sentence_provided'
            new_examples['knowledge'] = knowledges
            new_examples['topic_map'] = self._expert_labels[episode_num].tolist()

            new_episodes.append(new_examples)
        return new_episodes
    

def get_wow_topic_dataloader(args, split, cluster_path, tokenizer, bsz, shuffle=True, cal_time=False):
    episodes = load_wow_episodes(args.data_dir, split, args.history_in_context, args.max_episode_length, cal_time=cal_time)

    if os.path.exists(cluster_path):
        with open(cluster_path, "rb") as f:
            expert_labels = np.load(f)
            if args.kadapter_one_hot:
                expert_labels_idx = np.argmax(expert_labels, axis=1)
                expert_labels = np.zeros((expert_labels.shape[0], expert_labels.shape[1]))
                expert_labels[np.arange(expert_labels.shape[0]), expert_labels_idx] = 1
            elif args.kadapter_equal:
                expert_labels = np.ones((expert_labels.shape[0], expert_labels.shape[1] )) / expert_labels.shape[1]
            elif getattr(args, "expert_case", False):
                expert_labels = np.zeros((expert_labels.shape[0], expert_labels.shape[1]))
                expert_labels[np.arange(expert_labels.shape[0]), np.array([args.expert_idx]*expert_labels.shape[0])] = 1
            elif args.gold_cluster:
                expert_labels = convert_one_hot(expert_labels, args.n_experts)

    else:
        # expert_labels = get_cluster_labels(episodes, args, split, cluster_path)
        print("please run train_ctm.py first if you are not having cluster labels")
        raise NotImplementedError
    reader = MoeDialogReader(
        episodes,
        tokenizer,
        mode=split,
        n_experts=args.n_experts,
        expert_labels=expert_labels,
        max_length = args.max_length,
        max_context_length = args.max_context_length,
        max_kn_length = args.max_kn_length,
        max_episode_length = args.max_episode_length,
        data_dir = args.data_dir,
        history_in_context = args.history_in_context,
        kn_in_context = args.kn_in_context,
        model_type = args.model_type,
        debug = args.debug if hasattr(args, 'debug') else False,
    )
    loader = getDataLoader(reader, bsz, test=False if shuffle else True)
    return loader

def get_wow_topic_dataloaders(args, tokenizer, train=True, valid=True, shuffle_train=True, cal_time=False):
    if train:
        train_loader = get_wow_topic_dataloader(args, "train", args.cluster_path.replace("split", "train"), tokenizer, args.bsz, shuffle=True, cal_time=cal_time)
    else:
        train_loader = None

    if train or valid:
        valid_loader = get_wow_topic_dataloader(args, "valid", args.cluster_path.replace("split", "valid"), tokenizer, args.eval_bsz, shuffle=False, cal_time=cal_time)

        valid_unseen_loader = get_wow_topic_dataloader(args, "valid_unseen", args.cluster_path.replace("split", "valid_unseen"), tokenizer, args.eval_bsz, shuffle=False, cal_time=cal_time)
    else:
        valid_loader = None

        valid_unseen_loader = None

    test_loader = get_wow_topic_dataloader(args, "test", args.cluster_path.replace("split", "test"), tokenizer, args.eval_bsz, shuffle=False, cal_time=cal_time)

    test_unseen_loader = get_wow_topic_dataloader(args, "test_unseen", args.cluster_path.replace("split", "test_unseen"), tokenizer, args.eval_bsz, shuffle=False, cal_time=cal_time)

    dataloaders = {
        "train": train_loader,
        "valid": valid_loader,
        "valid_unseen": valid_unseen_loader,
        "test": test_loader,
        "test_unseen": test_unseen_loader,
    }
    return dataloaders

def get_cmu_dog_topic_dataloader(args, split, cluster_path, tokenizer, bsz, shuffle=True):
    episodes = load_cmu_episodes(args.data_dir, split)
    if os.path.exists(cluster_path):
        with open(cluster_path, "rb") as f:
            expert_labels = np.load(f)
            if args.kadapter_one_hot:
                expert_labels_idx = np.argmax(expert_labels, axis=1)
                expert_labels = np.zeros((expert_labels.shape[0], expert_labels.shape[1]))
                expert_labels[np.arange(expert_labels.shape[0]), expert_labels_idx] = 1
            elif args.kadapter_equal:
                expert_labels = np.ones((expert_labels.shape[0], expert_labels.shape[1] )) / expert_labels.shape[1]
            elif args.gold_cluster:
                expert_labels = convert_one_hot(expert_labels, args.n_experts)
    else:
        # expert_labels = get_cluster_labels(episodes, args, split, cluster_path)
        print("please run train_ctm.py first if you are not having cluster labels")
        raise NotImplementedError
    reader = MoeDialogReader(
        episodes,
        tokenizer,
        task="cmu_dog",
        mode=split,
        n_experts=args.n_experts,
        expert_labels=expert_labels,
        max_length = args.max_length,
        max_context_length = args.max_context_length,
        max_kn_length = args.max_kn_length,
        max_episode_length = args.max_episode_length,
        data_dir = args.data_dir,
        history_in_context = args.history_in_context,
        kn_in_context = args.kn_in_context,
        model_type = args.model_type,
        debug = args.debug if hasattr(args, 'debug') else False,
    )
    loader = getDataLoader(reader, bsz, test=False if shuffle else True)
    return loader

def get_cmu_dog_topic_dataloaders(args, tokenizer, train=True, valid=True, shuffle_train=True):
    if train:
        train_loader = get_cmu_dog_topic_dataloader(args, "train", args.cluster_path.replace("split", "train"), tokenizer, args.bsz, shuffle=True)
    else:
        train_loader = None

    if train or valid:
        valid_loader = get_cmu_dog_topic_dataloader(args, "valid", args.cluster_path.replace("split", "valid"), tokenizer, args.eval_bsz, shuffle=False)
    else:
        valid_loader = None

    test_loader = get_cmu_dog_topic_dataloader(args, "test", args.cluster_path.replace("split", "test"), tokenizer, args.eval_bsz, shuffle=False)
 
    dataloaders = {
        "train": train_loader,
        "valid": valid_loader,
        "test": test_loader,
    }
    return dataloaders

def get_data_from_batch(batch, model_type="decoder_only"):
    kn_sent = batch["chosen_sentence"]
    kn_mask = batch["chosen_sentence_mask"]

    topic = batch["topic"]
    topic_map = batch["topic_map"]

    if model_type == "seq2seq":
        inputs = batch["context"]
        masks = batch["context_mask"]
        labels = batch["response"]
        label_masks = batch["response_mask"]
        response_masks = None

        label_starts = torch.Tensor([0]*inputs.size(0))
        label_idxs = torch.sum(label_masks, 1)

        token_type_ids = batch["token_type"]
    else:
        seqlen = batch["context"].size(1)
        inputs = batch["context"].narrow(1, 0, seqlen - 1)
        masks = batch["context_mask"].narrow(1, 0, seqlen - 1)
        labels = batch["context"].narrow(1, 1, seqlen-1)
        label_masks = batch["context_mask"].narrow(1, 1, seqlen - 1)
        response_masks = batch["response"].narrow(1, 1, seqlen - 1)

        label_starts = torch.sum(response_masks, 1)
        label_idxs = torch.sum(label_masks, 1)

        token_type_ids = batch["token_type"].narrow(1, 0, seqlen - 1)

    return inputs, masks, kn_sent, kn_mask, topic, topic_map, \
        labels, label_masks, response_masks, label_starts, label_idxs, token_type_ids


if __name__ == '__main__':
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    episodes = load_wow_episodes('/home/user/data', 'test', True, max_episode_length = 1)
    print(f"The length of episodes is {len(episodes)}.")
    cluster_file_path = '/home/user/save/results/ctm_new/wow_test_4.npy'
    with open(cluster_file_path, "rb") as f:
        expert_labels = np.load(f)
    
    print("{expert_labels.shape}")
    reader = MoeDialogReader(
        episodes,
        tokenizer,
        task="wow",
        mode='valid',
        n_experts=4,
        expert_labels=expert_labels,
        max_length = 128,
        max_context_length = 128,
        max_kn_length = 128,
        max_episode_length = 1,
        data_dir = './data',
        history_in_context = True,
        kn_in_context = False
    )