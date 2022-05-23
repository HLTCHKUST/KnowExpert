import os
import json
import pickle
import random
import math
from tqdm import tqdm
from operator import itemgetter
import pprint

from parlai.core.dict import DictionaryAgent
from parlai.core.worlds import create_task

import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AutoTokenizer

# from src.data_utils.vocab import Vocab
from src.data_utils.utils import pad_sents, get_mask, pad_list_of_sents, get_list_of_mask

def load_wow_episodes(data_dir, mode, history_in_context, max_episode_length, cal_time=False):
    def _get_parlai_opt(options): #: List[str] = []):
        from parlai.scripts.build_dict import setup_args
        parser = setup_args()
        opt = parser.parse_args(options)
        return opt
    """
    As default, it returns the following action dict:
    {
        'id': 'wizard_of_wikipedia'
        'text': chosen_topic\n # if first example in episode
                last_apprentice_message\n # if possible
                wizard_message # if --label-type is 'chosen_sent'
        'knowledge': title_1 sentence_1\n
                            .
                            .
                            .
                        title_m sentence_n # all knowledge available to wizard
        'labels': [title_checked sentence_checked] # default
                                OR
                    [wizard_response] # if --label-type set to 'response'
        'label_candidates': knowledge + [no_passages_used no_passages_used]
                                        OR
                            100 response candidates  # if 'validation' or 'test'
        'chosen_topic': chosen_topic as untokenized string
        'checked_sentence': checked sentence if wizard, else None # if --include_checked_sentence
        'title': title of checked sentence # if --include_checked_sentence
        --> if not exists, then checked_sentence = title = 'no_passages_used'
        'episode_done': (Boolean) whether episode is done or not
    }
    """
    parlai_opt = _get_parlai_opt([
        '--task', 'wizard_of_wikipedia:generator:topic_split' if 'unseen' in mode else 'wizard_of_wikipedia:generator:random_split',
        '--datatype', '{}:stream'.format(mode.split('_')[0]) if 'unseen' in mode else f'{mode}:stream',  # 'train' for shuffled data and 'train:stream' for unshuffled data
        '--datapath', data_dir,
        '--include_knowledge_separator', 'True',  # include speical __knowledge__ token between title and passage
        '--include_checked_sentence', 'True',
        '--label_type', 'response', # choices = ['response', 'chosen_sent']
    ])
    # As a default, world use "WizardDialogKnowledgeTeacher"
    agent = DictionaryAgent(parlai_opt)
    world = create_task(parlai_opt, agent)
    num_examples = world.num_examples()
    num_episodes = world.num_episodes()

    episodes = []
    for _ in range(num_episodes):
        examples = []
        while True:
            world.parley()
            example = world.acts[0]
            examples.append(example)
            if world.episode_done():
                episodes.append(examples)
                break
    return preprocess_wow_episodes(episodes, mode, history_in_context, max_episode_length=max_episode_length, cal_time=cal_time)

def preprocess_wow_episodes(episodes, mode, history_in_context, max_episode_length=1, cal_time=False):
    new_episodes = []
    for episode_num, episode in enumerate(tqdm(episodes, desc="Preprocess episodes", ncols=100)):
        for example_num, example in enumerate(episode):        
            new_examples = {'context': [],
                            'response': '',
                            'title': '',
                            'topic': '',
                            'checked_sentence': '',
                            'knowledge': [],
                            'sample_id': example_num,
                            'episode_num': episode_num}

            history = []
            if example_num != 0 and history_in_context:
                max_pair_length = math.ceil(max_episode_length/2)
                start_idx = max(0, example_num-max_pair_length)
                for num in range(start_idx, example_num):
                    if max_episode_length % 2 == 0 or num != start_idx:
                        history.append(episode[num]['text'].lower().strip())
                    history.append(episode[num]['labels'][0].lower().strip() if mode == "train" else episode[num]['eval_labels'][0].lower().strip())
                assert len(history) == max_episode_length
                
            context = history + [example['text'].lower().strip()]
            
            if mode == "train":
                response = example['labels'][0]
            else:
                response = example['eval_labels'][0]
            title = example['title']
            checked_sentence = example['checked_sentence']
            topic = example['chosen_topic']
            knowledges = example['knowledge'].rstrip().split('\n')

            new_examples['context'] = context
            new_examples['response'] = response.lower() 
            new_examples['title'] = title.lower()
            new_examples['topic'] = topic
            new_examples['checked_sentence'] = checked_sentence.lower()
            new_examples['knowledge'] = knowledges

            new_episodes.append(new_examples)
    
    if "test" in mode and cal_time:
        print("Testing the efficiency of the model!")
        with open(f"Efficiency/history_{mode}_ids.json", "r") as f:
            filter_ids = json.load(f)

        temp = []
        for idx in range(len(new_episodes)):
            if idx in filter_ids:
                temp.append(new_episodes[idx])
        new_episodes = temp
        print(f"There are {len(new_episodes)} samples in the final testing list.")
    return new_episodes


def getDataLoader(dataset, bsz, test=False):
    shuffle=False if test else True
    # prepare dataloader
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=bsz,
                                         shuffle=shuffle)
    return loader

if __name__ == "__main__":
    pass