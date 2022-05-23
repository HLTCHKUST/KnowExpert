import os
import json
import argparse
import logging
import random
import time

import numpy as np
from tqdm import tqdm

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer
from src.model.GPT2modeling import GPT2LMHeadModel

from src.data_utils.dialog_reader import get_wow_dataloader
from src.data_utils.cmu_dog_reader import get_cmu_dog_dataloader
from src.data_utils.moe_reader import get_wow_topic_dataloaders, get_cmu_dog_topic_dataloaders

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)

def prepare_inputs_for_generation(config, batch):
    if args.model_type=="decoder_only":
        inputs = batch["context"]
        masks = batch["response"]              # context 1 response 0 padding 0
        label_masks = batch["context_mask"]    # context 1 response 1 padding 0
        label_starts = torch.sum(masks, 1)
        label_idxs = torch.sum(label_masks, 1)

        inputs = inputs[:,:label_starts[0]]
        masks = masks[:,:label_starts[0]]
        if args.history_in_context and "token_type" in batch:
            token_type_ids = batch["token_type"][:,:label_starts[0]]
        else:
            token_type_ids = None
        
        assert inputs.shape[0] == 1
    elif args.model_type=="seq2seq":
        inputs = batch["context"]
        masks = batch["context_mask"]
        labels = batch["response"]
        label_masks = batch["response_mask"]

        token_type_ids = None
        label_starts = labels
        label_idxs = torch.sum(label_masks, 1)
    else:
        raise ValueError("Invalid model type when preparing inputs for generation!")
    
    if args.moe:
        experts = batch["topic_map"]
    else:
        experts = None
    return inputs, masks, token_type_ids, experts, label_starts, label_idxs


def prediction_step(args, model, batch, tokenizer, stop_token, pad_token_id, lens=None):
    inputs, masks, token_type_ids, experts, label_starts, label_idxs = prepare_inputs_for_generation(args, batch)
    kn_sent = batch["chosen_sentence"]
    knowledges = batch["knowledge"]

    if experts is not None:
        experts = experts.to(args.device)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(args.device)
    
    # max length
    max_length = args.length + inputs.shape[-1]

    start = time.time()
    generated_sequence = model.generate(
        input_ids=inputs.to(args.device),
        max_length=max_length, 
        temperature=args.temperature,
        top_k=args.k,
        top_p=args.p,
        repetition_penalty=args.repetition_penalty,
        do_sample=args.sampling,
        num_return_sequences=args.num_return_sequences,
        attention_mask=masks.to(args.device),
        experts=experts,
        token_type_ids=token_type_ids,
        pad_token_id=pad_token_id,
    )
    generated_sequence = generated_sequence[0]  # only take the first generated sentence

    if args.model_type=="decoder_only":
        generated_sequence = generated_sequence[inputs.shape[-1]:]
    
    gen_len = generated_sequence.size()[0]

    if args.debug:
        logger.info(f"The shape of the output sequences {len(generated_sequence)}.")

    # Decode text
    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)  # DO NOT skip_special_tokens
    if not args.debug:
        # Remove all text after the stop token
        text = text[: text.find(stop_token) if stop_token and text.find(stop_token)>0 else None].replace("</s>", "").replace("<s>", "")
    
    interval = time.time() - start

    response = batch["context"][0,label_starts[0]:label_idxs[0]-1] if args.model_type=="decoder_only" else label_starts[0,:label_idxs[0]-1]
    response_text = tokenizer.decode(response, clean_up_tokenization_spaces=True, skip_special_tokens=True)

    kn_text = tokenizer.decode(kn_sent[0], clean_up_tokenization_spaces=True)
    kn_text = kn_text[: kn_text.find(stop_token) if stop_token else None]

    if args.debug:
        print(f"The generated sentence is: {text}")
        print(f"The golden sentence is:    {response_text}")
        print(f"The knowledge sentence is: {kn_text}")
        print(f"The whole input is: {tokenizer.decode(inputs[0,:label_starts[0]], clean_up_tokenization_spaces=True)}")
        print(f"The knowledge candidates are:")
        for kn in knowledges:
            print(kn[0].split("__knowledge__"))
        print("="*80)
        input()

    return text, response_text, kn_text, interval, gen_len


def Inference(args):
    os.makedirs(f"{args.save_path}/{args.exp}", exist_ok=True)

    # Initialize the model and tokenizer
    model_name_or_path = os.path.join(args.model_folder, args.exp) + "/checkpoint-" + args.checkpoint if "best" not in args.checkpoint else os.path.join(args.model_folder, args.exp) + "/model-" + args.checkpoint

    config = AutoConfig.from_pretrained(model_name_or_path)
    config.lm = args.lm
    config.task_adapter = args.task_adapter
    config.n_neck = args.t_neck
    config.kadapter = args.kadapter
    config.kn_neck = args.kn_neck
    config.num_kadapter = args.n_experts if args.kadapter else 0
    config.kadapter_one_hot = args.kadapter_one_hot
    config.num_beams = 1 if not args.beam_search else 4
    config.dual_kadapter = args.dual_kadapter
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if "gpt2" in args.pretrained_model:
        model, loading_info = GPT2LMHeadModel.from_pretrained(model_name_or_path, config=config, output_loading_info=True)

    print("loading_info[missing_keys]", sorted(loading_info["missing_keys"]))
    assert len(loading_info["missing_keys"]) == len(loading_info["unexpected_keys"])
    assert len(loading_info["missing_keys"]) == 0

    # Load ckpt
    if args.ckpt != '':
        logger.info("Load the fine-tuned model...")
        model.load_state_dict(torch.load(args.ckpt),strict=False)
    model.to(args.device)

    # get dataloaders
    args.inference = True
    if args.dataset == "wow":
        if args.moe:
            dataloaders = get_wow_topic_dataloaders(args, tokenizer, train=False if args.split!="train" else True, valid=True if "valid" in args.split else False, cal_time=args.cal_time)
        else:
            dataloaders = get_wow_dataloader(args, tokenizer, train=False if args.split!="train" else True)
    elif args.dataset == "cmu_dog":
        if args.moe:
            dataloaders = get_cmu_dog_topic_dataloaders(args, tokenizer, train=False if args.split!="train" else True, valid=True if "valid" in args.split else False)
        else:
            dataloaders = get_cmu_dog_dataloader(args, tokenizer, train=False if args.split!="train" else True)
    else:
        raise ValueError("Invalid dataset!")

    if tokenizer.sep_token is None:
        stop_token = tokenizer.eos_token
    else:
        stop_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    print(f"The stop token is {stop_token}")


    if args.dataset == "wow":
        splits = [args.split, args.split+"_unseen"]
    elif args.dataset == "cmu_dog":
        splits = [args.split]
    else:
        raise NotImplementedError

    for split in splits:
        if os.path.exists(f'{args.save_path}/{args.exp}/{split}_{args.checkpoint}_generation.txt') and os.path.exists(f'{args.save_path}/{args.exp}/{split}_{args.checkpoint}_gold.txt') and not args.overwrite and not args.debug and not args.cal_time:
            logger.info(f"The result already exists and saved in '{args.save_path}/{args.exp}/{split}_{args.checkpoint}_generation.txt'! Skip inference!")
            continue

        logger.info(f"Doing inference for split {split}. Will save the results in '{args.save_path}/{args.exp}/{split}_{args.checkpoint}_generation.txt'.")
        loader = dataloaders[split]

        lens = None

        avg_lens = []
        total_time = 0

        generated_sequences = []
        golden_sequences = []
        kn_sequences = []
        for i, batch in tqdm(enumerate(loader), desc=f'Inference', total=len(loader), ncols=100):
            gen_text, response_text, kn_text, interval, gen_len = prediction_step(args, model, batch, tokenizer, stop_token, pad_token_id, lens=lens if lens is not None else None)
            total_time += interval
            avg_lens.append(gen_len)

            generated_sequences.append(gen_text)
            golden_sequences.append(response_text)
            kn_sequences.append(kn_text)

        if not args.debug and not args.cal_time:
            if args.expert_case:
                with open(f'{args.save_path}/{args.exp}/{split}_{args.checkpoint}_{args.expert_idx}_generation.txt', "w") as f:
                    for line in generated_sequences:
                        f.write(line.replace("\n", " ")+"\n")
            else:
                with open(f'{args.save_path}/{args.exp}/{split}_{args.checkpoint}_generation.txt', "w") as f:
                    for line in generated_sequences:
                        f.write(line.replace("\n", " ")+"\n")
            with open(f'{args.save_path}/{args.exp}/{split}_{args.checkpoint}_gold.txt', "w") as f:
                for line in golden_sequences:
                    f.write(line.replace("\n", " ")+"\n")
            with open(f'{args.save_path}/{args.exp}/{split}_kn.txt', "w") as f:
                for line in kn_sequences:
                    f.write(line.replace("\n", " ")+"\n")


        logger.info(f"Inference process takes {total_time} seconds.")
        logger.info(f"Generation average length is {np.mean(avg_lens)} tokens.")


def Comparison(args):
    ## Get a subset of experiment examples for comparison RANDOM INDEX
    l = random.sample(range(900), 100)
    l.sort()
    print(f"The selected sample indexes are {l}")

    results = [] # the item stored in is DICT: model name & index & generations
    # Initialize the model and tokenizer -> tokenizer + a list of results from different models
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    if tokenizer.sep_token is None:
        stop_token = tokenizer.eos_token
    else:
        stop_token = tokenizer.sep_token
    print(f"The stop token is {stop_token}")
    
    args.inference = True
    dataloaders = get_wow_dataloader(args, tokenizer, train=False if args.split != "train" else True, valid=True if "valid" in args.split else False, shuffle_train=False)
    split = args.split
    dataloader = dataloaders[split]

    for exp, ckpt in zip(args.exps, args.checkpoints):
        file_path = os.path.join(args.save_path, exp, f"{split}_{ckpt}_generation.txt")
        print(f"Loading file '{file_path}'......")
        with open(file_path, "r") as f:
            lines = f.readlines()

        generated_sequences = {}
        for i, line in enumerate(lines):
            generated_sequences[i] = line.strip()

        result = {
            "name": exp+"-"+ckpt,
            "text": generated_sequences,
        }
        results.append(result)

    for idx, batch in enumerate(dataloader):
        if idx not in l:
            continue
        inputs = batch["context"]
        masks = batch["response"]              # context 1 response 0 padding 0
        label_masks = batch["context_mask"]    # context 1 response 1 padding 0
        label_starts = torch.sum(masks, 1)
        label_idxs = torch.sum(label_masks, 1)

        kn_sent = batch["chosen_sentence"]
        knowledges = batch["knowledge"]
        topic = batch["title"]
        response = inputs[0,label_starts[0]:label_idxs[0]-1]
        response_text = tokenizer.decode(response, clean_up_tokenization_spaces=True)

        kn_text = tokenizer.decode(kn_sent[0], clean_up_tokenization_spaces=True)
        kn_text = kn_text[: kn_text.find(stop_token) if stop_token else None]
        topic_text = tokenizer.decode(topic[0], clean_up_tokenization_spaces=True)

        print(f"The knowledge candidates are:")
        for kn in knowledges:
            print(kn[0].split("__knowledge__"))
        print(f"The knowledge sentence is: {topic_text.strip()}\t{kn_text}")
        print(f"The whole input is: ")
        input_turns = tokenizer.decode(inputs[0,:label_starts[0]], clean_up_tokenization_spaces=True).split("<|endoftext|>")
        for turn in input_turns:
            print("\t" + turn.replace("\n","\t"))
        print(f"The golden response is:    {response_text}")
        for result in results:
            # print the results
            print(f"Model {result['name']}: {result['text'][idx]}")
        print("="*80)
        input()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model settings
    parser.add_argument(
        "--model_folder",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--ckpt",
        default="",
        type=str,
    )
    parser.add_argument('--exp', type=str, default="wow-history")
    parser.add_argument('--checkpoint', type=str, default="9")
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--eval_bsz", type=int, default=1)
    # generation settings
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--beam_search", action="store_true")
    parser.add_argument("--sampling", action="store_true")
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)

    # data settings
    parser.add_argument('--dataset', type=str, default="wow")
    parser.add_argument('--data_dir', type=str, default="./data")
    parser.add_argument('--save_path', type=str, default="./save/results")
    parser.add_argument('--pretrained_model', type=str, default="gpt2")
    parser.add_argument('--kadapter', action="store_true")
    parser.add_argument('--task_adapter', action="store_true")
    parser.add_argument('--dual_kadapter', action="store_true")	
    parser.add_argument('--n_experts', type=int, default=1)  # number of kadapter
    parser.add_argument('--kadapter_one_hot', action='store_true')  # if set True, top 1 kadapter else weighted sum
    parser.add_argument('--kadapter_equal', action='store_true')  # if set True, the kadapters will be equally distributed
    parser.add_argument('--cluster_path', type=str, default='save/models/topic_models/ctm_4')
    parser.add_argument('--kadapter_path', type=str, default='')
    parser.add_argument('--kadapter_ckp', type=int, default=29)
    parser.add_argument('--moe', action="store_true")  # moe mode
    parser.add_argument("--lm", action="store_true", help="Train with language modeling.")
    parser.add_argument('--t_neck', help='the dimension of the bottleneck of the task adapter', type=int, default=256)  
    parser.add_argument('--kn_neck', help='the dimension of the bottleneck of the knowledge adapter', type=int, default=256)  
    parser.add_argument('--max_length', type=int, default=1024)     
    parser.add_argument('--max_context_length', type=int, default=1024)     # 256 if --history_in_context
    parser.add_argument('--max_kn_length', type=int, default=1024)
    parser.add_argument('--max_episode_length', type=int, default=1)       # history length
    parser.add_argument('-hic','--history_in_context', action="store_true")
    parser.add_argument('-kic','--kn_in_context', action="store_true")
    parser.add_argument('--mode', type=str, default="wizard")
    parser.add_argument('--split', type=str, choices=["train", "valid", "test"], default="valid")
    parser.add_argument('--model_type', type=str, default="decoder_only")

    # ablation study
    parser.add_argument('--expert_case', action="store_true")
    parser.add_argument('--expert_idx', type=int, default=0)

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")

    # for debug
    parser.add_argument("--debug", action="store_true", help="Enter DEBUG mode")
    parser.add_argument("--overwrite", help="Overwrite the inference results even though it exists already", type=bool, default=False)

    # model comparison
    parser.add_argument("--compare", action="store_true", help="Enter COMPARE mode")
    parser.add_argument('--exps', help="The path to the experiment results", action="append")
    parser.add_argument('--checkpoints', help="The checkpoint corresponding the experiment results", action="append")

    parser.add_argument("--cal_time", action="store_true", help="Calculate the inference time.")

    parser.add_argument('--gold_cluster', action="store_true")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.compare:
        Comparison(args)
    else:
        Inference(args)
