import os	
import pickle	
import random	
import logging
import argparse	
import numpy as np	
from tqdm import tqdm, trange	
from datetime import datetime

import torch
import transformers	
from transformers import AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup	

from src.model.GPT2modeling import GPT2LMHeadModel

from src.data_utils.utils import build_input_for_seq2seq_model
from src.utils.training_utils import save_adapter, save_model, check_saving_path
from src.utils.cluster_utils import getClusters, getWikiClusters
from src.data_utils.doc_reader import DialOrientDocReader

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',filename=f'lm-finetune-{datetime.today().strftime("%m-%d-%H-%M-%S")}.log',filemode='w')
logger = logging.getLogger(__name__)	


def main(args):	
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')	
    # device = 'cpu'

    # check whether the saving path could be used or not, in case we get error after the first epoch
    check_saving_path(args.save_path)

    seed = args.seed # Added here for reproducibility	
    torch.manual_seed(seed)	
    np.random.seed(seed)
    random.seed(seed)	

    tokenizer = AutoTokenizer.from_pretrained(args.tok)	

    config = AutoConfig.from_pretrained(args.pretrained_model)
    config.kadapter = args.kadapter
    config.num_kadapter = 1
    config.task_adapter = args.task_adapter
    config.lm = args.lm
    config.n_neck = args.t_neck
    config.kn_neck = args.kn_neck
    config.dual_kadapter = args.dual_kadapter
    if "gpt2" in args.pretrained_model:
        model = GPT2LMHeadModel.from_pretrained(args.pretrained_model, config=config)
    model.to(device)

    # prepare data
    data, topics = getClusters(args.doc_path, args.cluster_path, args.index, data_dir=args.data_dir, \
                load_cmu=args.load_cmu, cmu_doc=args.cmu_doc, cmu_path=args.cmu_path, topic_modeling=args.tm, \
                load_topic=args.load_topic, topic_path=args.topic_path, cmu_topic_path=args.cmu_topic_path)
    
    if args.load_wiki:
        wiki_data, wiki_topics = getWikiClusters(args.wiki_path, args.index)
        print(f"The size of the knowledge corpus from WoW is {len(data)}.")
        data += wiki_data
        topics += wiki_topics
        print(f"The size of the wiki data is {len(wiki_data)}.")
        print(f"After adding wiki data, the size of the knowledge corpus is extend to {len(data)}.")
    
    if args.mlm:
        with open(args.ents_path, "rb") as f:
            wiki_spans = pickle.load(f)
        with open(args.time_path, "rb") as f:
            wiki_time = pickle.load(f)
        if args.load_cmu:
            with open(args.cmu_entity, "rb") as f:
                cmu_wiki_spans = pickle.load(f)
            with open(args.cmu_time, "rb") as f:
                cmu_wiki_time = pickle.load(f)
            wiki_spans += cmu_wiki_spans
            wiki_time += cmu_wiki_time
        assert len(wiki_spans) == len(data)
        assert len(wiki_time) == len(data)
    else:
        wiki_spans = []
        wiki_time = []
    
    if args.load_half_wow:
        half_wow = int(len(data)/2)
        data = data[:half_wow]
        topics = topics[:half_wow]

    if args.load_topic:
        dataset = DialOrientDocReader(
            data, 
            topics,
            tokenizer, 
            max_length=args.max_length,
            model_type=args.model_type,
            perm_times=args.perm_times,
            )
    elif args.mlm:
        dataset = DocReader(
            data, 
            tokenizer, 
            max_length=args.max_length,
            model_type=args.model_type,
            entity_spans=wiki_spans,
            time_spans=wiki_time,
            random_masking=args.random_masking,
            random_only=args.random_only,
            mask_ratio=args.mlm_probability,
            scale=args.scale,
            percent=args.percent,
            )
    else:
        dataset = DocReader(
            data, 
            tokenizer, 
            max_length=args.max_length,
            model_type=args.model_type,
            entity_spans=wiki_spans,
            time_spans=wiki_time,
            random_masking=args.random_masking,
            random_only=args.random_only,
            mask_ratio=args.mlm_probability,
            scale=args.scale,
            percent=args.percent,
            )

    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=args.bsz,
                                         shuffle=args.shuffle)	
    logger.info(f"Shuffle training data: {args.shuffle}")

    # finetuning	
    t_total = len(loader) // args.gradient_accumulation_steps * args.epoch	

    logger.info("***** Running training *****")	
    logger.info("  Num batches = %d", len(loader))	
    logger.info("  Num Epochs = %d", args.epoch)	
    logger.info("  Batch size = %d", args.bsz)	
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)	
    logger.info("  Total optimization steps = %d", t_total)	

    global_step = 0	
    epochs_trained = 0	
    steps_trained_in_current_epoch = 0	

    warmup_steps = int(args.warmup_steps * t_total)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)	
    scheduler = get_linear_schedule_with_warmup(	
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total	
    )		

    # Check if continuing training from a checkpoint	
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):	
        try:	
            # set global_step to gobal_step of last saved checkpoint from model path	
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]	
            epochs_trained = int(checkpoint_suffix)+1
            model.reset_kadapter_params(args.model_name_or_path+"/kadapter.pt")
            
            logger.info(f"  Loading the knowledge adapter from checkpoint {args.model_name_or_path}")
            logger.info("  Continuing training from checkpoint, will skip to saved epochs")	
            logger.info("  Continuing training from epoch %d", epochs_trained)	
            logger.info("  Continuing training from global step %d", global_step)	
        except ValueError:	
            logger.info("  Starting fine-tuning.")	
    
    if (	
        args.model_name_or_path	
        and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))	
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))	
    ):	
        # Load in optimizer and scheduler states	
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))	
        if scheduler:	
            scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    model.zero_grad()	
    train_iterator = trange(	
        epochs_trained, int(args.epoch), desc="Epoch", ncols=100)	

    tr_loss = 0.0	
    for epoch in train_iterator:	
        for step, batch in tqdm(enumerate(loader), desc="Iteration", total=len(loader), ncols=100):	
            # Skip past any already trained steps if resuming training	
            if steps_trained_in_current_epoch > 0:	
                steps_trained_in_current_epoch -= 1	
                continue	
            
            if args.load_topic:
                inputs, masks, token_type_ids, labels, label_masks = batch
            else:
                inputs, masks, labels, label_masks = batch
                token_type_ids = None

            if args.mlm:	
                decoder_inputs, decoder_masks, labels, label_masks = build_input_for_seq2seq_model(labels, label_masks)
                decoder_inputs = decoder_inputs.to(device)
                decoder_masks = decoder_masks.to(device)
                labels = labels.to(device)
                label_masks = label_masks.to(device)

            masked_indices = label_masks.bool()	
            labels[~masked_indices] = -100  # We only compute loss on masked tokens	
            inputs = inputs.to(device)	
            masks = masks.to(device)
            labels = labels.to(device)	

            model.train()
            if args.mlm:
                outputs = model(inputs, 
                                attention_mask=masks,
                                decoder_input_ids=decoder_inputs,
                                decoder_attention_mask=decoder_masks,
                                labels=labels,
                                )
            else:
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)	
                outputs = model(inputs, attention_mask=masks, token_type_ids=token_type_ids, labels=labels)	

            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)	

            if args.gradient_accumulation_steps > 1:	
                loss = loss / args.gradient_accumulation_steps	

            loss.backward()	

            tr_loss += loss.item()	
            if (step + 1) % args.gradient_accumulation_steps == 0:	
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)	
                optimizer.step()	
                if scheduler is not None:	
                    scheduler.step()  # Update learning rate schedule	
                model.zero_grad()	
                global_step += 1	

                if (step + 1) % ( args.log_steps * args.gradient_accumulation_steps)  == 0:	
                    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)	
                tr_loss = 0.0	
        
        if args.tm:
            if (epoch+1) % 10 == 0:
                save_adapter(args, tokenizer, model, optimizer, scheduler, epoch)	
        else:
            save_model(args, tokenizer, model, optimizer, scheduler, epoch)	



if __name__ == "__main__":	
    parser = argparse.ArgumentParser(description='LM Finetung')	
    # general and training	
    parser.add_argument("--bsz", type=int, default=32)	
    parser.add_argument("-ep", "--epoch", type=int, default=10)	
    parser.add_argument("--lr", type=float, default=1e-5)	
    parser.add_argument("-wd", "--weight-decay", type=float, default=0)	
    parser.add_argument("-gas","--gradient_accumulation_steps", type=int, default=1)	
    parser.add_argument("--log_steps", type=int, default=50)	
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")	
    parser.add_argument("--warmup_steps", default=0.0, type=float, help="Linear warmup over warmup_steps.")	
    parser.add_argument('--save_path', type=str, default="save")	
    parser.add_argument('--exp', type=str, default="gpt2-lm")	

    parser.add_argument(
        "--lm", action="store_true", help="Train with language modeling."
    )
    # For BERT-based models	
    parser.add_argument(	
        "--mlm", action="store_true", help="Train with masked-language modeling loss instead of language modeling."	
    )	
    parser.add_argument(	
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"	
    )	

    # Set random seed	
    parser.add_argument("--seed", type=int, default=0)	

    parser.add_argument('--data_dir', type=str, default="")	
    parser.add_argument('--tm', type=bool, default=False)
    parser.add_argument('--doc_path', type=str, default="./data/wiki_articles.txt")	
    parser.add_argument('--cluster_path', type=str, default=None)	
    parser.add_argument("--index", type=int, default=0)	
    parser.add_argument('--tok', type=str, default="gpt2")	
    parser.add_argument('--pretrained_model', type=str, default="gpt2")	
    parser.add_argument('-mnp','--model_name_or_path', type=str, default="")	
    parser.add_argument('--t_neck', help='the dimension of the bottleneck of the task adapter', type=int, default=256) 
    parser.add_argument('--task_adapter', action="store_true")	
    parser.add_argument('--kadapter', action="store_true")	
    parser.add_argument('--dual_kadapter', action="store_true")	
    parser.add_argument('--kn_neck', help='the dimension of the bottleneck of the knowledge adapter', type=int, default=256) 
    parser.add_argument('--max_length', type=int, default=512)	
    parser.add_argument('--mode', type=str, default="full")	
    parser.add_argument('--model_type', type=str, default="decoder_only")

    parser.add_argument('--ents_path', type=str, default="./data/wiki_entity/ctm_8_entity_0.pkl")
    parser.add_argument('--time_path', type=str, default="./data/wiki_time/sutime_0_ctm8.pkl")
    parser.add_argument('--random_masking', type=bool, default=True)
    parser.add_argument('--random_only', type=bool, default=False)
    parser.add_argument('--scale', type=int, default=10)
    parser.add_argument('--percent', type=float, default=1)

    parser.add_argument('--load_cmu', type=bool, default=False)
    parser.add_argument('--cmu_doc', type=str, default="./data/cmu_dog_docs.txt")		
    parser.add_argument('--cmu_path', type=str, default="./save/results/topics/cmu_lda_topics_8.npy")	
    parser.add_argument('--cmu_entity', type=str, default="./data/wiki_entity/ctm_8_entity_0_cmu.pkl")	
    parser.add_argument('--cmu_time', type=str, default="./data/wiki_time/sutime_0_ctm8_cmu.pkl")	

    parser.add_argument('--shuffle', action="store_true")

    parser.add_argument('--load_topic', action="store_true")
    parser.add_argument('--topic_path', type=str, default="./data/wiki_topics.json")		
    parser.add_argument('--cmu_topic_path', type=str, default="./data/cmu_dog_topics.json")
    parser.add_argument("--perm_times", type=int, default=10)	

    parser.add_argument('--load_wiki', action="store_true")
    parser.add_argument('--load_half_wow', action="store_true")
    parser.add_argument('--wiki_path', type=str, default="./data/sample_wiki_10/wiki_sample10_cluster8_idxNUM.txt")		

    args = parser.parse_args()	

    if args.mlm:
        logger.info("Pre-train model with masked language modeling!")
        from src.data_utils.doc_reader import MaskedDocReader as DocReader
    else:
        from src.data_utils.doc_reader import DocReader

    print(args)
    main(args)