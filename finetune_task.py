import os
import logging
import argparse
import numpy as np
from datetime import datetime

import torch
from torch import nn

import transformers
from transformers import AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup
from src.model.GPT2modeling import GPT2LMHeadModel

from src.data_utils.moe_reader import get_data_from_batch
from src.modules.trainer import Trainer
from src.data_utils.dialog_reader import get_wow_dataloader
from src.data_utils.cmu_dog_reader import get_cmu_dog_dataloader
from src.data_utils.moe_reader import MoeDialogReader, get_wow_topic_dataloaders, get_cmu_dog_topic_dataloaders

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',filename=f'finetune-{datetime.today().strftime("%m-%d-%H-%M-%S")}.log',filemode='w')
logger = logging.getLogger(__name__)


def main(args):
    # Fix random seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'   # for debug

    # load configuration
    config_to_load =  os.path.join(f"{args.save_path}/models/{args.exp}", f"model-{args.checkpoint}") if args.valid or args.test else args.pretrained_model
    config = AutoConfig.from_pretrained(config_to_load)
    config.task_adapter = args.task_adapter
    config.lm = args.lm 
    config.kadapter = args.kadapter
    config.n_neck = args.t_neck
    config.kn_neck = args.kn_neck
    config.dual_kadapter = args.dual_kadapter

    if args.moe:
        config.num_kadapter = args.n_experts if args.kadapter else 0
        config.kadapter_one_hot = args.kadapter_one_hot


    path_to_model = os.path.join(f"{args.save_path}/models/{args.exp}", f"model-{args.checkpoint}") if args.valid or args.test else args.pretrained_model
    # load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tok) if not (args.valid or args.test) else AutoTokenizer.from_pretrained(path_to_model)
    # log model
    if "gpt2" in args.pretrained_model:
        model = GPT2LMHeadModel.from_pretrained(path_to_model, config=config if not (args.valid or args.test) else None)
    else:
        raise ValueError("Invalid pretrained model type!")
    model = model.to(device)
    

    if args.dataset == "wow":
        dataloaders = get_wow_topic_dataloaders(args, tokenizer, train=False if args.test or args.valid else True, valid=True if not args.test else False) if args.moe \
                    else get_wow_dataloader(args, tokenizer, train=False if args.test or args.valid else True)
    elif args.dataset == "cmu_dog":
        dataloaders = get_cmu_dog_topic_dataloaders(args, tokenizer, train=False if args.test or args.valid else True, valid=True if not args.test else False) if args.moe \
                    else get_cmu_dog_dataloader(args, tokenizer, train=False if args.test or args.valid else True)
    else:
        raise ValueError("Invalid dataset!")

    t_total = len(dataloaders["train"]) // args.gradient_accumulation_steps * args.epoch if not (args.test or args.valid) else 0

    logger.info("***** Running training *****")
    logger.info("  Num batches = %d", len(dataloaders["train"]) if not (args.test or args.valid) else 0)
    logger.info("  Num Epochs = %d", args.epoch)
    logger.info("  Batch size = %d", args.bsz)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.model_name_or_path == "":
        args.model_name_or_path = None        
    if (
        args.model_name_or_path
        and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        logger.info("Reload the model ...")
        # Reload the model
        if "gpt2" in args.pretrained_model:
            model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
        else:
            raise ValueError("Invalid pretrained model type!")
        model = model.to(device)

        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        if scheduler:
            scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
        
        
    criterion = nn.CrossEntropyLoss()
    if args.moe and not args.scratch and not (args.test or args.valid):
        if args.dual_kadapter:
            print("Reset Dual Knowledge Adapter Parameters")
            logger.info("Reset Dual Knowledge Adapter Parameters")
            model.reset_dual_kadapter_params(args.kadapter_path, args.kadapter_ckp)
        else:
            print("Reset Knowledge Adapter Parameters")
            logger.info("Reset Knowledge Adapter Parameters")
            model.reset_kadapter_params(args.kadapter_path, args.kadapter_ckp)
    elif args.scratch:
        print("Initialize The Model Without Reseting Knowledge Adapter Parameters")
        logger.info("Initialize The Model Without Reseting Knowledge Adapter Parameters")
        
    trainer = Trainer(
        tokenizer=tokenizer,
        model=model,
        dataloader=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        args=args,
        get_data_from_batch_fn=get_data_from_batch,
    )

    if args.test or args.valid:
        logger.info(f"Evaluating the checkpoint {args.exp}-{args.checkpoint}!")
        trainer.test("seen", args.valid, args.test)
        if args.dataset == "wow":
            trainer.test("unseen", args.valid, args.test)
    else:
        if args.dataset == "wow":
            trainer.Train(eval_splits=["valid", "valid_unseen"])
        elif args.dataset == "cmu_dog":
            trainer.Train()
        
        trainer.load_best_model()
        logger.info("Evaluating the checkpoint `model-best`!")
        trainer.test("seen", args.valid, args.test)
        if args.dataset == "wow":
            trainer.test("unseen", args.valid, args.test)

            trainer.load_best_model(ckpt="best_seen")
            logger.info("Evaluating the checkpoint `model-best_seen`!")
            trainer.test("seen", args.valid, args.test)
            if args.dataset == "wow":
                trainer.test("unseen", args.valid, args.test)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LM Finetung')
    # general and training
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--eval_bsz", type=int, default=32)
    parser.add_argument("-ep", "--epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("-wd", "--weight-decay", type=float, default=0)
    parser.add_argument("-gas","--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--patience", help="Training patience for early stop", type=int, default=5)
    parser.add_argument('--save_path', type=str, default="save")
    parser.add_argument('--exp', type=str, default="gpt2-wow")
    parser.add_argument('--checkpoint', type=str, default="best")
    
    # KLD
    parser.add_argument('--train_kld', action="store_true")

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

    parser.add_argument('--dataset', type=str, default="wow")
    parser.add_argument('--data_dir', type=str, default="./data")
    parser.add_argument('--tok', type=str, default="gpt2")
    parser.add_argument('-mnp','--model_name_or_path', type=str, default="save/models/wow-history")
    parser.add_argument('--pretrained_model', type=str, default="gpt2")
    parser.add_argument('--kadapter', action="store_true")
    parser.add_argument('--dual_kadapter', action="store_true")	
    parser.add_argument('--n_experts', type=int, default=1)  # number of kadapter
    parser.add_argument('--kadapter_one_hot', action='store_true')  # if set True, top 1 kadapter else weighted sum 
    parser.add_argument('--kadapter_equal', action='store_true')  # if set True, the kadapters will be equally distributed
    parser.add_argument('--cluster_path', type=str, default='save/models/topic_models/ctm_4')
    parser.add_argument('--kadapter_path', type=str, default='/home/xuyan/dialog-kn/KnGroundedDial/save/models/gpt2-lm-topic-ctm-4')
    parser.add_argument('--kadapter_ckp', type=int, default=29)
    parser.add_argument('--kn_neck', help='the dimension of the bottleneck of the knowledge adapter', type=int, default=256) 
    parser.add_argument('--task_adapter', action="store_true")
    parser.add_argument('--t_neck', help='the dimension of the bottleneck of the task adapter', type=int, default=256)  

    parser.add_argument('--scratch', action="store_true")
    parser.add_argument('--max_length', type=int, default=128)     
    parser.add_argument('--max_context_length', type=int, default=128)     # 256 if --history_in_context
    parser.add_argument('--max_kn_length', type=int, default=128)
    parser.add_argument('--max_episode_length', type=int, default=1)       # history length
    parser.add_argument('-hic','--history_in_context', action="store_true")
    parser.add_argument('-kic','--kn_in_context', action="store_true")
    parser.add_argument('--mode', type=str, default="wizard")
    parser.add_argument('--model_type', type=str, default="decoder_only")

    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--valid', action="store_true")

    parser.add_argument('--moe', action="store_true")
    parser.add_argument('--expert_case', action="store_true")
    parser.add_argument('--expert_idx', type=int, default=0)

    parser.add_argument('--gold_cluster', action="store_true")

    args = parser.parse_args()


    main(args)