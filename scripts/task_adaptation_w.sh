DATASET=$1
CLUSTER_PRED=$2
EXP_NAME=$3

python finetune_task.py \
--dataset $DATASET \
-hic \
--moe \
--n_experts 4 \
--save_path save/models \
--kadapter_path save/models/gpt2-lm-topic-ctm-4-cmu-20k-kn768-new \
--kadapter_ckp 4 \
--exp $EXP_NAME \
--max_episode_length 2 \
--lr 1e-5 \
-ep 50 \
--kadapter \
--kn_neck 768 \
--cluster_path save/models/topic_models/$CLUSTER_PRED \
--bsz 16 \
--eval_bsz 32 \
--lm


python finetune_task.py \
--dataset $DATASET \
-hic \
--moe \
--n_experts 4 \
--save_path save/models \
--kadapter_path save/models/gpt2-lm-topic-ctm-4-cmu-20k-kn768-new \
--kadapter_ckp 4 \
--exp $EXP_NAME \
--max_episode_length 2 \
--lr 1e-5 \
-ep 50 \
--kadapter \
--kn_neck 768 \
--cluster_path save/models/topic_models/$CLUSTER_PRED \
--bsz 16 \
--eval_bsz 32 \
--lm \
--test \
--checkpoint best
