DATASET=$1
CLUSTER_PRED=$2
EXP_NAME=$3

python inference_task.py \
--dataset $DATASET \
--data_dir data \
--save_path save/results \
--length 200 \
--pretrained_model gpt2 \
--model_folder save/models \
--split test \
--ckpt "" \
--max_episode_length 2 \
--max_length 824 \
--max_context_length 824 \
--max_kn_length 824 \
--kadapter \
--kn_neck 768 \
-hic \
--moe \
--repetition_penalty 1.0 \
--n_experts 4 \
--cluster_path save/models/topic_models/$CLUSTER_PRED \
--checkpoint best \
--exp $EXP_NAME

