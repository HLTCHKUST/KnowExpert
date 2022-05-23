DATA_PATH=data
TOPIC_PATH=save/models/topic_models
OUTPUT_PATH=save/models
INDEX=$1

python lm_finetuning_cluster.py \
--data_dir ./data \
--tm True \
--load_topic \
--cluster_path $TOPIC_PATH/ctm_20k_topics_4/ctm_20k_topics_4.npy \
--index $INDEX \
--tok gpt2 \
--model_name_or_path gpt2 \
--max_length 256 \
--gradient_accumulation_steps 1 \
--bsz 12 \
--perm_times 10 \
--epoch 5 \
--lr 1e-4 \
--warmup_steps 0.05 \
--log_steps 1000 \
--exp gpt2-lm-topic-ctm-4-cmu-20k-kn768-new \
--doc_path $DATA_PATH/wiki_articles.txt \
--load_cmu True \
--cmu_doc $DATA_PATH/cmu_dog_docs.txt \
--cmu_path $TOPIC_PATH/ctm_20k_topics_4/cmu_ctm_20k_topics_4.npy \
--kadapter \
--save_path $OUTPUT_PATH \
--kn_neck 768 \
--shuffle   

