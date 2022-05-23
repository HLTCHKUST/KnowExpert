python train_ctm.py \
--do_train \
--do_eval_doc \
--dataset data/wiki_articles.txt \
--vocab_path save/models/topic_models/ctm_new_vocab_20k.pkl \
--data_preparation_file save/models/topic_models/data_cache.pkl \
--model_path_prefix save/models/topic_models/ctm_20k_topics_ \
--output_path save/models/topic_models/ctm_20k_topics_NCLUSTER/DATASET_ctm_20k_topics_ \
--sbert_name sentence-transformers/stsb-roberta-base-v2

# save/models/topic_models/ctm_20k_topics_4/cmu_ctm_20k_topics_4.npy
# save/models/topic_models/ctm_20k_topics_4/wow_ctm_20k_topics_4.npy