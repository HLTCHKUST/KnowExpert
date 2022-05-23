echo "Inference with both dialogue history and response"

python train_ctm.py \
--do_eval_dial \
--dataset wow \
--vocab_path save/models/topic_models/ctm_new_vocab_20k.pkl \
--data_preparation_file save/models/topic_models/data_cache.pkl \
--model_path_prefix save/models/topic_models/ctm_20k_topics_ \
--output_path save/models/topic_models/ctm_20k_new_hisres_NCLUSTER/wow_20k_new_hisres_ \
--sbert_name sentence-transformers/stsb-roberta-base-v2 \
--hisres


python train_ctm.py \
--do_eval_dial \
--dataset cmu \
--vocab_path save/models/topic_models/ctm_new_vocab_20k.pkl \
--data_preparation_file save/models/topic_models/data_cache.pkl \
--model_path_prefix save/models/topic_models/ctm_20k_topics_ \
--output_path save/models/topic_models/ctm_20k_new_hisres_NCLUSTER/cmu_20k_new_hisres_ \
--sbert_name sentence-transformers/stsb-roberta-base-v2 \
--hisres

# save/models/topic_models/ctm_20k_new_hisres_4/cmu_20k_new_hisres_test_4.npy
# save/models/topic_models/ctm_20k_new_hisres_4/wow_20k_new_hisres_test_4.npy


echo "Inference with dialogue history only"


python train_ctm.py \
--do_eval_dial \
--dataset wow \
--vocab_path save/models/topic_models/ctm_new_vocab_20k.pkl \
--data_preparation_file save/models/topic_models/data_cache.pkl \
--model_path_prefix save/models/topic_models/ctm_20k_topics_ \
--output_path save/models/topic_models/ctm_20k_new_SB_NCLUSTER/wow_20k_new_SB_ \
--sbert_name save/models/topic_models/his_only_sentenceBert


python train_ctm.py \
--do_eval_dial \
--dataset cmu \
--vocab_path save/models/topic_models/ctm_new_vocab_20k.pkl \
--data_preparation_file save/models/topic_models/data_cache.pkl \
--model_path_prefix save/models/topic_models/ctm_20k_topics_ \
--output_path save/models/topic_models/ctm_20k_new_SB_NCLUSTER/cmu_20k_new_SB_ \
--sbert_name save/models/topic_models/his_only_sentenceBert

# save/models/topic_models/ctm_20k_new_SB_4/cmu_20k_new_SB_test_4.npy
# save/models/topic_models/ctm_20k_new_SB_4/wow_20k_new_SB_test_4.npy