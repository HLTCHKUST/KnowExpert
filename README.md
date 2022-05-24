# KnowExpert

<img src="img/pytorch-logo-dark.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

<img align="right" src="img/caire.png" width="20%"> <img align="right" src="img/HKUST.jpeg" width="12%">

The implementation of the paper "Retrieval-Free Knowledge-Grounded Dialogue Response Generation with Adapters":

**Knowledge-Grounded Dialogue Response Generation with Adapters**. [**Yan Xu**](https://yana-xuyan.github.io), [**Etsuko Ishii**](https://etsukokuste.github.io/), [Samuel Cahyawijaya](https://samuelcahyawijaya.github.io/), [Zihan Liu](https://zliucr.github.io/), [Genta Indra Winata](https://gentawinata.com/), [Andrea Madotto](https://andreamad8.github.io), Dan Su, Pascale Fung **DialDoc@ACL2022** [[PDF]](https://aclanthology.org/2022.dialdoc-1.10.pdf)

If you use any source codes included in this toolkit in your work, please cite the following paper. The bibtex is listed below:

<pre>
@article{xu2021retrieval,
  title={Retrieval-free knowledge-grounded dialogue response generation with adapters},
  author={Xu, Yan and Ishii, Etsuko and Cahyawijaya, Samuel and Liu, Zihan and Winata, Genta Indra and Madotto, Andrea and Su, Dan and Fung, Pascale},
  journal={arXiv preprint arXiv:2105.06232},
  year={2021}
}
</pre>

## Install environment

```console
pip install -r requirements.txt
pip install -U contextualized_topic_models==2.0.1
```

## Prepare data

In this paper, we conduct experiments on Wizard of Wikipedia (WoW) 
and CMU DoG dataset. 

1. For WoW dataset, we download it from PARLAI source. For CMU_DoG 
  dataset, we need to download CMU_DoG dataset from the 
  [Github Repo](https://github.com/festvox/datasets-CMU_DoG). All 
  the data will be put it under `data` folder.

2. We also follow the same preprocess procedure on the CMU_DoG dataset 
  as what [ITDD paper](https://arxiv.org/abs/1907.08854) 
  ([Github](https://github.com/lizekang/ITDD)) has done. As there 
  is a little overlap between training set and testing set in the 
  original dataset, they remove the duplicates and format the data 
  for our model. The preprocessed data could be downloaded 
  [here](https://drive.google.com/file/d/16AcawDtG4HqUlQHV_zb4tZD4KNCAx_Vf/view?usp=sharing). 
  The data should be put under `data` folder and decompressed.

To make this process simple, just run the following command:

```console
sh scripts/prepare_data.sh
```

3. Based on the two datasets, we collect the involved articles as the 
  resource to pre-train the knowledge experts. We provide the pre-
  processed articles [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/yxucb_connect_ust_hk/EkLDqvQtkl9PtKN9LreNyskBUWABHkeH0zHyRlVlzfSm8g\?e\=KiqfEo).
  Please download the folder and move the content under `data` folder.

## Models

To use our model for evaluation, we provide the checkpoints of the models 
and the corresponding predictions.

Please download the models and the results from [OneDrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/yxucb_connect_ust_hk/EpATLSCrAgNAtxYkpQ6YN00B2vjtUUHbEukpXOFrQiyPow?e=fCh6dS) 
and put the downloaded folders under the `save` folder.


## Topic Modeling Training

### Train and inference on knowledge corpus

We train a CTM on the knowledge corpus and classify all the articles into 
a specific number of clusters:

```console
CUDA_VISIBLE_DEVICES=0 sh scripts/train_ctm.sh
```

### Further fine-tune on dialogue data

To better predict the topic distribution of the response from dialogue 
history, we further fine-tune sentenceBERT to minimize the MSE loss between 
the representation of `history + response` and that of `history-only`.

```console
CUDA_VISIBLE_DEVICES=0 python train_sentenceBERT.py -bs 8 --lr 1e-6 --wd 0 -ep 20 -pa 5 --output_dir save/models/topic_models/his_only_sentenceBert --do_train --do_eval --dataset wow
```

### Inference

Next, we classify all the dialogue samples into different clusters:

```console
CUDA_VISIBLE_DEVICES=0 sh scripts/inference_ctm_for_dials.sh
```


## Knowledge Expert Training

Now, we train the four knowledge experts individually

```console
CUDA_VISIBLE_DEVICES=0 sh scripts/expert_training.sh <index 0,1,2,3>
```

Here we provide the example for *#cluster* as 4 (same as the setting in our paper).

## Task Adaptation

The following scripts reproduce the task adaptation process, along with evaluating 
the perplexity of generating the gold responses. All the loss information will be 
saved into a log file.

- Task adaptation under weighted-sum setting

```console
CUDA_VISIBLE_DEVICES=0 sh scripts/task_adaptation_w.sh <dataset name:wow/cmu_dog> <cluster prediction path after `topic_models`> <exp name>
```

- Task adaptation under one-hot setting

```console
CUDA_VISIBLE_DEVICES=0 sh scripts/task_adaptation_o.sh <dataset name> <cluster prediction path after `topic_models`> <exp name>
```

For instance, to reproduce the *KnowExpert*_w results on WoW dataset, run:

```console
CUDA_VISIBLE_DEVICES=0 sh scripts/task_adaptation_w.sh wow ctm_20k_new_hisres_4/wow_20k_new_hisres_split_4.npy wow-moe-cluster4-ckp49-ctm-cmu-20knew-all-1e5-kn768-newp-hisres
```

## Inference

Now, we conduct inference on WoW and CMU DoG with the obtained models. The results 
will be saved under `save/results`.

- Inference under weighted-sum setting

```console
CUDA_VISIBLE_DEVICES=0 sh scripts/inference_w.sh <dataset name> <cluster prediction path after `topic_models`> <exp name>
```

- Inference under one-hot setting

```console
CUDA_VISIBLE_DEVICES=0 sh scripts/inference_o.sh <dataset name> <cluster prediction path after `topic_models`> <exp name>
```

For instance, to reproduce the *KnowExpert*_w results on WoW dataset, run:

```console
CUDA_VISIBLE_DEVICES=0 sh scripts/inference_w.sh wow ctm_20k_new_SB/wow_20k_new_SB_split_4.npy wow-moe-cluster4-ckp49-ctm-cmu-20knew-all-1e5-kn768-newp-hisres
```

## Generation Evaluation

In this paper, three autometic evaluation metrics are involved to evaluate the 
generated response: Uni-gram F1, Distinct-1, and Distinct-2.

```console
python evaluation.py --split test --checkpoint best --save_path save/results --exp <exp name> --unigram_f1 --dist
```
