import os 
import sys 
import argparse
import json
import collections

import numpy as np
from tabulate import tabulate
from nltk import word_tokenize
from typing import List
from collections import defaultdict

from src.utils.parlai_utils import normalize_answer

import matplotlib.pyplot as plt

import nlp

def get_ngrams(text, n):
    """
    Returns all ngrams that are in the text.
    Note: this function does NOT lowercase text. If you want to lowercase, you should
    do so before calling this function.
    Inputs:
      text: string, space-separated
      n: int
    Returns:
      list of strings (each is a ngram, space-separated)
    """
    tokens = text.split()
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-(n-1))]  # list of str

def get_ngram_counter(text, n):
    """
    Returns a counter, indicating how many times each n-gram appeared in text.
    Note: this function does NOT lowercase text. If you want to lowercase, you should
    do so before calling this function.
    Input:
      text: is a string, with tokens space-separated.
    Returns:
      counter: mapping from each n-gram (a space-separated string) appearing in text,
        to the number of times it appears
    """
    ngrams = get_ngrams(text, n)
    counter = collections.Counter()
    counter.update(ngrams)
    return counter

def _distinct_n(sample, n):
    """
    Returns (total number of unique ngrams in story_text) / (total number of ngrams in story_text, including duplicates).
    Text is lowercased before counting ngrams.
    Returns None if there are no ngrams
    """
    # ngram_counter maps from each n-gram to how many times it appears
    ngram_counter = get_ngram_counter(sample.strip().lower(), n)
    if sum(ngram_counter.values()) == 0:
        print("Warning: encountered a story with no {}-grams".format(n))
        print(sample.strip().lower())
        print("ngram_counter: ", ngram_counter)
        return None
    return len(ngram_counter) / sum(ngram_counter.values())

def distinct_1(sample):
    return _distinct_n(sample, 1)

def distinct_2(sample):
    return _distinct_n(sample, 2)

def get_distinct(pred):
    dist_1 = []
    dist_2 = []
    for line in pred:
        d1 = distinct_1(line)
        d2 = distinct_2(line)
        if d1 is not None:
            dist_1.append(d1)
        if d2 is not None:
            dist_2.append(d2)
    return (np.mean(dist_1), np.mean(dist_2))

"""
Compute distinct metrics in corpus level
"""
def _distinct_corpus_n(samples, n):
    """
    Returns (total number of unique ngrams in story_text) / (total number of ngrams in story_text, including duplicates).
    Text is lowercased before counting ngrams.
    Returns None if there are no ngrams
    """
    # ngram_counter maps from each n-gram to how many times it appears
    ngram_counter = collections.Counter()
    for sample in samples:
        ngrams = get_ngrams(sample.strip().lower(), n)
        ngram_counter.update(ngrams)
    if sum(ngram_counter.values()) == 0:
        return None
    return len(ngram_counter) / sum(ngram_counter.values())

def distinct_corpus_1(samples):
    return _distinct_corpus_n(samples, 1)

def distinct_corpus_2(samples):
    return _distinct_corpus_n(samples, 2)

def distinct_corpus_3(samples):
    return _distinct_corpus_n(samples, 3)

def get_corpus_distinct(pred):
    return (distinct_corpus_1(pred), distinct_corpus_2(pred), distinct_corpus_3(pred))

"""
Compute unigram-F1 score
"""
def _prec_recall_f1_score(pred_items, gold_items):
    """
    PARLAI
    Computes precision, recall and f1 given a set of gold and prediction items.
    :param pred_items: iterable of predicted values
    :param gold_items: iterable of gold values
    :return: tuple (p, r, f1) for precision, recall, f1
    """
    common = collections.Counter(gold_items) & collections.Counter(pred_items)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(gold_items)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

"""
Compute bigram-F1 score
"""
def _bigram_prec_recall_f1_score(pred_items, gold_items):
    """
    PARLAI
    Computes precision, recall and f1 given a set of gold and prediction items.
    :param pred_items: iterable of predicted values
    :param gold_items: iterable of gold values
    :return: tuple (p, r, f1) for precision, recall, f1
    """
    common = get_ngram_counter(gold_items, 2) & get_ngram_counter(pred_items, 2)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(gold_items)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def get_unigram_F1(pred,gold):
    f1 = []
    for p,g in zip(pred,gold):
        p = normalize_answer(p)
        g = normalize_answer(g)
        f1.append(_prec_recall_f1_score(p.split(" "),g.split(" ")))
    return np.mean(f1)

def get_bigram_F1(pred, gold):
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    f1 = _bigram_prec_recall_f1_score(p, g)
    return f1

def compute_EM(pred,gold):
    EM = []
    for p,g in zip(pred,gold):
        p = normalize_answer(p)
        g = normalize_answer(g)
        EM.append(1 if p == g else 0)
    return np.mean(EM)



def _calc_ngram_dict(tokens:List[str], ngram:int, dict_ref=None):
    ngram_dict = defaultdict(int) if dict_ref is None else dict_ref
    total = len(tokens)
    for i in range(0, total - ngram + 1):
        item = tuple(tokens[i:i + ngram])
        ngram_dict[item] += 1
    return ngram_dict

def _calc_distinct_ngram(cands, ngram):
    ngram_total = 0.00001
    ngram_distinct_count = 0.00001
    pred_dict = defaultdict(int)
    for cand_tokens in cands:
        _calc_ngram_dict(cand_tokens, ngram, pred_dict)
    for key, freq in pred_dict.items():
        ngram_total += freq
        ngram_distinct_count += 1
    return ngram_distinct_count / ngram_total


def calc_corpus_distinct(cands):
    distinct1 = _calc_distinct_ngram(cands, 1)
    distinct2 = _calc_distinct_ngram(cands, 2)
    return distinct1, distinct2



def main(args):
    metrics = {}
    bleu_dict = {}

    pred_path = os.path.join(args.save_path, args.exp) + f"/{args.split}_{args.checkpoint}_generation.txt"
    gold_path = os.path.join(args.save_path, args.exp) + f"/{args.split}_{args.checkpoint}_gold.txt"

    with open(pred_path, "r") as f:
        preds = f.readlines()
    with open(gold_path, "r") as f:
        gold = f.readlines()

    for i, line in enumerate(preds):
        line = line.strip()
        if not len(line) > 0:
            print(f"/{args.split}_{args.checkpoint}_generation.txt", i)
    for i, line in enumerate(gold):
        line = line.strip()
        if not len(line) > 0:
            print(f"/{args.split}_{args.checkpoint}_gold.txt", i)

    if args.unigram_f1:
        f1 = get_unigram_F1(preds, gold)
        metrics["F1"] = f1
        print(f"{args.split} F1 score {f1}")
    
    if args.bigram_f1:
        r2 = get_bigram_F1(preds, gold)
        metrics["R2"] = r2
        print(f"{args.split} R2 score {r2}")
    
    if args.exact_match:
        EM = compute_EM(preds, gold)
        metrics["EM"] = EM
        print(f"{args.split} Exact Match score {EM}")

    if args.kn_f1:
        gold_kn_path = os.path.join(args.save_path, args.exp) + f"/{args.split}_kn.txt"
        with open(gold_kn_path, "r") as f:
            gold_kn = f.readlines()

        kf1 = get_unigram_F1(preds, gold_kn)
        metrics["KF1"] = kf1
        print(f"{args.split} KF1 score {kf1}")

    
    if args.multi_bleu:
        os.system(f"sh multi_bleu.sh {os.path.join(args.save_path, args.exp)} {args.split}_{args.checkpoint}_generation.txt {args.split}_{args.checkpoint}_gold.txt {args.split}_{args.checkpoint}_multi_bleu")

    if args.dist:
        help_tokenize = lambda x: word_tokenize(x.lower())
        cands = []
        for cand in preds:
            cands.append(help_tokenize(cand.lower()))
        cdiv1, cdiv2 = calc_corpus_distinct(cands)
        metrics["dist_1"] = cdiv1
        metrics["dist_2"] = cdiv2


    results = {}
    for k,v in metrics.items():
        results[k] = str(v)
    for k,v in bleu_dict.items():
        if k != "EmbeddingAverageCosineSimilairty":
            results[k] = str(v)

    # save in json file
    result_json = json.dumps(results, indent=4)
    if args.overwrite or not os.path.exists(os.path.join(os.path.join(args.save_path, args.exp), f"{args.split}_{args.checkpoint}_results")):
        with open(os.path.join(os.path.join(args.save_path, args.exp), f"{args.split}_{args.checkpoint}_results"), "w") as f:
            f.write(result_json)

    print(tabulate([results],headers="keys",tablefmt='latex',floatfmt=".4f"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default="valid")
    parser.add_argument('--checkpoint', type=str, default="8")
    parser.add_argument('--save_path', type=str, default="./save/results")
    parser.add_argument('--exp', help="The path to the experiment results", type=str, default="wow-history")

    parser.add_argument('-f1', '--unigram_f1', action="store_true")
    parser.add_argument('-kf1', '--kn_f1', action="store_true")
    parser.add_argument('-r2', '--bigram_f1', action="store_true")
    parser.add_argument('-em', '--exact_match', action="store_true")
    parser.add_argument('--bleu', action="store_true")
    parser.add_argument('--multi-bleu', action="store_true")
    parser.add_argument('--dist', action="store_true")

    parser.add_argument('--overwrite', action="store_true")

    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    sys.argv = sys.argv[:1]
    main(args)
