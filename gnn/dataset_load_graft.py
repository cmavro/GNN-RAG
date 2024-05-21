import json
import numpy as np
import re
from tqdm import tqdm
import torch
from collections import Counter
import random
import warnings
import pickle
warnings.filterwarnings("ignore")
from modules.question_encoding.tokenizers import LSTMTokenizer#, BERTTokenizer
from transformers import AutoTokenizer
import time

import os

from dataset_load import BasicDataLoader

class GraftBasicDataLoader(BasicDataLoader):
    """ 
    Basic Dataloader contains all the functions to read questions and KGs from json files and
    create mappings between global entity ids and local ids that are used during GNN updates.
    """
    def __init__(self, config, word2id, relation2id, entity2id, tokenize, data_type):
        super(GraftBasicDataLoader, self).__init__(config, word2id, relation2id, entity2id, tokenize, data_type)

    def create_kb_adj_mats_facts(self, sample_id):
        sample = self.data[sample_id]
        g2l = self.global2local_entity_maps[sample_id]
        entity2fact_e, entity2fact_f = [], []
        fact2entity_f, fact2entity_e = [], []
        kb_fact_rels = np.full(self.max_facts, self.num_kb_relation, dtype=int)
        for i, tpl in enumerate(sample['subgraph']['tuples']):
            sbj, rel, obj = tpl
            try:

                if isinstance(sbj, dict) and  'text' in sbj:
                    head = g2l[self.entity2id[sbj['text']]]
                    rel = self.relation2id[rel['text']]
                    tail = g2l[self.entity2id[obj['text']]]
                else:
                    head = g2l[self.entity2id[sbj]]
                    rel = self.relation2id[rel]
                    tail = g2l[self.entity2id[obj]]
            except:
                head = g2l[sbj]
                try:
                    rel = int(rel)
                except:
                    rel = self.relation2id[rel]
                tail = g2l[obj]
            if not self.use_inverse_relation:
                entity2fact_e += [head]
                entity2fact_f += [i]
                fact2entity_f += [i]
                fact2entity_e += [tail]
                kb_fact_rels[i] = rel
            else:
                entity2fact_e += [head, tail]
                entity2fact_f += [2 * i, 2 * i + 1]
                fact2entity_f += [2 * i, 2 * i + 1]
                fact2entity_e += [tail, head]
                kb_fact_rels[2 * i] = rel
                kb_fact_rels[2 * i + 1] = rel + len(self.relation2id)
        kb_adj_mats = (np.array(entity2fact_f, dtype=int), \
            np.array(entity2fact_e, dtype=int), np.array([1.0] * len(entity2fact_f))),\
            (np.array(fact2entity_e, dtype=int), np.array(fact2entity_f, dtype=int), np.array([1.0] * len(fact2entity_e)))
        return kb_adj_mats, kb_fact_rels

    def _build_fact_mat_maxfacts(self, sample_ids, fact_dropout):
        """Create sparse matrix representation for batched data"""
        kb_fact_rels = np.full((len(sample_ids), self.max_facts), self.num_kb_relation, dtype=int)

        mats0_batch = np.array([], dtype=int)
        mats0_0 = np.array([], dtype=int)
        mats0_1 = np.array([], dtype=int)
        vals0 = np.array([], dtype=float)

        mats1_batch = np.array([], dtype=int)
        mats1_0 = np.array([], dtype=int)
        mats1_1 = np.array([], dtype=int)
        vals1 = np.array([], dtype=float)

        for i, sample_id in enumerate(sample_ids):
            ((mat0_0, mat0_1, val0), (mat1_0, mat1_1, val1)), kb_fact_rel = self.create_kb_adj_mats_facts(sample_id)
            kb_fact_rels[i] = kb_fact_rel
            assert len(val0) == len(val1)
            num_fact = len(val0)
            num_keep_fact = int(np.floor(num_fact * (1 - fact_dropout)))
            mask_index = np.random.permutation(num_fact)[ : num_keep_fact]
            # mat0
            mats0_batch = np.append(mats0_batch, np.full(len(mask_index), i, dtype=int))
            mats0_0 = np.append(mats0_0, mat0_0[mask_index])
            mats0_1 = np.append(mats0_1, mat0_1[mask_index])
            vals0 = np.append(vals0, val0[mask_index])
            # mat1
            mats1_batch = np.append(mats1_batch, np.full(len(mask_index), i, dtype=int))
            mats1_0 = np.append(mats1_0, mat1_0[mask_index])
            mats1_1 = np.append(mats1_1, mat1_1[mask_index])
            vals1 = np.append(vals1, val1[mask_index])

        return ((mats0_batch, mats0_0, mats0_1, vals0), (mats1_batch, mats1_0, mats1_1, vals1)), kb_fact_rels



class GraftSingleDataLoader(GraftBasicDataLoader):
    """
    Single Dataloader creates training/eval batches during KGQA.
    """
    def __init__(self, config, word2id, relation2id, entity2id, tokenize, data_type="train"):
        super(GraftSingleDataLoader, self).__init__(config, word2id, relation2id, entity2id, tokenize, data_type)
        
    def get_batch(self, iteration, batch_size, fact_dropout, q_type=None, test=False):
        start = batch_size * iteration
        end = min(batch_size * (iteration + 1), self.num_data)
        sample_ids = self.batches[start: end]
        self.sample_ids = sample_ids
        # true_batch_id, sample_ids, seed_dist = self.deal_multi_seed(ori_sample_ids)
        # self.sample_ids = sample_ids
        # self.true_sample_ids = ori_sample_ids
        # self.batch_ids = true_batch_id
        true_batch_id = None
        seed_dist = self.seed_distribution[sample_ids]
        q_input = self.deal_q_type(q_type)
        kb_adj_mats = self._build_fact_mat(sample_ids, fact_dropout=fact_dropout)
        kb_fact_rels = self.kb_fact_rels[sample_ids]
        kb_adj_mats_graft, _ = self._build_fact_mat_maxfacts(sample_ids, fact_dropout=fact_dropout)
        
        if test:
            return self.candidate_entities[sample_ids], \
                   self.query_entities[sample_ids], \
                   kb_adj_mats, \
                   kb_adj_mats_graft, \
                   q_input, \
                   kb_fact_rels, \
                   seed_dist, \
                   true_batch_id, \
                   self.answer_dists[sample_ids], \
                   self.answer_lists[sample_ids],\

        return self.candidate_entities[sample_ids], \
               self.query_entities[sample_ids], \
               kb_adj_mats, \
               kb_adj_mats_graft, \
               q_input, \
               kb_fact_rels, \
               seed_dist, \
               true_batch_id, \
               self.answer_dists[sample_ids]


def load_dict(filename):
    word2id = dict()
    with open(filename, encoding='utf-8') as f_in:
        for line in f_in:
            word = line.strip()
            word2id[word] = len(word2id)
    return word2id

def load_dict_int(filename):
    word2id = dict()
    with open(filename, encoding='utf-8') as f_in:
        for line in f_in:
            word = line.strip()
            word2id[int(word)] = int(word)
    return word2id

def load_data_graft(config, tokenize):

    """
    Creates train/val/test dataloaders (seperately).
    """
    if 'sr-cwq' in config['data_folder']:
        entity2id = load_dict_int(config['data_folder'] + config['entity2id'])
    else:
        entity2id = load_dict(config['data_folder'] + config['entity2id'])
    word2id = load_dict(config['data_folder'] + config['word2id'])
    relation2id = load_dict(config['data_folder'] + config['relation2id'])
    
    if config["is_eval"]:
        train_data = None
        valid_data = GraftSingleDataLoader(config, word2id, relation2id, entity2id, tokenize, data_type="dev")
        test_data = GraftSingleDataLoader(config, word2id, relation2id, entity2id, tokenize, data_type="test")
        num_word = test_data.num_word
    else:
        train_data = GraftSingleDataLoader(config, word2id, relation2id, entity2id, tokenize, data_type="train")
        valid_data = GraftSingleDataLoader(config, word2id, relation2id, entity2id, tokenize, data_type="dev")
        test_data = GraftSingleDataLoader(config, word2id, relation2id, entity2id, tokenize, data_type="test")
        num_word = train_data.num_word
    relation_texts = test_data.rel_texts
    relation_texts_inv = test_data.rel_texts_inv
    entities_texts = None
    dataset = {
        "train": train_data,
        "valid": valid_data,
        "test": test_data, #test_data,
        "entity2id": entity2id,
        "relation2id": relation2id,
        "word2id": word2id,
        "num_word": num_word,
        "rel_texts": relation_texts,
        "rel_texts_inv": relation_texts_inv,
        "ent_texts": entities_texts
    }
    return dataset


if __name__ == "__main__":
    st = time.time()
    #args = get_config()
    load_data_graft(args)
