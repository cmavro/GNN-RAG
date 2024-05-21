import argparse
import glob
import json
import os
import re
import string
from sklearn.metrics import precision_score

import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import utils

from datasets import load_dataset

import json
with open('entities_names.json') as f:
    entities_names = json.load(f)
names_entities = {v: k for k, v in entities_names.items()}


def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1

def eval_acc(prediction, answer):
    matched = 0.
    for a in answer:
        if match(prediction, a):
            matched += 1
    return matched / len(answer)

def eval_hit(prediction, answer):
    for a in answer:
        if match(prediction, a):
            return 1
    return 0

def eval_hit1(prediction, answer):
    for a in answer:
        if match(prediction[0], a):
            return 1
    return 0

def eval_f1(prediction, answer):
    if len(prediction) == 0:
        return 0, 0, 0
    matched = 0
    prediction_str = ' '.join(prediction)
    for a in answer:
        if match(prediction_str, a):
            matched += 1
    precision = matched / len(prediction)
    recall = matched / len(answer)
    if precision + recall == 0:
        return 0, precision, recall
    else:
        return 2 * precision * recall / (precision + recall), precision, recall

def extract_topk_prediction(prediction, k=-1):
    results = {}
    for p in prediction:
        if p in results:
            results[p] += 1
        else:
            results[p] = 1
    if k > len(results) or k < 0:
        k = len(results)
    results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:k]]

def eval_result(predict_file1, encrypt=False, cal_f1=True, topk = -1):
    # predict_file = os.path.join(result_path, 'predictions.jsonl')
    
    # Load results
    acc_list = []
    hit1_list = []
    hit_list = []
    f1_list = []
    precission_list = []
    recall_list = []
    input_len = []
    data_file_d = {}
    data_file_dg = {}
    
    g_data_file = predict_file1 #'results/KGQA-G/RoG-cwq/RoG/test/results_gen_rule_path_RoG-cwq_RoG_test_predictions_3_False_jsonl/False/predictions.jsonl'


    input_file = os.path.join("rmanluo", "RoG-webqsp")
    rule_postfix = "no_rule"
    # Load dataset
    dataset = load_dataset(input_file, split="test")

    print(dataset)

    all_found = []
    counter = 0
    with  open(g_data_file, 'r') as fg:
        for lineg in fg:
            
            data = json.loads(lineg)

            id = data['id']
            prediction = data['prediction'].strip()
            answer = data['ground_truth']

            if True: 
                prediction = data['prediction'] 
                if not isinstance(prediction, list):
                    prediction = prediction.split("\n")
                prediction_str = ' '.join(prediction)
                answer = data["ground_truth"]
                example = dataset[counter]
                counter+=1
                graph = utils.build_graph(example['graph'], [], False)
                reasoning_paths = utils.get_truth_paths(example['q_entity'], data["ground_truth"], graph)

                found = 0
                for ans in data["ground_truth"]:
                    if ans in data['input']:
                        found = 1 
                
                q_entity = len(example["q_entity"])
                #"""
                hop = 1
                for path in reasoning_paths:
                    #print(path)
                    hop = max(hop,len(path))
                #"""
                #print(hop)
                if hop > 1:
                    all_found += [found]
                    input_len += [len(data['input']) / 4] #avg chars
                    # print("\n\n")
                    f1_score, precision_score, recall_score = eval_f1(prediction, answer)
                    f1_list.append(f1_score)
                    hit1 = eval_hit1(prediction, answer) #eval_hit(prediction_str, answer)
                    hit = eval_hit(prediction_str, answer)
                    hit1_list.append(hit1)
                    hit_list.append(hit)
    import statistics
    print("Input len: ", statistics.median(input_len))
    print("Coverage: ", statistics.mean(all_found))
    result_str = " Hit: " + str(sum(hit_list) * 100 / len(hit_list)) + " Hit1: " + str(sum(hit1_list) * 100 / len(hit1_list))  + " F1: " + str(sum(f1_list) * 100 / len(f1_list)) 
    print(result_str, len(hit1_list))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-d', type=str, default='results/KGQA-GNN-RAG/rearev-sbert/RoG-webqsp/RoG/test/results_gen_rule_path_RoG-webqsp_RoG_test_predictions_3_False_jsonl/')
    argparser.add_argument('--cal_f1', action="store_true")
    argparser.add_argument('--top_k', type=int, default=-1)
    args = argparser.parse_args()
    
    print(args.d)
    f1 = args.d + "False/predictions.jsonl"
    eval_result(f1, args.cal_f1, args.top_k)