
from tqdm import tqdm
tqdm.monitor_iterval = 0
import torch
import numpy as np
import math, os
import json
import pickle

def cal_accuracy(pred, answer_dist):
    """
    pred: batch_size
    answer_dist: batch_size, max_local_entity
    """
    num_correct = 0.0
    num_answerable = 0.0
    for i, l in enumerate(pred):
        num_correct += (answer_dist[i, l] != 0)
    for dist in answer_dist:
        if np.sum(dist) != 0:
            num_answerable += 1
    return num_correct / len(pred), num_answerable / len(pred)


def f1_and_hits(answers, candidate2prob, id2entity, entity2name, eps=0.5):
    ans = []
    retrieved = []
    for a in answers:
        if entity2name is None:
            ans.append(id2entity[a])
        else:
            ans.append(entity2name[id2entity[a]])
    correct = 0
    cand_list = sorted(candidate2prob, key=lambda x:x[1], reverse=True)
    if len(cand_list) == 0:
        best_ans = -1
    else:
        best_ans = cand_list[0][0]
    # max_prob = cand_list[0][1]
    tp_prob = 0.0
    for c, prob in cand_list:
        if entity2name is None:
            retrieved.append((id2entity[c], prob))
        else:
           retrieved.append((entity2name[id2entity[c]], prob))
        tp_prob += prob
        if c in answers:
            correct += 1
        if tp_prob > eps:
            break
    if correct > 0:
        em = 1
    else:
        em = 0
    if len(answers) == 0:
        if len(retrieved) == 0:
            return 1.0, 1.0, 1.0, 1.0, 1.0, 0, retrieved, ans  # precision, recall, f1, hits, em
        else:
            return 0.0, 1.0, 0.0, 1.0, 1.0, 1, retrieved , ans # precision, recall, f1, hits, em
    else:
        hits = float(best_ans in answers)
        if len(retrieved) == 0:
            return 1.0, 0.0, 0.0, hits, hits, 2, retrieved , ans # precision, recall, f1, hits, em
        else:
            p, r = correct / len(retrieved), correct / len(answers)
            f1 = 2.0 / (1.0 / p + 1.0 / r) if p != 0 and r != 0 else 0.0
            return p, r, f1, hits, em, 3, retrieved, ans


class Evaluator:

    def __init__(self, args, model, entity2id, relation2id, device):
        self.model = model
        self.args = args
        self.eps = args['eps']
        self.model_name = args["model_name"]
        
        id2entity = {idx: entity for entity, idx in entity2id.items()}
        self.id2entity = id2entity

        self.entity2name = None
        if 'sr-' in args["data_folder"]:
            file = open('ent2id.pickle', 'rb')
            self.entity2name = list((pickle.load(file)).keys())
            file.close()

            
        id2relation = {idx: relation for relation, idx in relation2id.items()}
        num_rel_ori = len(relation2id)

        if 'use_inverse_relation' in args:
            self.use_inverse_relation = args['use_inverse_relation']
            if self.use_inverse_relation:
                for i in range(len(id2relation)):
                    id2relation[i + num_rel_ori] = id2relation[i] + "_rev"

        if 'use_self_loop' in args:
            self.use_self_loop = args['use_self_loop']
            if self.use_self_loop:
                id2relation[len(id2relation)] = "self_loop"

        self.id2relation = id2relation
        self.file_write = None
        self.device = device

    def write_info(self, valid_data, tp_list, num_step):
        question_list = valid_data.get_quest()
        #num_step = steps
        obj_list = []
        if tp_list is not None:
            # attn_list = [tp[1] for tp in tp_list]
            action_list = [tp[0] for tp in tp_list]
        for i in range(len(question_list)):
            obj_list.append({})
        for j in range(num_step):
            if tp_list is None:
                actions = None
            else:
                actions = action_list[j]
                actions = actions.cpu().numpy()
            # if attn_list is not None:
            #     attention = attn_list[j].cpu().numpy()
            for i in range(len(question_list)):
                tp_obj = obj_list[i]
                q = question_list[i]
                # real_index = self.true_batch_id[i][0]
                tp_obj['question'] = q
                tp_obj[j] = {}
                # print(actions)
                if tp_list is not None:
                    action = actions[i]
                    rel_action = self.id2relation[action]
                    tp_obj[j]['rel_action'] = rel_action
                    tp_obj[j]['action'] = str(action)
                    # if attn_list is not None:
                    #     attention_tp = attention[i]
                    #     tp_obj[j]['attention'] = attention_tp.tolist()
        return obj_list

    def evaluate(self, valid_data, test_batch_size=20, write_info=False):
        write_info = True
        self.model.eval()
        self.count = 0
        eps = self.eps
        id2entity = self.id2entity
        eval_loss, eval_acc, eval_max_acc = [], [], []
        f1s, hits, ems,  precisions, recalls = [], [], [], [], []
        valid_data.reset_batches(is_sequential=True)
        num_epoch = math.ceil(valid_data.num_data / test_batch_size)
        if write_info and self.file_write is None:
            filename = os.path.join(self.args['checkpoint_dir'],
                                    "{}_test.info".format(self.args['experiment_name']))
            self.file_write = open(filename, "w")
        case_ct = {}
        max_local_entity = valid_data.max_local_entity
        ignore_prob = (1 - eps) / max_local_entity
        for iteration in tqdm(range(num_epoch)):
            batch = valid_data.get_batch(iteration, test_batch_size, fact_dropout=0.0, test=True)
            with torch.no_grad():
                loss, extras, pred_dist, tp_list = self.model(batch[:-1])
                pred = torch.max(pred_dist, dim=1)[1]
            if self.model_name == 'GraftNet':
                local_entity, query_entities, _, _, query_text, _, \
                seed_dist, true_batch_id, answer_dist, answer_list = batch
            else:
                local_entity, query_entities, _, query_text, \
                seed_dist, true_batch_id, answer_dist, answer_list = batch
            # self.true_batch_id = true_batch_id
            if write_info:
                obj_list = self.write_info(valid_data, tp_list, self.model.num_iter)
                # pred_sum = torch.sum(pred_dist, dim=1)
                # print(pred_sum)
            candidate_entities = torch.from_numpy(local_entity).type('torch.LongTensor')
            true_answers = torch.from_numpy(answer_dist).type('torch.FloatTensor')
            query_entities = torch.from_numpy(query_entities).type('torch.LongTensor')
            # acc, max_acc = cal_accuracy(pred, true_answers.cpu().numpy())
            eval_loss.append(loss.item())
            # eval_acc.append(acc)
            # eval_max_acc.append(max_acc)
            #pr_dist2 = pred_dist#.copy()
            #pred_dist = pr_dist2[-1]
            batch_size = pred_dist.size(0)
            batch_answers = answer_list
            batch_candidates = candidate_entities
            pad_ent_id = len(id2entity)
            #pr_dist2 = pred_dist.copy()
            #for pred_dist in pr_dist2:
            for batch_id in range(batch_size):
                answers = batch_answers[batch_id]
                candidates = batch_candidates[batch_id, :].tolist()
                probs = pred_dist[batch_id, :].tolist()
                seed_entities = query_entities[batch_id, :].tolist()
                #print(seed_entities)
                #print(candidates)
                candidate2prob = []
                for c, p, s in zip(candidates, probs, seed_entities):
                    if s == 1.0:
                        # ignore seed entities
                        #print(c, self.id2entity)
                        # print(c, p, s)
                        # if c < pad_ent_id:
                        #     tp_obj['seed'] = self.id2entity[c]
                        continue
                    if c == pad_ent_id:
                        continue
                    if p < ignore_prob:
                        continue
                    candidate2prob.append((c, p))
                precision, recall, f1, hit, em, case, retrived , ans = f1_and_hits(answers, candidate2prob, self.id2entity, self.entity2name ,eps)
                if write_info:
                    tp_obj = obj_list[batch_id]
                    tp_obj['answers'] = ans
                    tp_obj['precison'] = precision
                    tp_obj['recall'] = recall
                    tp_obj['f1'] = f1
                    tp_obj['hit'] = hit
                    tp_obj['em'] = em
                    tp_obj['cand'] = retrived
                    self.file_write.write(json.dumps(tp_obj) + "\n")
                case_ct.setdefault(case, 0)
                case_ct[case] += 1
                f1s.append(f1)
                hits.append(hit)
                ems.append(em)
                precisions.append(precision)
                recalls.append(recall)
        print('evaluation.......')
        print('how many eval samples......', len(f1s))
        # print('avg_f1', np.mean(f1s))
        print('avg_em', np.mean(ems))
        print('avg_hits', np.mean(hits))
        print('avg_f1', np.mean(f1s))
        print('avg_precision', np.mean(precisions))
        print('avg_recall', np.mean(recalls))
        
        print(case_ct)
        if write_info:
            self.file_write.close()
            self.file_write = None
        return np.mean(f1s), np.mean(hits), np.mean(ems)



