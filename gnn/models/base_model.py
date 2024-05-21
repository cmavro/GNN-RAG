import torch
import numpy as np
import torch.nn as nn

import numpy as np

VERY_SMALL_NUMBER = 1e-10

class BaseModel(torch.nn.Module):
    """
    Base model functions: create embeddings, store relations, compute f1/h1 scores, etc.
    """

    def __init__(self, args, num_entity, num_relation, num_word):
        super(BaseModel, self).__init__()
        self.num_relation = num_relation
        self.num_entity = num_entity
        self.num_word = num_word
        print('Num Word', self.num_word)
        self.kge_frozen = args['kge_frozen']
        self.kg_dim = args['kg_dim']
        #self._parse_args(args)
        self.entity_emb_file = args['entity_emb_file']
        self.relation_emb_file = args['relation_emb_file']
        self.relation_word_emb = args['relation_word_emb']
        self.word_emb_file = args['word_emb_file']
        self.entity_dim = args['entity_dim']
        
        self.lm = args['lm']
        if self.lm in ['bert']:
            #self.word_dim = 768
            args['word_dim'] = 768
        
        self.word_dim = args['word_dim']

        self.rel_texts = None

        
        #self.share_module_def()
        #self.model_name = args['model_name'].lower()
        self.device = torch.device('cuda' if args['use_cuda'] else 'cpu')
       
        print("Entity: {}, Relation: {}, Word: {}".format(num_entity, num_relation, num_word))

        
        self.kld_loss = nn.KLDivLoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss()

        for k, v in args.items():
            if k.endswith('dim'):
                setattr(self, k, v)
            if k.endswith('emb_file') or k.endswith('kge_file'):
                if v is None:
                    setattr(self, k, None)
                else:
                    setattr(self, k, args['data_folder'] + v)

        self.reset_time = 0

        if 'use_inverse_relation' in args:
            self.use_inverse_relation = args['use_inverse_relation']
        if 'use_self_loop' in args:
            self.use_self_loop = args['use_self_loop']
        self.eps = args['eps']

        self.embedding_def()
        args['word_dim'] = self.word_dim
        
    def embedding_def(self):
        num_entity = self.num_entity
        num_relation = self.num_relation
        num_word = self.num_word

        if self.lm != 'lstm':
            self.word_dim = 768
            self.word_embedding = nn.Embedding(num_embeddings=num_word + 1, embedding_dim=self.word_dim,
                                           padding_idx=num_word)
        elif self.word_emb_file is not None:
            word_emb = np.load(self.word_emb_file)
            _ , self.word_dim = word_emb.shape
            print('Word emb dim', self.word_dim)
            self.word_embedding = nn.Embedding(num_embeddings=num_word + 1, embedding_dim=self.word_dim,
                                           padding_idx=num_word)
            self.word_embedding.weight = nn.Parameter(
                torch.from_numpy(
                    np.pad(np.load(self.word_emb_file), ((0, 1), (0, 0)), 'constant')).type(
                    'torch.FloatTensor'))
            self.word_embedding.weight.requires_grad = False
        else:
            #self.word_dim = 768
            self.word_embedding = nn.Embedding(num_embeddings=num_word + 1, embedding_dim=self.word_dim,
                                           padding_idx=num_word)


        if self.entity_emb_file is not None:
            self.encode_type = False
            emb = np.load(self.entity_emb_file)
            ent_num , self.ent_dim = emb.shape
            # if ent_num != num_entity:
            #     print('Number of entities in KG embeddings do not match: Random Init.')
            
            self.entity_embedding = nn.Embedding(num_embeddings=num_entity + 1, embedding_dim=self.ent_dim,
                                                padding_idx=num_entity)
            if ent_num != num_entity:
                print('Number of entities in KG embeddings do not match: Random Init.')
            else:
                self.entity_embedding.weight = nn.Parameter(
                    torch.from_numpy(np.pad(emb, ((0, 1), (0, 0)), 'constant')).type(
                        'torch.FloatTensor'))
            if self.kge_frozen:
                self.entity_embedding.weight.requires_grad = False
            else:
                self.entity_embedding.weight.requires_grad = True
        else:
            self.ent_dim = self.kg_dim 
            self.encode_type = True
            #self.entity_embedding = nn.Embedding(num_embeddings=num_entity + 1, embedding_dim=self.ent_dim,
                                                #padding_idx=num_entity)

        # initialize relation embedding
        if self.relation_emb_file is not None:
            np_tensor = self.load_relation_file(self.relation_emb_file)
            #print('check?', np_tensor.shape)
            rel_num, self.rel_dim = np_tensor.shape
            self.relation_embedding = nn.Embedding(num_embeddings=num_relation+1, embedding_dim=self.rel_dim)
            if rel_num != num_relation:
                 print('Number of relations in KG embeddings do not match: Random Init.')
            else:
                self.relation_embedding.weight = nn.Parameter(torch.from_numpy(np_tensor).type('torch.FloatTensor'))
            if self.kge_frozen:
                self.relation_embedding.weight.requires_grad = False
            else:
                self.relation_embedding.weight.requires_grad = True

        elif self.relation_word_emb:
            self.rel_dim = self.entity_dim
            self.relation_embedding = nn.Embedding(num_embeddings=num_relation+1, embedding_dim=self.rel_dim)
            self.relation_embedding.weight.requires_grad = True
            self.relation_embedding_inv = nn.Embedding(num_embeddings=num_relation+1, embedding_dim=self.rel_dim)
            self.relation_embedding_inv.weight.requires_grad = True
            pass
        else:
            self.rel_dim = 2*self.kg_dim 
            self.relation_embedding = nn.Embedding(num_embeddings=num_relation+1, embedding_dim=self.rel_dim)
            self.relation_embedding_inv = nn.Embedding(num_embeddings=num_relation+1, embedding_dim=self.rel_dim)

        # initialize text embeddings
        
        
    

    def load_relation_file(self, filename):
        half_tensor = np.load(filename)
        num_pad = 0
        if self.use_self_loop:
            num_pad = 2
        if self.use_inverse_relation:
            load_tensor = np.concatenate([half_tensor, half_tensor])
        else:
            load_tensor = half_tensor
        return np.pad(load_tensor, ((0, num_pad), (0, 0)), 'constant')

    def use_rel_texts(self, rel_texts, rel_texts_inv):
        self.rel_texts = torch.from_numpy(rel_texts).type('torch.LongTensor').to(self.device)
        self.rel_texts_inv = torch.from_numpy(rel_texts_inv).type('torch.LongTensor').to(self.device)

    def encode_rel_texts(self, rel_texts, rel_texts_inv):
        self.rel_texts = torch.from_numpy(rel_texts).type('torch.LongTensor').to(self.device)
        self.rel_texts_inv = torch.from_numpy(rel_texts_inv).type('torch.LongTensor').to(self.device)
        self.instruction.eval()
        with torch.no_grad():
            self.rel_features = self.instruction.encode_question(self.rel_texts, store=False)
            self.rel_features_inv = self.instruction.encode_question(self.rel_texts_inv, store=False)
        self.rel_features.requires_grad = False
        self.rel_features_inv.requires_grad = False

    def init_hidden(self, num_layer, batch_size, hidden_size):
        return self.instruction.init_hidden(num_layer, batch_size, hidden_size)

    def encode_question(self, q_input):
        return self.instruction.encode_question(q_input)

    def get_instruction(self, query_hidden_emb, query_mask, states):
        return self.instruction.get_instruction(query_hidden_emb, query_mask, states)

    def get_loss_bce(self, pred_dist_score, answer_dist):
        answer_dist = (answer_dist > 0).float() * 0.9   # label smooth
        # answer_dist = answer_dist * 0.9  # label smooth
        loss = self.bce_loss_logits(pred_dist_score, answer_dist)
        return loss

    def get_loss_kl(self, pred_dist, answer_dist):
        answer_len = torch.sum(answer_dist, dim=1, keepdim=True)
        answer_len[answer_len == 0] = 1.0
        answer_prob = answer_dist.div(answer_len)
        log_prob = torch.log(pred_dist + 1e-8)
        loss = self.kld_loss(log_prob, answer_prob)
        return loss

    def get_loss(self, pred_dist, answer_dist, reduction='mean'):
        if self.loss_type == "bce":
            tp_loss = self.get_loss_bce(pred_dist, answer_dist)
            if reduction == 'none':
                return tp_loss
            else:
                # mean
                return torch.mean(tp_loss)
        else:
            tp_loss = self.get_loss_kl(pred_dist, answer_dist)
            if reduction == 'none':
                return tp_loss
            else:
                # batchmean
                return torch.sum(tp_loss) / pred_dist.size(0)

    def f1_and_hits(self, answers, candidate2prob, eps=0.5):
        retrieved = []
        correct = 0
        cand_list = sorted(candidate2prob, key=lambda x:x[1], reverse=True)
        if len(cand_list) == 0:
            best_ans = -1
        else:
            best_ans = cand_list[0][0]
        # max_prob = cand_list[0][1]
        tp_prob = 0.0
        for c, prob in cand_list:
            retrieved.append((c, prob))
            tp_prob += prob
            if c in answers:
                correct += 1
            if tp_prob > eps:
                break
        if len(answers) == 0:
            if len(retrieved) == 0:
                return 1.0, 1.0, 1.0, 1.0  # precision, recall, f1, hits
            else:
                return 0.0, 1.0, 0.0, 1.0  # precision, recall, f1, hits
        else:
            hits = float(best_ans in answers)
            if len(retrieved) == 0:
                return 1.0, 0.0, 0.0, hits  # precision, recall, f1, hits
            else:
                p, r = correct / len(retrieved), correct / len(answers)
                f1 = 2.0 / (1.0 / p + 1.0 / r) if p != 0 and r != 0 else 0.0
                return p, r, f1, hits


    def calc_f1_new(self, curr_dist, dist_ans, h1_vec):
        batch_size = curr_dist.size(0)
        max_local_entity = curr_dist.size(1)
        seed_dist = self.seed_entities #self.dist_history[0]
        local_entity = self.local_entity
        ignore_prob = (1 - self.eps) / max_local_entity
        pad_ent_id = self.num_entity
        # hits_list = []
        f1_list = []
        for batch_id in range(batch_size):
            if h1_vec[batch_id].item() == 0.0:
                f1_list.append(0.0)
                # we consider cases which own hit@1 as prior to reduce computation time
                continue
            candidates = local_entity[batch_id, :].tolist()
            probs = curr_dist[batch_id, :].tolist()
            answer_prob = dist_ans[batch_id, :].tolist()
            seed_entities = seed_dist[batch_id, :].tolist()
            answer_list = []
            candidate2prob = []
            for c, p, p_a, s in zip(candidates, probs, answer_prob, seed_entities):
                if s > 0:
                    # ignore seed entities
                    continue
                if c == pad_ent_id:
                    continue
                if p_a > 0:
                    answer_list.append(c)
                if p < ignore_prob:
                    continue
                candidate2prob.append((c, p))
            precision, recall, f1, hits = self.f1_and_hits(answer_list, candidate2prob, self.eps)
            # hits_list.append(hits)
            f1_list.append(f1)
        # hits_vec = torch.FloatTensor(hits_list).to(self.device)
        f1_vec = torch.FloatTensor(f1_list).to(self.device)
        return f1_vec

    def calc_h1(self, curr_dist, dist_ans, eps=0.01):
        greedy_option = curr_dist.argmax(dim=-1, keepdim=True)
        dist_top1 = torch.zeros_like(curr_dist).scatter_(1, greedy_option, 1.0)
        dist_ans = (dist_ans > eps).float()
        h1 = torch.sum(dist_top1 * dist_ans, dim=-1)
        return (h1 > 0).float()
    
    def get_eval_metric(self, pred_dist, answer_dist):
        with torch.no_grad():
            h1 = self.calc_h1(curr_dist=pred_dist, dist_ans=answer_dist, eps=VERY_SMALL_NUMBER)
            f1 = self.calc_f1_new(pred_dist, answer_dist, h1)
        return h1, f1