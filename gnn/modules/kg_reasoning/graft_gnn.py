import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import time

from .base_gnn import BaseGNNLayer

VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


class GraftLayer(BaseGNNLayer):
    def __init__(self, args, num_entity, num_relation, entity_dim):
        super(GraftLayer, self).__init__(args, num_entity, num_relation)
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.entity_dim = entity_dim
        self.entity_dim = entity_dim
        self.num_layer = args['num_layer']
        self.pagerank_lambda = args['pagerank_lambda']
        self.fact_scale = args['fact_scale']
        self.k = 3
        self.init_layers(args)

    def init_layers(self, args):
        entity_dim = self.entity_dim
        self.softmax_d1 = nn.Softmax(dim=1)
        self.score_func = nn.Linear(in_features=entity_dim, out_features=1)
        
        self.linear_dropout = args['linear_dropout']
        self.linear_drop = nn.Dropout(p=self.linear_dropout)
        for i in range(self.num_layer):
            self.add_module('q2e_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
            self.add_module('e2q_linear' + str(i), nn.Linear(in_features=self.k * entity_dim, out_features=entity_dim))
            self.add_module('e2e_linear' + str(i), nn.Linear(in_features=self.k * entity_dim, out_features=entity_dim))

            self.add_module('kb_head_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
            self.add_module('kb_tail_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
            self.add_module('kb_self_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))

           

    def init_reason(self, local_entity, kb_adj_mat, kb_adj_mat_graft, kb_fact_rel, local_entity_emb, rel_features, query_node_emb=None):
        batch_size, max_local_entity = local_entity.size()
        self.local_entity_mask = (local_entity != self.num_entity).float()
        self.batch_size = batch_size
        self.max_local_entity = max_local_entity
        self.edge_list = kb_adj_mat
        self.edge_list2 = kb_adj_mat_graft
        self.rel_features = rel_features
        self.local_entity_emb = local_entity_emb
        self.num_relation = self.rel_features.size(0)
        self.possible_cand = []
        self.kb_fact_rel = kb_fact_rel #torch.LongTensor(kb_fact_rel).to(self.device)
        _, self.max_fact = kb_fact_rel.shape
        self.query_node_emb = query_node_emb
        self.build_matrix()
        self.build_adj_facts()
        
    

    def compute_attention(self, query_hidden_emb, query_mask):
        batch_size = self.batch_size
        max_local_entity = self.max_local_entity
        rel_features = self.rel_features
        num_rels = rel_features.size(0)

        #print(self.kb_fact_rel, self.kb_fact_rel.size())
        fact_rel = torch.index_select(rel_features, dim=0, index=self.batch_rels)
        #batch_rels = rel_features[self.batch_ids, self.batch_rels]
        #print(fact_rel.size())
        local_fact_emb = rel_features[self.kb_fact_rel]#torch.sparse.mm(self.fact2rel_mat, fact_rel).view(batch_size, -1, self.entity_dim)
        
        # attention fact2question
        div = float(np.sqrt(self.entity_dim))
        fact2query_sim = torch.bmm(query_hidden_emb, local_fact_emb.transpose(1, 2)) / div # batch_size, max_query_word, max_fact
        fact2query_sim = self.softmax_d1(fact2query_sim + (1 - query_mask.unsqueeze(dim=2)) * VERY_NEG_NUMBER) # batch_size, max_query_word, max_fact
        fact2query_att = torch.sum(fact2query_sim.unsqueeze(dim=3) * query_hidden_emb.unsqueeze(dim=2), dim=1) # batch_size, max_fact, entity_dim
        W = torch.sum(fact2query_att * local_fact_emb, dim=2) / div # batch_size, max_fact
        W_max = torch.max(W, dim=1, keepdim=True)[0] # batch_size, 1
        self.W_tilde = torch.exp(W - W_max) # batch_size, max_fact
        e2f_softmax = torch.bmm(self.entity2fact_mat.transpose(1, 2), self.W_tilde.unsqueeze(dim=2)).squeeze(dim=2) # batch_size, max_local_entity
        self.e2f_softmax = torch.clamp(e2f_softmax, min=VERY_SMALL_NUMBER)
        #e2f_out_dim = use_cuda(Variable(torch.sum(entity2fact_mat.to_dense(), dim=1), requires_grad=False)) # batch_size, max_local_entity
        assert not torch.isnan(self.e2f_softmax).any()

    def reason_layer(self, curr_dist, kb_self_linear, kb_head_linear, kb_tail_linear):
        batch_size = self.batch_size
        max_local_entity = self.max_local_entity
        # num_relation = self.num_relation
        rel_features = self.rel_features

        local_fact_emb = rel_features[self.kb_fact_rel]
        e2f_emb = F.relu(kb_self_linear(local_fact_emb) + torch.bmm(self.entity2fact_mat, kb_head_linear(self.linear_drop(self.local_entity_emb)))) # batch_size, max_fact, entity_dim
        e2f_softmax_normalized = self.W_tilde.unsqueeze(dim=2) * torch.bmm(self.entity2fact_mat, (curr_dist / self.e2f_softmax).unsqueeze(dim=2)) # batch_size, max_fact, 1
        e2f_emb = e2f_emb * e2f_softmax_normalized # batch_size, max_fact, entity_dim
        f2e_emb = F.relu(kb_self_linear(self.local_entity_emb) + torch.bmm(self.fact2entity_mat, kb_tail_linear(self.linear_drop(e2f_emb))))
                
        next_curr_dist = torch.bmm(self.fact2entity_mat, e2f_softmax_normalized).squeeze(dim=2)
        next_curr_dist = self.pagerank_lambda * next_curr_dist + (1 - self.pagerank_lambda) * curr_dist # batch_size, max_local_entity

        assert not torch.isnan(f2e_emb).any()
        neighbor_rep = f2e_emb

        return neighbor_rep, next_curr_dist

    

    def forward(self, current_dist, query_hidden_emb, query_mask, step=0, return_score=True):
        # get linear transformation functions for each layer
        q2e_linear = getattr(self, 'q2e_linear' + str(step))
        e2e_linear = getattr(self, 'e2e_linear' + str(step))
        e2q_linear = getattr(self, 'e2q_linear' + str(step))
        kb_self_linear = getattr(self, 'kb_self_linear' + str(step))
        kb_head_linear = getattr(self, 'kb_head_linear' + str(step))
        kb_tail_linear = getattr(self, 'kb_tail_linear' + str(step))

        batch_size = self.batch_size
        max_local_entity = self.max_local_entity

        if step == 0:
            query_node_emb = self.query_node_emb#.unsqueeze(1)
            self.compute_attention(query_hidden_emb, query_mask)
            #q2e_emb = q2e_linear(self.linear_drop(self.query_node_emb)).expand(batch_size, max_local_entity, self.entity_dim)
        else:
            query_node_emb = self.query_emb#.unsqueeze(1)

        q2e_emb = q2e_linear(self.linear_drop(query_node_emb)).expand(batch_size, max_local_entity, self.entity_dim) # batch_size, max_local_entity, entity_dim

        next_local_entity_emb = torch.cat((self.local_entity_emb, q2e_emb), dim=2)
        # score_func = getattr(self, 'score_func' + str(step))
        score_func = self.score_func
        #relational_ins = relational_ins.squeeze(1)
        neighbor_rep, next_curr_dist = self.reason_layer(current_dist, kb_self_linear, kb_head_linear, kb_tail_linear)

        next_local_entity_emb = torch.cat((next_local_entity_emb, self.fact_scale*neighbor_rep), dim=2)
        #self.query_emb = torch.bmm(init_dist.unsqueeze(dim=1), e2q_linear(self.linear_drop(next_local_entity_emb)))
        self.query_emb = torch.bmm(next_curr_dist.unsqueeze(dim=1), e2q_linear(self.linear_drop(next_local_entity_emb)))

        self.local_entity_emb = F.relu(e2e_linear(self.linear_drop(next_local_entity_emb)))

        score_tp = score_func(self.linear_drop(self.local_entity_emb)).squeeze(dim=2)
        answer_mask = self.local_entity_mask
        self.possible_cand.append(answer_mask)
        score = score_tp + (1 - answer_mask) * VERY_NEG_NUMBER
        score = self.softmax_d1(score) #F.sigmoid(score) #* self.local_entity_mask #* answer_mask #+ (1 - answer_mask) * VERY_NEG_NUMBER
        #current_dist = self.softmax_d1(score_tp)
        current_dist = next_curr_dist
        if return_score:
            return score_tp, score, current_dist
        return score_tp, current_dist

