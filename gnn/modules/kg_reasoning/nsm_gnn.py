import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import time

from .base_gnn import BaseGNNLayer

VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


class NSMBaseLayer(BaseGNNLayer):
    def __init__(self, args, num_entity, num_relation, entity_dim):
        super(NSMBaseLayer, self).__init__(args, num_entity, num_relation)
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.entity_dim = entity_dim
        self.num_steps = args['num_step']
        self.reason_kb = args['reason_kb']
        self.init_layers(args)

    def init_layers(self, args):
        entity_dim = self.entity_dim
        self.softmax_d1 = nn.Softmax(dim=1)
        self.score_func = nn.Linear(in_features=entity_dim, out_features=1)
        
        self.lin = nn.Linear(in_features=2*entity_dim, out_features=entity_dim)

        
        self.linear_dropout = args['linear_dropout']
        self.linear_drop = nn.Dropout(p=self.linear_dropout)
        for i in range(self.num_steps):
            self.add_module('rel_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
            self.add_module('e2e_linear' + str(i), nn.Linear(in_features=entity_dim + entity_dim, out_features=entity_dim))
            
    def init_reason(self, local_entity, kb_adj_mat, local_entity_emb, rel_features, query_node_emb=None):
        batch_size, max_local_entity = local_entity.size()
        self.local_entity_mask = (local_entity != self.num_entity).float()
        self.batch_size = batch_size
        self.max_local_entity = max_local_entity
        self.edge_list = kb_adj_mat
        self.rel_features = rel_features
        self.local_entity_emb = local_entity_emb
        self.num_relation = self.rel_features.size(0)
        self.possible_cand = []
        self.build_matrix()
        
       
    def reason_layer(self, curr_dist, instruction, rel_linear):
        pass

    def forward(self, current_dist, relational_ins, step=0, return_score=False):
        rel_linear = getattr(self, 'rel_linear' + str(step))
        e2e_linear = getattr(self, 'e2e_linear' + str(step))
        
        # score_func = getattr(self, 'score_func' + str(step))
        score_func = self.score_func
        relational_ins = relational_ins.squeeze(1)
        neighbor_rep, possible_tail = self.reason_layer(current_dist, relational_ins, rel_linear)
        next_local_entity_emb = torch.cat((self.local_entity_emb, neighbor_rep), dim=2)
        self.local_entity_emb = e2e_linear(self.linear_drop(next_local_entity_emb))
        
        self.local_entity_emb = F.relu(self.local_entity_emb)

        score_tp = score_func(self.linear_drop(self.local_entity_emb)).squeeze(dim=2)
        if self.reason_kb:
            answer_mask = self.local_entity_mask * possible_tail
        else:
            answer_mask = self.local_entity_mask
        self.possible_cand.append(answer_mask)
        score_tp = score_tp + (1 - answer_mask) * VERY_NEG_NUMBER
        current_dist = self.softmax_d1(score_tp)
        if return_score:
            return score_tp, current_dist
        return current_dist


    


class NSMLayer(NSMBaseLayer):
    def __init__(self, args, num_entity, num_relation, entity_dim):
        super(NSMLayer, self).__init__(args, num_entity, num_relation, entity_dim)

    def reason_layer(self, curr_dist, instruction, rel_linear):
        batch_size = self.batch_size
        max_local_entity = self.max_local_entity
        # num_relation = self.num_relation
        rel_features = self.rel_features
        
        fact_rel = torch.index_select(rel_features, dim=0, index=self.batch_rels) #rels (facts), entity_dim


        fact_query = torch.index_select(instruction, dim=0, index=self.batch_ids) #one query per batch entry: rels (facts), entity_dim
        fact_val = F.relu(rel_linear(fact_rel) * fact_query)
        fact_prior = torch.sparse.mm(self.head2fact_mat, curr_dist.view(-1, 1)) #rels (facts), 1 (scaling)


        possible_tail = torch.sparse.mm(self.fact2tail_mat, fact_prior) # batch_size * max_local_entity, 1
        # (batch_size *max_local_entity, num_fact) (num_fact, 1)
        possible_tail = (possible_tail > VERY_SMALL_NUMBER).float().view(batch_size, max_local_entity)

        fact_val = fact_val * fact_prior
        
        f2e_emb = torch.sparse.mm(self.fact2tail_mat, fact_val)  # batch_size * max_local_entity, entity_dim 
        assert not torch.isnan(f2e_emb).any()

        neighbor_rep = f2e_emb.view(batch_size, max_local_entity, self.entity_dim)
        
        return neighbor_rep, possible_tail

class NSMLayer_back(NSMBaseLayer):
    def __init__(self, args, num_entity, num_relation, entity_dim):
        super(NSMLayer_back, self).__init__(args, num_entity, num_relation, entity_dim)

    def reason_layer(self, curr_dist, instruction, rel_linear):
        batch_size = self.batch_size
        max_local_entity = self.max_local_entity
        # num_relation = self.num_relation
        rel_features = self.rel_features_inv
        
        fact_rel = torch.index_select(rel_features, dim=0, index=self.batch_rels)
        
        fact_query = torch.index_select(instruction, dim=0, index=self.batch_ids)
        fact_val = F.relu(rel_linear(fact_rel) * fact_query)
        fact_prior = torch.sparse.mm(self.tail2fact_mat, curr_dist.view(-1, 1))
        
        possible_head = torch.sparse.mm(self.fact2head_mat, fact_prior)
        # (batch_size *max_local_entity, num_fact) (num_fact, 1)
        possible_head = (possible_head > VERY_SMALL_NUMBER).float().view(batch_size, max_local_entity)

        fact_val = fact_val * fact_prior

        f2e_emb = torch.sparse.mm(self.fact2head_mat, fact_val)
        assert not torch.isnan(f2e_emb).any()

        neighbor_rep = f2e_emb.view(batch_size, max_local_entity, self.entity_dim)
        
        
        return neighbor_rep, possible_head