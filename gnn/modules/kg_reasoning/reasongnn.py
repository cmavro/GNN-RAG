
import torch
import torch.nn.functional as F
import torch.nn as nn


from .base_gnn import BaseGNNLayer

VERY_NEG_NUMBER = -100000000000

class ReasonGNNLayer(BaseGNNLayer):
    """
    GNN Reasoning
    """
    def __init__(self, args, num_entity, num_relation, entity_dim, alg):
        super(ReasonGNNLayer, self).__init__(args, num_entity, num_relation)
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.entity_dim = entity_dim
        self.alg = alg
        self.num_ins = args['num_ins']
        self.num_gnn = args['num_gnn']
        
        self.use_posemb = args['pos_emb']
        self.init_layers(args)

    def init_layers(self, args):
        entity_dim = self.entity_dim
        self.softmax_d1 = nn.Softmax(dim=1)
        self.score_func = nn.Linear(in_features=entity_dim, out_features=1)
        self.glob_lin = nn.Linear(in_features=entity_dim, out_features=entity_dim)
        self.lin = nn.Linear(in_features=2*entity_dim, out_features=entity_dim)
        assert self.alg == 'bfs'
        self.linear_dropout = args['linear_dropout']
        self.linear_drop = nn.Dropout(p=self.linear_dropout)
        for i in range(self.num_gnn):
            self.add_module('rel_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
            if self.alg == 'bfs':
                self.add_module('e2e_linear' + str(i), nn.Linear(in_features=2*(self.num_ins)*entity_dim + entity_dim, out_features=entity_dim))

            if self.use_posemb:
                self.add_module('pos_emb' + str(i), nn.Embedding(self.num_relation, entity_dim))
                self.add_module('pos_emb_inv' + str(i), nn.Embedding(self.num_relation, entity_dim))
        self.lin_m =  nn.Linear(in_features=(self.num_ins)*entity_dim, out_features=entity_dim)

    def init_reason(self, local_entity, kb_adj_mat, local_entity_emb, rel_features, rel_features_inv, query_entities, query_node_emb=None):
        batch_size, max_local_entity = local_entity.size()
        self.local_entity_mask = (local_entity != self.num_entity).float()
        self.batch_size = batch_size
        self.max_local_entity = max_local_entity
        self.edge_list = kb_adj_mat
        self.rel_features = rel_features
        self.rel_features_inv = rel_features_inv
        self.local_entity_emb = local_entity_emb
        self.num_relation = self.rel_features.size(0)
        self.possible_cand = []
        self.build_matrix()
        self.query_entities = query_entities
       

    def reason_layer(self, curr_dist, instruction, rel_linear, pos_emb):
        """
        Aggregates neighbor representations
        """
        batch_size = self.batch_size
        max_local_entity = self.max_local_entity
        # num_relation = self.num_relation
        rel_features = self.rel_features
        
        
        fact_rel = torch.index_select(rel_features, dim=0, index=self.batch_rels)
        
        fact_query = torch.index_select(instruction, dim=0, index=self.batch_ids)
        if pos_emb is not None:
            pe = pos_emb(self.batch_rels)
            # fact_rel = torch.cat([fact_rel, pe], 1)
            fact_val = F.relu((rel_linear(fact_rel)+pe) * fact_query)
        else :
            fact_val = F.relu(rel_linear(fact_rel) * fact_query)
        fact_prior = torch.sparse.mm(self.head2fact_mat, curr_dist.view(-1, 1))

        fact_val = fact_val * fact_prior
        
        f2e_emb = torch.sparse.mm(self.fact2tail_mat, fact_val)
        assert not torch.isnan(f2e_emb).any()

        neighbor_rep = f2e_emb.view(batch_size, max_local_entity, self.entity_dim)
        
        return neighbor_rep

    def reason_layer_inv(self, curr_dist, instruction, rel_linear, pos_emb_inv):
        batch_size = self.batch_size
        max_local_entity = self.max_local_entity
        # num_relation = self.num_relation
        rel_features = self.rel_features_inv
        
        fact_rel = torch.index_select(rel_features, dim=0, index=self.batch_rels)
        
        fact_query = torch.index_select(instruction, dim=0, index=self.batch_ids)
        if pos_emb_inv is not None:
            pe = pos_emb_inv(self.batch_rels)
            # fact_rel = torch.cat([fact_rel, pe], 1)
            fact_val = F.relu((rel_linear(fact_rel)+pe) * fact_query)
        else :
            fact_val = F.relu(rel_linear(fact_rel) * fact_query)
        fact_prior = torch.sparse.mm(self.tail2fact_mat, curr_dist.view(-1, 1))
        

        fact_val = fact_val * fact_prior

        f2e_emb = torch.sparse.mm(self.fact2head_mat, fact_val)
        assert not torch.isnan(f2e_emb).any()

        neighbor_rep = f2e_emb.view(batch_size, max_local_entity, self.entity_dim)
        
        return neighbor_rep

    def combine(self,emb):
        """
        Combines instruction-specific representations.
        """
        local_emb = torch.cat(emb, dim=-1)
        local_emb = F.relu(self.lin_m(local_emb))

        score_func = self.score_func
        
        score_tp = score_func(self.linear_drop(local_emb)).squeeze(dim=2)
        answer_mask = self.local_entity_mask
        self.possible_cand.append(answer_mask)
        score_tp = score_tp + (1 - answer_mask) * VERY_NEG_NUMBER
        current_dist = self.softmax_d1(score_tp)
        return current_dist, local_emb

    def forward(self, current_dist, relational_ins, step=0, return_score=False):
        """
        Compute next probabilistic vectors and current node representations.
        """
        rel_linear = getattr(self, 'rel_linear' + str(step))
        e2e_linear = getattr(self, 'e2e_linear' + str(step))
        # score_func = getattr(self, 'score_func' + str(step))
        score_func = self.score_func
        neighbor_reps = []
        
        if self.use_posemb :
            pos_emb = getattr(self, 'pos_emb' + str(step))
            pos_emb_inv = getattr(self, 'pos_emb_inv' + str(step))
        else :
            pos_emb, pos_emb_inv = None, None

        for j in range(relational_ins.size(1)):
            # we do the same procedure for existing and inverse relations
            neighbor_rep = self.reason_layer(current_dist, relational_ins[:,j,:], rel_linear, pos_emb)
            neighbor_reps.append(neighbor_rep)

            neighbor_rep = self.reason_layer_inv(current_dist, relational_ins[:,j,:], rel_linear, pos_emb_inv)
            neighbor_reps.append(neighbor_rep)

        neighbor_reps = torch.cat(neighbor_reps, dim=2)
        
        
        next_local_entity_emb = torch.cat((self.local_entity_emb, neighbor_reps), dim=2)
        #print(next_local_entity_emb.size())
        self.local_entity_emb = F.relu(e2e_linear(self.linear_drop(next_local_entity_emb)))

        score_tp = score_func(self.linear_drop(self.local_entity_emb)).squeeze(dim=2)
        answer_mask = self.local_entity_mask
        self.possible_cand.append(answer_mask)
        score_tp = score_tp + (1 - answer_mask) * VERY_NEG_NUMBER
        current_dist = self.softmax_d1(score_tp)
        if return_score:
            return score_tp, current_dist
        
        
        return current_dist, self.local_entity_emb 


