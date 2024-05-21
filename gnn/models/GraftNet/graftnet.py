import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

from models.base_model import BaseModel
from modules.kg_reasoning.graft_gnn import GraftLayer
from modules.question_encoding.lstm_encoder import LSTMInstruction#, BERTInstruction
from modules.question_encoding.bert_encoder import BERTInstruction#, BERTInstruction

from modules.layer_init import TypeLayer
from modules.query_update import AttnEncoder



VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


class GraftNet(BaseModel):
    def __init__(self, args, num_entity, num_relation, num_word):
        """
        num_relation: number of relation including self-connection
        """
        super(GraftNet, self).__init__(args, num_entity, num_relation, num_word)
        self.num_layer = args['num_layer']
        self.loss_type =  args['loss_type']
        self.model_name = args['model_name'].lower()
        self.lm = args['lm']
        self.norm_rel = args['norm_rel']
        self.num_iter = self.num_layer
        self.layers(args)
        self.private_module_def(args, num_entity, num_relation)
        self.to(self.device)

    def layers(self, args):
        # initialize entity embedding
        word_dim = self.word_dim
        kg_dim = self.kg_dim
        entity_dim = self.entity_dim

        self.linear_dropout = args['linear_dropout']
        
        self.entity_linear = nn.Linear(in_features=self.ent_dim, out_features=entity_dim)
        self.relation_linear1 = nn.Linear(in_features=self.rel_dim, out_features=entity_dim)

        # dropout
        self.linear_drop = nn.Dropout(p=self.linear_dropout)

        if self.encode_type:
            self.type_layer = TypeLayer(in_features=entity_dim, out_features=entity_dim,
                                        linear_drop=self.linear_drop, device=self.device, norm_rel=self.norm_rel)

        self.self_att_r = AttnEncoder(self.entity_dim)
        self.kld_loss = nn.KLDivLoss(reduction='none')
        self.bce_loss_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss()
    
    
    def private_module_def(self, args, num_entity, num_relation):
        # initialize entity embedding
        word_dim = self.word_dim
        kg_dim = self.kg_dim
        entity_dim = self.entity_dim
        self.reasoning = GraftLayer(args, num_entity, num_relation, entity_dim)
        if args['lm'] == 'lstm':
            self.instruction = LSTMInstruction(args, self.word_embedding, self.num_word)
        else:
            
            self.instruction = BERTInstruction(args, self.word_embedding, self.num_word, args['lm'])
            self.relation_linear = nn.Linear(in_features=self.word_dim, out_features=entity_dim)

    def get_ent_init(self, local_entity, kb_adj_mat, rel_features):
        if self.encode_type:
            local_entity_emb = self.type_layer(local_entity=local_entity,
                                               edge_list=kb_adj_mat,
                                               rel_features=rel_features)
        else:
            local_entity_emb = self.entity_embedding(local_entity)  # batch_size, max_local_entity, word_dim
            local_entity_emb = self.entity_linear(local_entity_emb)
        
        return local_entity_emb
    
    def get_rel_feature(self):
        if self.rel_texts is None:
            rel_features = self.relation_embedding.weight
            rel_features = self.relation_linear1(rel_features)
        else:
            #rel_features = self.instruction.encode_question(self.rel_texts, store=False)
            rel_features = self.rel_features # self.relation_linear(self.rel_features)
            #print(rel_features.size())
            #print(self.instruction.question_emb)
            rel_features = self.instruction.question_emb(rel_features)
            #rel_features = self.relation_linear(rel_features)
            rel_features = self.self_att_r(rel_features,  (self.rel_texts != self.instruction.pad_val).float())
            if self.lm == 'lstm':
                rel_features = self.self_att_r(rel_features, (self.rel_texts != self.num_relation+1).float())
            # else:
            #     rel_features = self.self_att_r(rel_features,  (self.rel_texts != self.instruction.pad_val).float())

        return rel_features
    
    
    def init_reason(self, curr_dist, local_entity, kb_adj_mat, kb_adj_mat_graft, kb_fact_rel, q_input):
        # batch_size = local_entity.size(0)
        self.local_entity = local_entity
        self.instruction_list, self.attn_list = self.instruction(q_input)
        self.query_hidden_emb = self.instruction.query_hidden_emb
        self.query_node_emb = self.instruction.query_node_emb
        self.query_mask = self.instruction.query_mask
        rel_features = self.get_rel_feature()
        self.local_entity_emb = self.get_ent_init(local_entity, kb_adj_mat, rel_features)
        self.curr_dist = curr_dist
        self.dist_history = []
        self.action_probs = []
        self.seed_entities = curr_dist


        self.reasoning.init_reason(local_entity=local_entity,
                                   kb_adj_mat=kb_adj_mat,
                                   kb_adj_mat_graft=kb_adj_mat_graft,
                                   kb_fact_rel = kb_fact_rel,
                                   local_entity_emb=self.local_entity_emb,
                                   rel_features=rel_features,
                                   query_node_emb=self.query_node_emb)
    
    def calc_loss_label(self, curr_dist, teacher_dist, label_valid):
        tp_loss = self.get_loss(pred_dist=curr_dist, answer_dist=teacher_dist, reduction='none')
        tp_loss = tp_loss * label_valid
        cur_loss = torch.sum(tp_loss) / curr_dist.size(0)
        return cur_loss


    def forward(self, batch, training=False):
        local_entity, query_entities, kb_adj_mat ,kb_adj_mat_graft,  query_text, kb_fact_rel, seed_dist, true_batch_id, answer_dist = batch
        local_entity = torch.from_numpy(local_entity).type('torch.LongTensor').to(self.device)

        # local_entity_mask = (local_entity != self.num_entity).float()
        query_entities = torch.from_numpy(query_entities).type('torch.FloatTensor').to(self.device)
        answer_dist = torch.from_numpy(answer_dist).type('torch.FloatTensor').to(self.device)
        seed_dist = torch.from_numpy(seed_dist).type('torch.FloatTensor').to(self.device)
        current_dist = Variable(seed_dist, requires_grad=True)

        q_input= torch.from_numpy(query_text).type('torch.LongTensor').to(self.device)
        if self.lm == 'bert':
            query_mask = (q_input != 0).float()
        else:
            query_mask = (q_input != self.num_word).float()
        #query_mask = (q_input != self.num_word).float()

        
        #instruction generation
        self.init_reason(curr_dist=current_dist, local_entity=local_entity,
                         kb_adj_mat=kb_adj_mat, kb_adj_mat_graft=kb_adj_mat_graft, kb_fact_rel=kb_fact_rel, q_input=q_input)
        self.instruction.init_reason(q_input)

        
        #reasoning
        self.curr_dist = current_dist   
        self.ent_dist = current_dist
        self.dist_history.append(self.curr_dist)
        for i in range(self.num_layer):
            score_tp, score, self.curr_dist= self.reasoning(self.curr_dist, self.query_hidden_emb, self.query_mask, step=i, return_score=True)
            self.dist_history.append(score)

        pred_dist = self.dist_history[-1]
        answer_number = torch.sum(answer_dist, dim=1, keepdim=True)
        case_valid = (answer_number > 0).float()
        loss = self.calc_loss_label(curr_dist=score_tp, teacher_dist=answer_dist, label_valid=case_valid)
        pred_dist = self.dist_history[-1]
        pred = torch.max(pred_dist, dim=1)[1]

        # answer_mask = self.local_entity_mask
        # self.possible_cand.append(answer_mask)
        # score_tp = score_tp + (1 - answer_mask) * VERY_NEG_NUMBER

        if training:
            h1, f1 = self.get_eval_metric(pred_dist, answer_dist)
            tp_list = [h1.tolist(), f1.tolist()]
        else:
            tp_list = None
        return loss, pred, pred_dist, tp_list