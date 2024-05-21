import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

from models.base_model import BaseModel
from modules.kg_reasoning.nsm_gnn import NSMLayer, NSMLayer_back
from modules.question_encoding.lstm_encoder import LSTMInstruction#, BERTInstruction
from modules.question_encoding.bert_encoder import BERTInstruction
from modules.layer_init import TypeLayer
from modules.query_update import AttnEncoder


VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


class NSM(BaseModel):
    def __init__(self, args, num_entity, num_relation, num_word):
        """
        num_relation: number of relation including self-connection
        """
        super(NSM, self).__init__(args, num_entity, num_relation, num_word)
        self.num_step = args['num_step']
        self.num_iter = self.num_step
        self.loss_type =  args['loss_type']
        self.model_name = args['model_name'].lower()
        self.lambda_constrain = args['lambda_constrain']
        self.lambda_back = args['lambda_back']
        self.lm = args['lm']
        self.norm_rel = args['norm_rel']
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
        self.relation_linear2 = nn.Linear(in_features=self.rel_dim, out_features=entity_dim)

        self.kg_lin = nn.Linear(in_features=entity_dim, out_features=entity_dim)
        self.softmax_d1 = nn.Softmax(dim=1)
        self.score_func = nn.Linear(in_features=2*entity_dim, out_features=1)
        # dropout
        self.linear_drop = nn.Dropout(p=self.linear_dropout)

        if self.encode_type:
            self.type_layer = TypeLayer(in_features=entity_dim, out_features=entity_dim,
                                        linear_drop=self.linear_drop, device=self.device, norm_rel=self.norm_rel)

        
        self.self_att_r = AttnEncoder(self.entity_dim)
        self.self_att_r2 = AttnEncoder(self.entity_dim)
        self.kld_loss = nn.KLDivLoss(reduction='none')
        self.kld_loss_1 = nn.KLDivLoss(reduction='none')
        self.bce_loss_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss()
    
    
    def private_module_def(self, args, num_entity, num_relation):
        # initialize entity embedding
        word_dim = self.word_dim
        kg_dim = self.kg_dim
        entity_dim = self.entity_dim
        self.reasoning = NSMLayer(args, num_entity, num_relation, entity_dim)
        self.reasoning2 = NSMLayer(args, num_entity, num_relation, entity_dim)
        if self.lambda_back != 0.0 or self.lambda_constrain != 0.0:
            self.reasoning_back = NSMLayer_back(args, num_entity, num_relation, entity_dim)
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
            rel_features = self.instruction.question_emb(self.rel_features)
            #rel_features = self.relation_linear(rel_features)
            rel_features = self.self_att_r(rel_features,  (self.rel_texts != self.instruction.pad_val).float())
            if self.lm == 'lstm':
                rel_features = self.self_att_r(rel_features, (self.rel_texts != self.num_relation+1).float())
            # else:
            #     rel_features = self.self_att_r(rel_features,  (self.rel_texts != self.instruction.pad_val).float())

        return rel_features

    
    def init_reason(self, curr_dist, local_entity, kb_adj_mat, q_input):
        # batch_size = local_entity.size(0)
        self.local_entity = local_entity
        self.instruction_list, self.attn_list = self.instruction(q_input)
        rel_features = self.get_rel_feature()
        #print(self.rel_features1)
        #self.rel_features2 = self.get_rel_feature2()
        self.local_entity_emb = self.get_ent_init(local_entity, kb_adj_mat, rel_features)
        #self.kge_entity_emb = self.get_ent_init2(local_entity, kb_adj_mat, self.rel_features)
        self.curr_dist = curr_dist
        self.dist_history = []
        self.dist_history2 = []
        self.backward_history = []
        self.action_probs = []
        self.seed_entities = curr_dist


        self.reasoning.init_reason(local_entity=local_entity,
                                   kb_adj_mat=kb_adj_mat,
                                   local_entity_emb=self.local_entity_emb,
                                   rel_features=rel_features)

        if self.lambda_back != 0.0 or self.lambda_constrain != 0.0:
            self.reasoning_back.init_reason(local_entity=local_entity,
                                   kb_adj_mat=kb_adj_mat,
                                   local_entity_emb=self.local_entity_emb,
                                   rel_features=rel_features)
    
    def get_js_div(self, dist_1, dist_2):
        mean_dist = (dist_1 + dist_2) / 2
        log_mean_dist = torch.log(mean_dist + 1e-8)
        # loss_kl_1 = self.kld_loss_1(log_mean_dist, dist_1)
        # loss_kl_2 = self.kld_loss_1(log_mean_dist, dist_2)
        # print(loss_kl_1.item(), loss_kl_2.item())
        loss = 0.5 * (self.kld_loss_1(log_mean_dist, dist_1) + self.kld_loss_1(log_mean_dist, dist_2))
        return loss
    
    def calc_loss_backward(self, case_valid):
        back_loss = None
        constrain_loss = None
        for i in range(self.num_step):
            forward_dist = self.dist_history[i]
            backward_dist = self.backward_history[i]
            if i == 0:
                # back_loss = self.get_loss_new(backward_dist, forward_dist)
                back_loss = self.calc_loss_label(curr_dist=backward_dist,
                                                 teacher_dist=forward_dist,
                                                 label_valid=case_valid)
                # backward last step should be similar with seed distribution
            else:
                tp_loss = self.get_js_div(forward_dist, backward_dist)
                tp_loss = torch.sum(tp_loss * case_valid) / forward_dist.size(0)
                if constrain_loss is None:
                    constrain_loss = tp_loss
                else:
                    constrain_loss += tp_loss
        return back_loss, constrain_loss

    def calc_loss_label(self, curr_dist, teacher_dist, label_valid):
        tp_loss = self.get_loss(pred_dist=curr_dist, answer_dist=teacher_dist, reduction='none')
        tp_loss = tp_loss * label_valid
        cur_loss = torch.sum(tp_loss) / curr_dist.size(0)
        return cur_loss


    def forward(self, batch, training=False):
        local_entity, query_entities, kb_adj_mat, query_text, seed_dist, true_batch_id,  answer_dist = batch
        local_entity = torch.from_numpy(local_entity).type('torch.LongTensor').to(self.device)

        # local_entity_mask = (local_entity != self.num_entity).float()
        query_entities = torch.from_numpy(query_entities).type('torch.FloatTensor').to(self.device)
        answer_dist = torch.from_numpy(answer_dist).type('torch.FloatTensor').to(self.device)
        seed_dist = torch.from_numpy(seed_dist).type('torch.FloatTensor').to(self.device)
        current_dist = Variable(seed_dist, requires_grad=True)

        q_input= torch.from_numpy(query_text).type('torch.LongTensor').to(self.device)
        #ent_texts= torch.from_numpy(ent_texts).type('torch.LongTensor').to(self.device)
        #ent_texts= torch.from_numpy(ent_texts).type('torch.FloatTensor').to(self.device)
        if self.lm != 'lstm':
            pad_val = self.instruction.pad_val #tokenizer.convert_tokens_to_ids(self.instruction.tokenizer.pad_token)
            query_mask = (q_input != pad_val).float()
            
        else:
            query_mask = (q_input != self.num_word).float()

        
        """
        Instruction generations
        """
        self.init_reason(curr_dist=current_dist, local_entity=local_entity,
                         kb_adj_mat=kb_adj_mat,  q_input=q_input)
        self.instruction.init_reason(q_input)
        
        for i in range(self.num_step):
            relational_ins, attn_weight = self.instruction.get_instruction(self.instruction.relational_ins, step=i)
            self.instruction.instructions.append(relational_ins.unsqueeze(1))
            self.instruction.relational_ins = relational_ins
        
        """
        GNN reasoning
        """
        self.curr_dist = current_dist    
        self.dist_history.append(self.curr_dist)
        self.dist_history2.append(self.curr_dist)

        for i in range(self.num_step):
            
            self.curr_dist = self.reasoning(self.curr_dist, self.instruction_list[i], step=i)
            self.dist_history.append(self.curr_dist)

        """
        NSM backward learning (if used)
        """
        if self.lambda_back != 0.0 or self.lambda_constrain != 0.0:
            answer_len = torch.sum(answer_dist, dim=1, keepdim=True)
            answer_len[answer_len == 0] = 1.0
            answer_prob = answer_dist.div(answer_len)
            self.curr_dist_back = answer_prob
            self.backward_history.append(self.curr_dist_back)
            for i in range(self.num_step):
                self.curr_dist_back = self.reasoning_back(self.curr_dist_back, self.instruction_list[self.num_step-i-1], step=i)
                self.backward_history.append(self.curr_dist_back)

        pred_dist = self.dist_history[-1]
        answer_number = torch.sum(answer_dist, dim=1, keepdim=True)
        case_valid = (answer_number > 0).float()
        # filter no answer training case
        # loss = torch.sum(tp_loss * case_valid) / pred_dist.size(0)
        loss =self.calc_loss_label(curr_dist=pred_dist, teacher_dist=answer_dist, label_valid=case_valid)
        
        if self.lambda_back > 0.0 or self.lambda_constrain > 0.0:
             back_loss, constrain_loss = self.calc_loss_backward(case_valid)
             loss = loss + self.lambda_back * back_loss + self.lambda_constrain * constrain_loss
        pred = torch.max(pred_dist, dim=1)[1]
        if training:
            h1, f1 = self.get_eval_metric(pred_dist, answer_dist)
            tp_list = [h1.tolist(), f1.tolist()]
        else:
            tp_list = None
        #return loss, pred, 0.5*(pred_dist+pred_dist2), tp_list
        return loss, pred, pred_dist, tp_list
