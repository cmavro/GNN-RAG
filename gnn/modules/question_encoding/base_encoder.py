import torch
import torch.nn.functional as F
import torch.nn as nn

VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000

class BaseInstruction(torch.nn.Module):

    def __init__(self, args, constraint):
        super(BaseInstruction, self).__init__()
        self.constraint = constraint
        self._parse_args(args)
        self.share_module_def()

    def _parse_args(self, args):
        self.device = torch.device('cuda' if args['use_cuda'] else 'cpu')
        

        # self.share_encoder = args['share_encoder']
        self.q_type = args['q_type']
        if 'num_step' in args:
            self.num_ins = args['num_step']
        elif 'num_ins' in args:
            self.num_ins = args['num_ins']
        elif 'num_layer' in args:
            self.num_ins = args['num_layer']
        elif 'num_expansion_ins' in args and 'num_backup_ins' in args:
            self.num_ins = args['num_backup_ins'] if self.constraint else args['num_expansion_ins']
        else:
            self.num_ins = 1
        
        self.lm_dropout = args['lm_dropout']
        self.linear_dropout = args['linear_dropout']
        self.lm_frozen = args['lm_frozen']

        for k, v in args.items():
            if k.endswith('dim'):
                setattr(self, k, v)
            if k.endswith('emb_file') or k.endswith('kge_file'):
                if v is None:
                    setattr(self, k, None)
                else:
                    setattr(self, k, args['data_folder'] + v)

        self.reset_time = 0

    def share_module_def(self):
        # dropout
        self.lstm_drop = nn.Dropout(p=self.lm_dropout)
        self.linear_drop = nn.Dropout(p=self.linear_dropout)

    def init_hidden(self, num_layer, batch_size, hidden_size):
        return (torch.zeros(num_layer, batch_size, hidden_size).to(self.device),
                torch.zeros(num_layer, batch_size, hidden_size).to(self.device))

    def encode_question(self, *args):
        # constituency tree or query_text
        pass

    @staticmethod
    def get_node_emb(query_hidden_emb, action):
        '''

        :param query_hidden_emb: (batch_size, max_hyper, emb)
        :param action: (batch_size)
        :return: (batch_size, 1, emb)
        '''
        batch_size, max_hyper, _ = query_hidden_emb.size()
        row_idx = torch.arange(0, batch_size).type(torch.LongTensor)
        q_rep = query_hidden_emb[row_idx, action, :]
        return q_rep.unsqueeze(1)

    def init_reason(self, query_text):
        self.batch_size = query_text.size(0)
        self.max_query_word = query_text.size(1)
        self.encode_question(query_text)
        self.relational_ins = torch.zeros(self.batch_size, self.entity_dim).to(self.device)
        self.instructions = []
        self.attn_list = []

    def get_instruction(self, relational_ins, step=0, query_node_emb=None):
        
        query_hidden_emb = self.query_hidden_emb
        
        query_mask = self.query_mask
        if query_node_emb is None:
            query_node_emb = self.query_node_emb
        
        relational_ins = relational_ins.unsqueeze(1)
        question_linear = getattr(self, 'question_linear' + str(step))
        q_i = question_linear(self.linear_drop(query_node_emb))
        cq = self.cq_linear(self.linear_drop(torch.cat((relational_ins, q_i, q_i-relational_ins,q_i*relational_ins), dim=-1)))
        # batch_size, 1, entity_dim
        ca = self.ca_linear(self.linear_drop(cq * query_hidden_emb))
        # batch_size, max_local_entity, 1
        # cv = self.softmax_d1(ca + (1 - query_mask.unsqueeze(2)) * VERY_NEG_NUMBER)
        attn_weight = F.softmax(ca + (1 - query_mask.unsqueeze(2)) * VERY_NEG_NUMBER, dim=1)
        # batch_size, max_local_entity, 1
        relational_ins = torch.sum(attn_weight * query_hidden_emb, dim=1)
        return relational_ins, attn_weight
        


    def forward(self, query_text, lm=None):
        if lm is not None:
            self.node_encoder = lm
        self.init_reason(query_text)
        for i in range(self.num_ins):
            relational_ins, attn_weight = self.get_instruction(self.relational_ins, step=i)
            self.instructions.append(relational_ins)
            self.attn_list.append(attn_weight)
            self.relational_ins = relational_ins
        return self.instructions, self.attn_list

