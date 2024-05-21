

import torch.nn as nn
from utils import get_dict
from .base_encoder import BaseInstruction

VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000

class LSTMInstruction(BaseInstruction):

    def __init__(self, args, word_embedding, num_word):
        super(LSTMInstruction, self).__init__(args)
        self.word2id = get_dict(args['data_folder'],args['word2id'])

        self.word_embedding = word_embedding
        self.num_word = num_word
        self.encoder_def()
        entity_dim = self.entity_dim
        self.cq_linear = nn.Linear(in_features=4 * entity_dim, out_features=entity_dim)
        self.ca_linear = nn.Linear(in_features=entity_dim, out_features=1)
        for i in range(self.num_ins):
            self.add_module('question_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))

    def encoder_def(self):
        # initialize entity embedding
        word_dim = self.word_dim
        entity_dim = self.entity_dim
        self.node_encoder = nn.LSTM(input_size=word_dim, hidden_size=entity_dim,
                                    batch_first=True, bidirectional=False)

    def encode_question(self, query_text, store=True):
        batch_size = query_text.size(0)
        query_word_emb = self.word_embedding(query_text)  # batch_size, max_query_word, word_dim
        query_hidden_emb, (h_n, c_n) = self.node_encoder(self.lstm_drop(query_word_emb),
                                                         self.init_hidden(1, batch_size,
                                                                          self.entity_dim))  # 1, batch_size, entity_dim
        if store:
            self.instruction_hidden = h_n
            self.instruction_mem = c_n
            self.query_node_emb = h_n.squeeze(dim=0).unsqueeze(dim=1)  # batch_size, 1, entity_dim
            self.query_hidden_emb = query_hidden_emb
            self.query_mask = (query_text != self.num_word).float()
            return query_hidden_emb, self.query_node_emb
        else:
            return query_hidden_emb
    

    

