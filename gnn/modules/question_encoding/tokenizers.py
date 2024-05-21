import re
import numpy as np
from transformers import BertTokenizer

class LSTMTokenizer():
    def __init__(self, word2id, max_query_word):
        super(LSTMTokenizer, self).__init__()
        self.word2id = word2id
        self.max_query_word = max_query_word

    def tokenize(self, question):
        tokens = self.tokenize_sent(question)
        query_text = np.full(self.max_query_word, len(self.word2id), dtype=int)
        #tokens = question.split()
        #if self.data_type == "train":
        #    random.shuffle(tokens)
        for j, word in enumerate(tokens):
            if j < self.max_query_word:
                    if word in self.word2id:
                        query_text[j] = self.word2id[word]
                        
            else:
                query_text[j] = len(self.word2id)

        return query_text

    @staticmethod
    def tokenize_sent(question_text):
        question_text = question_text.strip().lower()
        question_text = re.sub('\'s', ' s', question_text)
        words = []
        toks = enumerate(question_text.split(' '))
        
        for w_idx, w in toks:
            w = re.sub('^[^a-z0-9]|[^a-z0-9]$', '', w)
            if w == '':
                continue
            words += [w]
        return words

class BERTTokenizer():
    def __init__(self, max_query_word):
        super(BERTTokenizer, self).__init__()
        self.q_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_query_word = max_query_word
        self.num_word = self.q_tokenizer.encode("[UNK]")[0] #len(self.q_tokenizer.vocab.keys())

        
    
    def tokenize(self, question):
        query_text = np.full(self.max_query_word, 0, dtype=int)
        tokens =  self.q_tokenizer.encode_plus(text=question, max_length=self.max_query_word, \
                    pad_to_max_length=True, return_attention_mask = False, truncation=True)
        return np.array(tokens['input_ids'])