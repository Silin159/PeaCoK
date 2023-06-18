import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import json
from collections import OrderedDict
from nltk.tokenize import word_tokenize

import numpy as np


class LSTMBinaryClassifier(nn.Module):

    def __init__(self, args):
        super(LSTMBinaryClassifier, self).__init__()

        self.embed_dim = args.embed_dim
        self.hidden_dim = args.hidden_dim
        self.forward_dim = args.forward_dim
        self.lstm_layers = args.lstm_layers
        self.vocab_size = args.vocab_size
        self.dropout_rate = args.dropout
        self.device = args.device

        self.dropout = nn.Dropout(self.dropout_rate)
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_dim, num_layers=self.lstm_layers,
                            batch_first=True, bidirectional=True, dropout=self.dropout_rate)
        self.fc1 = nn.Linear(4 * self.hidden_dim, self.forward_dim)
        self.fc2 = nn.Linear(self.forward_dim, 2)

    def forward(self, input_ids, labels=None):
        h = torch.zeros((2 * self.lstm_layers, input_ids.size(0), self.hidden_dim)).to(self.device)
        c = torch.zeros((2 * self.lstm_layers, input_ids.size(0), self.hidden_dim)).to(self.device)

        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)

        out = self.embedding(input_ids)
        out, _ = self.lstm(out, (h, c))
        out = self.dropout(out)
        out = torch.cat([out[:, 0, :], out[:, -1, :]], -1)
        out = torch.relu_(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        logits = out.view(-1, out.size(-1))
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels.view(-1))
        outputs = (loss, logits)

        return outputs

    def load_glove_embedding(self, args, tokenizer):
        initial_arr = self.embedding.weight.data.cpu().numpy()
        emb = torch.from_numpy(self.get_glove_matrix(args.glove_path, tokenizer, initial_arr))
        self.embedding.weight.data.copy_(emb)

    def get_glove_matrix(self, glove_path, tokenizer, initial_embedding_np):
        vec_array = initial_embedding_np
        vec_array = vec_array.astype(np.float32)

        with open(glove_path, 'r', encoding='UTF-8') as ef:
            for line in ef.readlines():
                line = line.strip().split(' ')
                word, vec = line[0], line[1:]
                vec = np.array(vec, np.float32)
                if not tokenizer.has_word(word):
                    pass
                else:
                    word_idx = tokenizer.encode(word)
                    vec_array[word_idx] = vec

        return vec_array


class Tokenizer(object):
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self._idx2word = {}
        self._word2idx = {}
        self._freq_dict = {}
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        for w in [self.pad_token, self.unk_token, "<bos>", "<n_sep>", "<f_sep>", "<eon>", "<eos>"]:
            self._absolute_add_word(w)

    def _absolute_add_word(self, w):
        idx = len(self._idx2word)
        self._idx2word[idx] = w
        self._word2idx[w] = idx

    def add_word(self, word):
        if word not in self._freq_dict:
            self._freq_dict[word] = 0
        self._freq_dict[word] += 1

    def add_word_sentence(self, sentence):
        for w in word_tokenize(sentence):
            self.add_word(w)

    def has_word(self, word):
        return (word in self._word2idx) and (self._word2idx[word] < self.vocab_size)

    def _add_to_vocab(self, word):
        if word not in self._word2idx:
            idx = len(self._idx2word)
            self._idx2word[idx] = word
            self._word2idx[word] = idx

    def construct(self):
        # print('Constructing Vocab...')
        freq_sort = sorted(self._freq_dict.keys(), key=lambda x: -self._freq_dict[x])
        if len(freq_sort) + len(self._idx2word) < self.vocab_size:
            print('Actual vocab size smaller than that configured: {}/{}'
                  .format(len(freq_sort) + len(self._idx2word), self.vocab_size))
        for word in freq_sort:
            self._add_to_vocab(word)
        # print('Vocab Constructed.')

    def load_vocab(self, vocab_path):
        self._freq_dict = json.loads(open(vocab_path + 'freq.json', 'r').read())
        self._word2idx = json.loads(open(vocab_path + 'word2idx.json', 'r').read())
        self._idx2word = {}
        for w, idx in self._word2idx.items():
            self._idx2word[idx] = w
        # print('Vocab file loaded.')

    def save_vocab(self, vocab_path):
        freq_dict = OrderedDict(sorted(self._freq_dict.items(), key=lambda kv:kv[1], reverse=True))
        with open(vocab_path + 'word2idx.json', 'w') as f:
            json.dump(self._word2idx, f, indent=2)
        with open(vocab_path + 'freq.json', 'w') as f:
            json.dump(freq_dict, f, indent=2)
        # print('Vocab file saved.')

    def encode(self, word):
        word = self.unk_token if word not in self._word2idx else word
        idx = self._word2idx[word]
        return self._word2idx[self.unk_token] if idx >= self.vocab_size else idx

    def sentence_encode(self, sentence):
        return [self.encode(w) for w in word_tokenize(sentence)]

    def decode(self, idx):
        if idx >= self.vocab_size or idx not in self._idx2word:
            return self.unk_token
        else:
            return self._idx2word[idx]

    def sentence_decode(self, index_list, eos=None):
        tokens = [self.decode(idx) for idx in index_list]
        if not eos or eos not in tokens:
            return ' '.join(tokens)
        else:
            idx = tokens.index(eos)
            return ' '.join(tokens[:idx])
