# coding: utf-8
# @Author: Qin WT
# @Time: 2019/7/7 19:58

import math

PAD_ID = 0
UNK_ID = 1
SOS_ID = 2
EOS_ID = 3

class MyBatch:
    def __init__(self, pairs, batch_size, token2idx):
        self.pairs = []
        for pair in pairs:
            s, t = pair[0], pair[1]
            _s = [token2idx.get(w, UNK_ID) for w in s] + [EOS_ID]
            _t = [SOS_ID] + [token2idx.get(w, UNK_ID) for w in t] + [EOS_ID]
            self.pairs.append((_s, _t))
        self.num_samples = len(self.pairs)
        self.batch_size = batch_size
        assert self.num_samples >= self.batch_size, 'warning: num_samples < batch_size'
        self.num_batch = math.ceil(self.num_samples/self.batch_size)
        self.cur_batch = 0

    def get_next(self):
        left = self.cur_batch * self.batch_size
        right = (self.cur_batch + 1) * self.batch_size
        if right > self.num_samples:
            ret = self.pairs[left:] + self.pairs[:right-self.num_samples]
            self.cur_batch = 0
        else:
            ret = self.pairs[left:right]
            self.cur_batch += 1
        return ret

    def process_batch(self, batch):
        x_seqlens = []
        y_seqlens = []
        x = []
        y = []
        decoder_inputs = []
        for s, t in batch:
            x_seqlens.append(len(s))
            y_seqlens.append((len(t)-1))
        max_x_len = max(x_seqlens)
        max_y_len = max(y_seqlens)
        for s, t in batch:
            x.append(s + (max_x_len - len(s))*[PAD_ID])
            y.append(t[1:] + (max_y_len - len(t) + 1)*[PAD_ID])
            decoder_inputs.append(t[:-1] + (max_y_len - len(t) + 1)*[PAD_ID])
        return x, x_seqlens, y, decoder_inputs, y_seqlens


def load_vocab(vocab_fpath):
    '''Loads vocabulary file and returns idx<->token maps
    vocab_fpath: string. vocabulary file path.
    Note that these are reserved
    0: <pad>, 1: <unk>, 2: <s>, 3: </s>

    Returns
    two dictionaries.
    '''
    vocab = [line.split()[0] for line in open(vocab_fpath, 'r').read().splitlines()]
    token2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2token = {idx: token for idx, token in enumerate(vocab)}
    return token2idx, idx2token
