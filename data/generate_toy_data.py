# coding: utf-8
# @Author: Qin WT
# @Time: 2019/7/6 11:52

vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
         'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

import random

def a_sample():
    size = random.randint(10, 20)
    s = []
    t = []
    for _ in range(size):
        idx = random.randint(0, 25)
        s.append(vocab[idx])
        t.append(vocab[(idx+3)%26])
    return ' '.join(s), ' '.join(t)


def to_file(file_name, data_n):
    with open(file_name, 'w', encoding='utf-8') as fw:
        for _ in range(data_n):
            s, t = a_sample()
            fw.write(s + '\t' + t + '\n')


# train data
to_file('train', 1000)
# eval data
to_file('eval', 100)
# test data
to_file('test', 100)

