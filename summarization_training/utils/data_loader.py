import re
import time

import numpy as np

from constants import SAMPLE_LIMIT, PAD_TOKEN
from constants import SEP_TOKEN, UNK_TOKEN, CLS_TOKEN



def load_vocab(vocab_file, do_lower=False):
    word2id, id2word = dict(), dict()
    with open(vocab_file, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            w = line.strip('\n')          
            word2id[w] = i
            id2word[i] = w
    print(len(word2id))
    assert len(word2id) == len(id2word)
    for i in range(len(word2id)):
        assert word2id[id2word[i]] == i
    return word2id, id2word




def load_src(src_file, seq_length, do_lower, vocab=None, substr_prefix='Ġ', limit=SAMPLE_LIMIT):
    print('Loading src file: {}'.format(src_file))

    tokenize = lambda x: x.strip().lstrip('<s>').lstrip(CLS_TOKEN).rstrip('</s>').rstrip(
            SEP_TOKEN).strip().split(' ')

    with open(src_file, 'r', encoding='utf8') as f:
        tokens = []
        mask = []
        ids = []
        ids_extend = []
        oovs = []
        oov_size = []
        for i, l in enumerate(f):
            if limit is not None and i >= limit:
                break

            l = tokenize(l)
            oov = []
            tmp_token = [CLS_TOKEN]
            tmp_extend = [vocab[CLS_TOKEN]]
            tmp = [vocab[CLS_TOKEN]]
            for w in l[:seq_length - 2]:
                tmp_token.append(w)
                if w in vocab:
                    tmp.append(vocab[w])
                    tmp_extend.append(vocab[w])
                elif w in oov:
                    tmp.append(vocab[UNK_TOKEN])
                    tmp_extend.append(len(vocab) + oov.index(w))
                else:
                    oov.append(w)
                    tmp.append(vocab[UNK_TOKEN])
                    tmp_extend.append(len(vocab) + oov.index(w))
            tmp_token.append(SEP_TOKEN)
            tmp_extend.append(vocab[SEP_TOKEN])
            tmp.append(vocab[SEP_TOKEN])
            mask.append(([1] * len(tmp_extend) + [0] * (seq_length - len(tmp_extend)))[:seq_length])

            tmp_extend += [vocab[PAD_TOKEN]] * (seq_length - len(tmp_extend))
            tmp += [vocab[PAD_TOKEN]] * (seq_length - len(tmp))

            tokens.append(tmp_token)
            ids_extend.append(tmp_extend[:seq_length])
            ids.append(tmp[:seq_length])
            oovs.append(oov)
            oov_size.append(len(oov))

            if i % 1000 == 0:
                print('\r{}/{}'.format(i, limit), end='')
    print('\r{}/{}'.format(i + 1, i + 1))
    return np.array(tokens), np.array(ids), np.array(ids_extend), np.array(mask), np.array(oov_size), np.array(oovs)


def load_dst(dst_file, seq_length, do_lower, vocab, src_oovs=None, src_ids=None, substr_prefix='Ġ', limit=SAMPLE_LIMIT):
    print('Loading dst file: {}'.format(dst_file))
    assert vocab is not None
    if src_oovs is not None or src_ids is not None:
        assert src_oovs is not None
        assert src_ids is not None
        assert len(src_oovs) == len(src_ids)

    vocab_size = len(vocab)


    tokenize = lambda x: x.strip().lstrip('<s>').lstrip(CLS_TOKEN).rstrip('</s>').rstrip(
            SEP_TOKEN).strip().split(' ')

    with open(dst_file, 'r', encoding='utf8') as f:
        tokens = []
        mask = []
        ids = []
        ids_extend = []
        ids_ext_sep = []
        for i, l in enumerate(f):
            if limit is not None and i >= limit:
                continue

            l = tokenize(l)
            tmp_token = []
            tmp_extend = []
            tmp_id = []
            for w in l[:seq_length - 1]:
                if w != UNK_TOKEN and w in vocab:
                    tmp_extend.append(vocab[w])
                elif w in src_oovs[i]:
                    tmp_extend.append(vocab_size + src_oovs[i].index(w))
                else:
                    tmp_extend.append(vocab[UNK_TOKEN])
                tmp_id.append(vocab[w] if w in vocab else vocab[UNK_TOKEN])
                tmp_token.append(w)

            tmp_token.append(SEP_TOKEN)
            tokens.append(tmp_token)
            ids_ext_sep.append(
                tmp_extend + [vocab[SEP_TOKEN]] + [vocab[PAD_TOKEN]] * (seq_length - len(tmp_extend) - 1))

            mask.append(([1] * (len(tmp_extend) + 1) + [0] * (seq_length - len(tmp_extend) - 1))[:seq_length])

            tmp_extend += [vocab[PAD_TOKEN]] * (seq_length - len(tmp_extend))
            ids_extend.append(tmp_extend[:seq_length])

            tmp_id += [vocab[PAD_TOKEN]] * (seq_length - len(tmp_id))
            ids.append(tmp_id[:seq_length])

        if i % 1000 == 0:
            print('\r{}/{}'.format(i, limit), end='')
    print('\r{}/{}'.format(i + 1, i + 1))
    return np.array(tokens), np.array(ids), np.array(ids_extend), np.array(ids_ext_sep), np.array(mask)

def load_senti(senti_file):
    print('Loading sentiment file: {}'.format(senti_file))
    with open(senti_file, 'r', encoding='utf-8') as f:
        overalls = []
        for line in f:
            line=line.replace(" ", "")
            overalls.append(int(float(line)))

    return np.array(overalls)
    


def id2text(ids, id2word, oov, vocab_size=None):
    if vocab_size is None:
        vocab_size = len(id2word)
    text = []
    if type(oov) != list:
        oov = oov.tolist()
    for i in ids:
        if i in id2word:
            text.append(id2word[i])
        elif (i-vocab_size)>=0 and (i-vocab_size)<=len(oov)-1:
            text.append(oov[i - vocab_size])
        else:
            text.append(id2word[word2id[UNK_TOKEN]])
    return ' '.join(text)


def ids2text(ids, id2word, oovs):
    vocab_size = len(id2word)
    texts = []
    for id_, oov in zip(ids, oovs):
        texts.append(id2text(ids=id_, id2word=id2word, oov=oov, vocab_size=vocab_size))
    return texts
