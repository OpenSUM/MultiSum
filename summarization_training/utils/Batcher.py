import numpy as np

from utils import BertConfig
from .data_loader import load_src, load_dst, load_vocab, load_senti


class Batcher:
    def __init__(self, y_token, y_ids, y_ids_ext_sep, y_ids_extend, y_mask, x_token, x_ids, x_ids_extend,
                 x_mask, oov_size, oovs, batch_size, sentiment):
        """
        :param y_token: target token
        :param y_ids: target token id (used for input of decoder, the oov words will be replaced with UNK token)
        :param y_ids_ext_sep: target extended token id with sep (used for calc loss)
        :param y_ids_extend: target extended token id (contains the id of oov words)
        :param y_mask: target sequence padding mask
        :param x_token: source token
        :param x_ids: source token id (used for input of encoder, the oov words will be replaced with UNK token)
        :param x_ids_extend: source extended token id (contains the id of oov words)
        :param x_mask: source sequence padding mask
        :param oov_size: the number of oov words
        :param oovs: the oov words
        :param batch_size: the hyper-parameter of batch_size
        """
        assert y_token.shape[0] == y_ids.shape[0] == y_mask.shape[0] == x_ids.shape[0] == \
               x_mask.shape[0] == oov_size.shape[0] == oovs.shape[0] == x_token.shape[0]

        np.random.seed(7890)
        self.indices = np.random.permutation(range(y_ids.shape[0]))
        self.iterations = (y_ids.shape[0] - 1) / batch_size + 1
        self.total_samples = y_ids.shape[0]

        self.y_token = y_token[self.indices]
        self.y_ids = y_ids[self.indices]
        self.y_ids_ext_sep = y_ids_ext_sep[self.indices]
        self.y_ids_extend = y_ids_extend[self.indices]
        self.y_mask = y_mask[self.indices]
        self.x_token = x_token[self.indices]
        self.x_ids = x_ids[self.indices]
        self.x_ids_extend = x_ids_extend[self.indices]
        self.x_mask = x_mask[self.indices]
        self.oov_size = oov_size[self.indices]
        self.oovs = oovs[self.indices]
        self.batch_size = batch_size
        self.sentiment = sentiment[self.indices]

    def shuffle(self, seed=None):
        if seed:
            np.random.seed(seed)
        self.indices = np.random.permutation(range(self.y_ids.shape[0]))

        self.y_token = self.y_token[self.indices]
        self.y_ids = self.y_ids[self.indices]
        self.y_ids_ext_sep = self.y_ids_ext_sep[self.indices]
        self.y_ids_extend = self.y_ids_extend[self.indices]
        self.y_mask = self.y_mask[self.indices]
        self.x_token = self.x_token[self.indices]
        self.x_ids = self.x_ids[self.indices]
        self.x_ids_extend = self.x_ids_extend[self.indices]
        self.x_mask = self.x_mask[self.indices]
        self.oov_size = self.oov_size[self.indices]
        self.oovs = self.oovs[self.indices]
        self.sentiment = self.sentiment[self.indices]

    def batch(self, batch_size=None):
        self.shuffle()
        i = 0
        if batch_size is None:
            batch_size = self.batch_size
        size = self.y_ids.shape[0]
        while batch_size * (i + 1) < size:
            yield [self.y_token[i * batch_size:(i + 1) * batch_size],
                   self.y_ids[i * batch_size:(i + 1) * batch_size],
                   self.y_ids_ext_sep[i * batch_size:(i + 1) * batch_size],
                   self.y_ids_extend[i * batch_size:(i + 1) * batch_size],
                   self.y_mask[i * batch_size:(i + 1) * batch_size],
                   self.x_token[i * batch_size:(i + 1) * batch_size],
                   self.x_ids[i * batch_size:(i + 1) * batch_size],
                   self.x_ids_extend[i * batch_size:(i + 1) * batch_size],
                   self.x_mask[i * batch_size:(i + 1) * batch_size],
                   self.oov_size[i * batch_size:(i + 1) * batch_size],
                   self.oovs[i * batch_size:(i + 1) * batch_size],
                   self.sentiment[i * batch_size:(i + 1) * batch_size], ]
            i += 1
        yield [self.y_token[i * batch_size:],
               self.y_ids[i * batch_size:],
               self.y_ids_ext_sep[i * batch_size:],
               self.y_ids_extend[i * batch_size:],
               self.y_mask[i * batch_size:],
               self.x_token[i * batch_size:],
               self.x_ids[i * batch_size:],
               self.x_ids_extend[i * batch_size:],
               self.x_mask[i * batch_size:],
               self.oov_size[i * batch_size:],
               self.oovs[i * batch_size:],
               self.sentiment[i * batch_size:], ]

def get_batcher(src_file, dst_file, senti_file, word2id, src_seq_length, dst_seq_length, batch_size, do_lower, substr_prefix='Ġ',
                limit=None):
    """
    :param src_file: path of the source file
    :param dst_file: path of the target file
    :param word2id: vocab object
    :param src_seq_length: max length of source sequence
    :param dst_seq_length: max length of target sequence
    :param batch_size: batch size
    :param do_lower: lower the text if True
    :param substr_prefix: the separator between subwords
    :param limit: max samples load from the data file
    :return: a Batcher object
    """
    src_tokens, src_ids, src_ids_extend, src_mask, src_oov_size, src_oovs = load_src(src_file=src_file,
                                                                                     seq_length=src_seq_length,
                                                                                     do_lower=do_lower,
                                                                                     vocab=word2id,
                                                                                     substr_prefix=substr_prefix,
                                                                                     limit=limit)
    dst_tokens, dst_ids, dst_ids_extend, dst_ids_ext_sep, dst_mask = load_dst(dst_file=dst_file,
                                                                              seq_length=dst_seq_length,
                                                                              do_lower=do_lower,
                                                                              vocab=word2id,
                                                                              src_oovs=src_oovs,
                                                                              src_ids=src_ids,
                                                                              substr_prefix=substr_prefix,
                                                                              limit=limit)
    sentiments = load_senti(senti_file=senti_file)
    print('Example tokens:')
    for i in range(min(1, len(src_tokens), len(dst_tokens), len(sentiments))):
        print('src: {}\ndst: {}\nsentiment: {}\n'.format(' '.join(src_tokens[i]), ' '.join(dst_tokens[i]), ' '.join(str(sentiments[i]))))
        print('src_ids:',src_ids[i])
        print('dst_ids:',dst_ids[i])

    batcher = Batcher(y_token=dst_tokens,
                      y_ids=dst_ids,
                      y_ids_ext_sep=dst_ids_ext_sep,
                      y_ids_extend=dst_ids_extend,
                      y_mask=dst_mask,
                      x_token=src_tokens,
                      x_ids=src_ids,
                      x_ids_extend=src_ids_extend,
                      x_mask=src_mask,
                      oov_size=src_oov_size,
                      oovs=src_oovs,
                      batch_size=batch_size,
                      sentiment=sentiments)
    return batcher
