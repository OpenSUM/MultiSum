import collections
import copy
import json
import re
import sys

import six
import yaml
import tensorflow as tf
from rouge import Rouge

from utils.word2char import word2char
from sklearn.metrics import matthews_corrcoef, f1_score, balanced_accuracy_score

rouger = Rouge()


class Config():

    def __init__(self, data=None):
        if data:
            for k, v in data.items():
                self.set(k, v)

    def set(self, k, v):
        if k != '_dict':
            setattr(self, k, v)

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        self.set(key, value)

    @property
    def _dict(self):
        dct = dict()
        for k in dir(self):
            if k != 'set' and not k.startswith('_'):
                dct[k] = self[k]
        return dct


def eprint(*args, **kwargs):
    print(file=sys.stderr, *args, **kwargs)


def load_config(filepath='lcsts.yaml'):
    with open(filepath, 'r', encoding='utf8') as f:
        x = yaml.load(f)
    return Config(x)


def id2text(ids, id2word):
    return ' '.join([id2word[j] for j in ids if j > 0])


def ids2text(ids, id2word):
    """
    transform id to text. the id2word should be the extended vocab.
    :param ids:
    :param id2word:
    :return:
    """
    texts = []
    for i, v in zip(ids, id2word):
        texts.append(id2text(i, v))
    return texts


def calc_rouge(cands, refs, placeholder=None):
    """
    calc rouge scores
    :param cands: list of texts, each of them is split by blanks. the "word" in each text can be a word or its id.
                    The [PAD]/0 should be removed.
    :param refs: the same as cands
    :return:
    """
    if placeholder:
        cands = [c.replace(placeholder + ' ', '').replace(' ' + placeholder, '') for c in cands]
        refs = [r.replace(placeholder + ' ', '').replace(' ' + placeholder, '') for r in refs]
    cands = [word2char(text, remove_tags=True, keep_least_one_tag_for_rouge=True) for text in cands]
    refs = [word2char(text, remove_tags=True, keep_least_one_tag_for_rouge=True) for text in refs]
    s = rouger.get_scores(cands, refs, avg=True)
    return s




def calc_acc(trues, preds):
    assert len(trues) == len(preds)
    length = len(trues)
    temp_trues = []
    temp_preds = []
    for i in range(length):
        for j in range(len(trues[i])):
            temp_trues.append(trues[i][j])
    for i in range(length):
        for j in range(len(preds[i])):
            temp_preds.append(preds[i][j])
    acc = balanced_accuracy_score(y_true=temp_trues,y_pred=temp_preds)
    f1 = f1_score(y_true=temp_trues, y_pred=temp_preds, average='macro')
    return acc,f1


class BertConfig(object):
    """Configuration for `BertModel`."""

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02):
        """Constructs BertConfig.

        Args:
          vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
          hidden_size: Size of the encoder layers and the pooler layer.
          num_hidden_layers: Number of hidden layers in the Transformer encoder.
          num_attention_heads: Number of attention heads for each attention layer in
            the Transformer encoder.
          intermediate_size: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
          hidden_act: The non-linear activation function (function or string) in the
            encoder and pooler.
          hidden_dropout_prob: The dropout probability for all fully connected
            layers in the embeddings, encoder, and pooler.
          attention_probs_dropout_prob: The dropout ratio for the attention
            probabilities.
          max_position_embeddings: The maximum sequence length that this model might
            ever be used with. Typically set this to something large just in case
            (e.g., 512 or 1024 or 2048).
          type_vocab_size: The vocabulary size of the `token_type_ids` passed into
            `BertModel`.
          initializer_range: The stdev of the truncated_normal_initializer for
            initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name_to_variable[name]
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
      tensor: A tf.Tensor to check the rank of.
      expected_rank: Python integer or list of integers, expected rank.
      name: Optional name of the tensor for the error message.

    Raises:
      ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.

    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)
    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape

