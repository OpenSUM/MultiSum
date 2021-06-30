import numpy as np
import tensorflow as tf

from utils import get_shape_list

_pad_mask_dict = {}


def get_pad_mask(relative_size, seq_length, heads):
    key = (relative_size, seq_length)
    if key in _pad_mask_dict:
        pad_mask = _pad_mask_dict[key]
    else:
        pad_mask = [[1] * max(relative_size - i, 0) + [0] * (
                2 * relative_size + 1 - max(relative_size - i, 0) - max(relative_size + i - seq_length, 0)) + [1] * max(
            relative_size + i - seq_length, 0) for i in range(seq_length)]
        pad_mask = tf.constant(pad_mask, dtype=tf.float32)
        pad_mask = tf.tile(tf.expand_dims(pad_mask, axis=0), [heads, 1, 1])
        pad_mask = tf.expand_dims(pad_mask, axis=0)
        _pad_mask_dict[key] = pad_mask
    return pad_mask


def trim_relative_attention(q, k, v, relative_size, mask_matrix, heads=12, session=None, **kwargs):
    """
    This is an implementation of trim relative attention v2.
    It should be used in encoder.
    This function doesn't support masked self attention now.
    If you want to implement a masked trim self attention,
    you can modify the get_pad_mask function.
    :param q: shape = [batch_size, seq_length, hidden_dim]
    :param k: shape = [batch_size, seq_length, hidden_dim]
    :param v: shape = [batch_size, seq_length, hidden_dim]
    :param relative_size:
    :param mask_matrix: shape = [batch_size, seq_length]
    :param heads: num of heads.
    :param session: if session is not None, calc and print the value of the nodes using the session.
    :return:
    """
    batch_size, seq_length, hidden_dim = get_shape_list(q)
    dim_per_head = hidden_dim // heads

    if mask_matrix is not None and mask_matrix.dtype != tf.float32:
        mask_matrix = tf.cast(mask_matrix, dtype=tf.float32)

    q = tf.reshape(q, shape=(batch_size, seq_length, heads, dim_per_head))
    q = tf.transpose(q, [0, 2, 1, 3])
    k = tf.reshape(k, shape=(batch_size, seq_length, heads, dim_per_head))
    k = tf.transpose(k, [0, 2, 1, 3])
    v = tf.reshape(v, shape=(batch_size, seq_length, heads, dim_per_head))
    v = tf.transpose(v, [0, 2, 1, 3])

    # get padding k
    # q/k/v.shape = [batch_size, heads, seq_length, dim_per_head]
    pad_for_k = tf.zeros(shape=(relative_size, batch_size, heads, dim_per_head))
    transpose_k = tf.transpose(k, [2, 0, 1, 3])  # [seq_length, batch_size, heads, hidden_dim]
    pad_k = tf.concat([pad_for_k, transpose_k, pad_for_k], axis=0)
    if session:
        print('pad_k:{}'.format(session.run(pad_k)))

    # get padding v
    transpose_v = tf.transpose(v, [2, 0, 1, 3])  # [seq_length, batch_size, heads, hidden_dim]
    pad_v = tf.concat([pad_for_k, transpose_v, pad_for_k], axis=0)
    if session:
        print('pad_v:{}'.format(session.run(pad_v)))

    # gather k
    indices = [list(range(i, 2 * relative_size + 1 + i)) for i in range(seq_length)]
    gather_k = tf.gather(pad_k, indices)  # [seq_length, 2c+1, batch_size, heads, dim_per_head]
    gather_v = tf.gather(pad_v, indices)  # [seq_length, 2c+1, batch_size, heads, dim_per_head]
    score = tf.einsum('jkhip,hijp->hijk', gather_k, q)  # [batch_size, heads, seq_length, 2c+1]
    if session:
        print('indices:{}'.format(indices))
        print('gather_k:{}'.format(session.run(gather_k)))
        print('score:{}'.format(session.run(score)))

    # get padding mask and gather it
    if mask_matrix is not None:
        transpose_mask = tf.transpose(mask_matrix, [1, 0])
        pad_for_mask = tf.zeros(shape=(relative_size, batch_size), dtype=tf.float32)
        pad_mask = tf.concat([pad_for_mask, transpose_mask, pad_for_mask], axis=0)
        gather_mask = tf.gather(pad_mask, indices=indices)
        final_mask = 1.0 - tf.expand_dims(tf.transpose(gather_mask, [2, 0, 1]), 1)
        if session:
            print('final_mask:{}'.format(session.run(final_mask)))
    else:
        # pad_mask is useless, because input_mask is more strict than pad mask.
        pad_mask = get_pad_mask(relative_size=relative_size, seq_length=seq_length, heads=heads)
        final_mask = pad_mask

    # calculate attention and context
    masked_score = score + final_mask * (-1e6)
    if session:
        print('masked_score:{}'.format(session.run(masked_score)))

    proba = tf.nn.softmax(masked_score, axis=-1)
    if session:
        print('proba:{}'.format(session.run(proba)))
    context = tf.einsum('jkhip,hijk->hjip', gather_v, proba)
    context = tf.reshape(context, shape=[batch_size, seq_length, hidden_dim])
    if session:
        print('context:{}'.format(session.run(context)))
    return context


def test_my_self_relative_attention_v2():
    np.random.seed(233)
    tf.set_random_seed(233)
    s = tf.Session()

    batch_size = 3
    seq_length = 7
    hidden_dim = 2
    heads = 1
    relative_size = 1
    a = np.random.random((batch_size, seq_length, hidden_dim))
    a = tf.constant(a, dtype=tf.float32)
    mask = tf.constant(np.concatenate((np.ones((batch_size, 4)), np.zeros((batch_size, 3))), axis=-1), dtype=np.float32)
    print(s.run(mask))
    ct = trim_relative_attention(a, a, a, heads=heads, mask_matrix=mask, relative_size=relative_size, session=s)
    print(ct.shape)
    print(s.run(tf.reduce_sum(ct)))


def test():
    test_my_self_relative_attention_v2()


if __name__ == '__main__':
    test()
