import multiprocessing
import time

import numpy as np

PAD = 0
n_process = 2  
pool = None


def _lcs(x, y):
    n = len(x)
    m = len(y)
    table = dict()

    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])

    return table[n, m], n, m


def s_rouge_l(eva, ref):
    same_len, eva_len, ref_len = map(float, _lcs(eva, ref[np.where(ref > PAD)]))

    r_lcs, p_lcs = same_len / ref_len, same_len / eva_len
    beta = p_lcs / (r_lcs + 1e-12)

    f_lcs = ((1 + (beta ** 2)) * r_lcs * p_lcs) / (r_lcs + ((beta ** 2) * p_lcs) + 1e-12)

    return f_lcs


def rouge_l_multi_process(evals, refs):
    assert evals.shape == refs.shape
    global pool
    if pool is None:
        pool = multiprocessing.Pool(processes=n_process)

    data_length, _ = evals.shape
    tmp_scores = [None] * data_length
    for i in range(data_length):
        tmp_scores[i] = pool.apply_async(s_rouge_l, (evals[i], refs[i]))

    scores = []
    for i in range(data_length):
        scores.append(tmp_scores[i].get())
    scores = np.asarray(scores, dtype=np.float32)

    return scores


def rouge_l_single_process(evals, refs):
    assert evals.shape == refs.shape

    data_length, _ = evals.shape
    scores = []
    for i in range(data_length):
        scores.append(s_rouge_l(evals[i], refs[i]))

    scores = np.asarray(scores, dtype=np.float32)
    return scores


rouge_l = rouge_l_single_process
