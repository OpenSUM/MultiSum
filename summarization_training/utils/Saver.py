import collections
import json
import os

import tensorflow as tf

from utils import get_assignment_map_from_checkpoint, eprint


def create_assignment_map_from_bert_or_roberta(layers=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)):
    embed_map = collections.OrderedDict()
    enc_map = collections.OrderedDict()
    dec_map = collections.OrderedDict()
    masked_map = collections.OrderedDict()
    # embeddings
    trainable = [
        'embeddings/position_embeddings',
        'embeddings/token_type_embeddings',
        'embeddings/word_embeddings',
        'embeddings/LayerNorm/beta',
        'embeddings/LayerNorm/gamma',
    ]
    ckpt = [
        'bert/embeddings/position_embeddings',
        'bert/embeddings/token_type_embeddings',
        'bert/embeddings/word_embeddings',
        'bert/embeddings/LayerNorm/beta',
        'bert/embeddings/LayerNorm/gamma',
    ]
    for ckpt_var, trainable_var in zip(ckpt, trainable):
        embed_map[trainable_var] = ckpt_var

    # transformer
    trainable = [
        '{scope}/encoder/layer_{index}/{masked}attention/self/query/kernel',
        '{scope}/encoder/layer_{index}/{masked}attention/self/query/bias',
        '{scope}/encoder/layer_{index}/{masked}attention/self/key/kernel',
        '{scope}/encoder/layer_{index}/{masked}attention/self/key/bias',
        '{scope}/encoder/layer_{index}/{masked}attention/self/value/kernel',
        '{scope}/encoder/layer_{index}/{masked}attention/self/value/bias',
        '{scope}/encoder/layer_{index}/{masked}attention/output/dense/kernel',
        '{scope}/encoder/layer_{index}/{masked}attention/output/dense/bias',
        '{scope}/encoder/layer_{index}/{masked}attention/output/LayerNorm/beta',
        '{scope}/encoder/layer_{index}/{masked}attention/output/LayerNorm/gamma',
        '{scope}/encoder/layer_{index}/intermediate/dense/kernel',
        '{scope}/encoder/layer_{index}/intermediate/dense/bias',
        '{scope}/encoder/layer_{index}/output/dense/kernel',
        '{scope}/encoder/layer_{index}/output/dense/bias',
        '{scope}/encoder/layer_{index}/output/LayerNorm/beta',
        '{scope}/encoder/layer_{index}/output/LayerNorm/gamma'
    ]
    ckpt = [
        'bert/encoder/layer_{index}/attention/self/query/kernel',
        'bert/encoder/layer_{index}/attention/self/query/bias',
        'bert/encoder/layer_{index}/attention/self/key/kernel',
        'bert/encoder/layer_{index}/attention/self/key/bias',
        'bert/encoder/layer_{index}/attention/self/value/kernel',
        'bert/encoder/layer_{index}/attention/self/value/bias',
        'bert/encoder/layer_{index}/attention/output/dense/kernel',
        'bert/encoder/layer_{index}/attention/output/dense/bias',
        'bert/encoder/layer_{index}/attention/output/LayerNorm/beta',
        'bert/encoder/layer_{index}/attention/output/LayerNorm/gamma',
        'bert/encoder/layer_{index}/intermediate/dense/kernel',
        'bert/encoder/layer_{index}/intermediate/dense/bias',
        'bert/encoder/layer_{index}/output/dense/kernel',
        'bert/encoder/layer_{index}/output/dense/bias',
        'bert/encoder/layer_{index}/output/LayerNorm/beta',
        'bert/encoder/layer_{index}/output/LayerNorm/gamma'
    ]
    for i, layer_index in enumerate(layers):
        for ckpt_var, trainable_var in zip(ckpt, trainable):
            enc_map[trainable_var.format(scope='enc', index=i, masked='')] = ckpt_var.format(index=layer_index)
        for ckpt_var, trainable_var in zip(ckpt, trainable):
            dec_map[trainable_var.format(scope='dec', index=i, masked='')] = ckpt_var.format(index=layer_index)

    # masked attention
    trainable = [
        '{scope}/encoder/layer_{index}/{masked}attention/self/query/kernel',
        '{scope}/encoder/layer_{index}/{masked}attention/self/query/bias',
        '{scope}/encoder/layer_{index}/{masked}attention/self/key/kernel',
        '{scope}/encoder/layer_{index}/{masked}attention/self/key/bias',
        '{scope}/encoder/layer_{index}/{masked}attention/self/value/kernel',
        '{scope}/encoder/layer_{index}/{masked}attention/self/value/bias',
        '{scope}/encoder/layer_{index}/{masked}attention/output/dense/kernel',
        '{scope}/encoder/layer_{index}/{masked}attention/output/dense/bias',
        '{scope}/encoder/layer_{index}/{masked}attention/output/LayerNorm/beta',
        '{scope}/encoder/layer_{index}/{masked}attention/output/LayerNorm/gamma',
    ]
    ckpt = [
        'bert/encoder/layer_{index}/attention/self/query/kernel',
        'bert/encoder/layer_{index}/attention/self/query/bias',
        'bert/encoder/layer_{index}/attention/self/key/kernel',
        'bert/encoder/layer_{index}/attention/self/key/bias',
        'bert/encoder/layer_{index}/attention/self/value/kernel',
        'bert/encoder/layer_{index}/attention/self/value/bias',
        'bert/encoder/layer_{index}/attention/output/dense/kernel',
        'bert/encoder/layer_{index}/attention/output/dense/bias',
        'bert/encoder/layer_{index}/attention/output/LayerNorm/beta',
        'bert/encoder/layer_{index}/attention/output/LayerNorm/gamma',
    ]

    # get_inited_names
    inited_names = []
    for n in embed_map.keys():
        inited_names.append(n)
        inited_names.append(n + ':0')
    for n in enc_map.keys():
        inited_names.append(n)
        inited_names.append(n + ':0')
    for n in dec_map.keys():
        inited_names.append(n)
        inited_names.append(n + ':0')
    for n in masked_map.keys():
        inited_names.append(n)
        inited_names.append(n + ':0')

    return (collections.OrderedDict((v, k) for k, v in enc_map.items()),
            collections.OrderedDict((v, k) for k, v in enc_map.items()),
            collections.OrderedDict((v, k) for k, v in dec_map.items()),
            collections.OrderedDict((v, k) for k, v in masked_map.items())), inited_names


class Saver:
    def __init__(self, ckpt_dir, max_to_keep=None, init_now=False):
        if os.path.exists(ckpt_dir):
            if os.path.isfile(ckpt_dir):
                self.ckpt_path = ckpt_dir
                self.ckpt_dir = os.path.dirname(ckpt_dir)
            else:
                self.ckpt_dir = ckpt_dir
                self.ckpt_path = None
        self.max_to_keep = max_to_keep
        if init_now:
            self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)
        else:
            self.saver = None
        self.file_writer = None
        self.merged_op = None
        self._stdout_fp = None

    def init_saver(self, force=False):
        if force or self.saver is None:
            self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)

    def init_file_writer(self, graph=None, verbose=False):
        if verbose:
            print('[Saver] Initializing FileWriter...')
        if graph is None:
            graph = tf.get_default_graph()
        self.merged_op = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(os.path.join(self.ckpt_dir, 'tensorboard'), graph)
        if verbose:
            print('[Saver] Finish initializing FileWriter.')

    def save_hyper_parameters(self, js, override=False):
        filepath = os.path.join(self.ckpt_dir, 'hyper_parameter.json')
        if not override and os.path.exists(filepath):
            i = 1
            while os.path.exists(os.path.join(self.ckpt_dir, 'hyper_parameter_%d.json' % i)):
                i += 1
            filepath = os.path.join(self.ckpt_dir, 'hyper_parameter_%d.json' % i)
        with open(filepath, 'w', encoding='utf8') as f:
            json.dump(js, f)

    @property
    def hyper_parameter_filepath(self):
        """
        Return the last hyper_parameter json filename. If no hyper parameter file exists, return 'hyper_parameter.json'.
        :return: filename of the hyper parameter file saved in checkpoint folder.
        """
        filepath = os.path.join(self.ckpt_dir, 'hyper_parameter.json')
        if not os.path.exists(filepath):
            return filepath
        i = 1
        while os.path.exists(os.path.join(self.ckpt_dir, 'hyper_parameter_%d.json' % i)):
            filepath = os.path.join(self.ckpt_dir, 'hyper_parameter_%d.json' % i)
            i += 1
        return filepath

    @staticmethod
    def parse_hyper_parameter_filepath(ckpt_path):
        """
        Return the last hyper_parameter json filename. If no hyper parameter file exists, return 'hyper_parameter.json'.
        :return: filename of the hyper parameter file saved in checkpoint folder.
        """
        if os.path.exists(ckpt_path):
            if os.path.isfile(ckpt_path):
                ckpt_path = os.path.dirname(ckpt_path)
            filepath = os.path.join(ckpt_path, 'hyper_parameter.json')
            if not os.path.exists(filepath):
                eprint('[ERROR] Invalid ckpt_path!')
                return None
            i = 1
            while os.path.exists(os.path.join(ckpt_path, 'hyper_parameter_%d.json' % i)):
                filepath = os.path.join(ckpt_path, 'hyper_parameter_%d.json' % i)
                i += 1
            return filepath
        else:
            eprint('[ERROR] Invalid ckpt_path!')
            return None

    @property
    def log_filepath(self):
        return self.stdout.name if self.stdout is not None else None

    @property
    def stdout(self):
        if self._stdout_fp is None or self._stdout_fp.closed:
            filepath = os.path.join(self.ckpt_dir, 'output.log')
            if os.path.exists(filepath):
                i = 1
                while os.path.exists(os.path.join(self.ckpt_dir, 'output%d.log' % i)):
                    i += 1
                filepath = os.path.join(self.ckpt_dir, 'output%d.log' % i)
            self._stdout_fp = open(filepath, 'w', encoding='utf8', buffering=1)
        return self._stdout_fp

    def close(self):
        self.stdout.flush()
        self.stdout.close()

    def summary(self, loss_senti=None,loss_summary=None,scores=None, prefix='', global_step=None, **kwargs):
        assert self.merged_op is not None and self.file_writer is not None

        if prefix != '' and not prefix.endswith('_'):
            prefix = prefix + '_'
        if loss_senti is not None:
            summary_loss_senti = tf.Summary()
            summary_loss_senti.value.add(tag='%sLoss_Senti' % prefix, simple_value=loss_senti)
            self.file_writer.add_summary(summary_loss_senti, global_step=global_step)
        if loss_summary is not None:
            summary_loss_summary = tf.Summary()
            summary_loss_summary.value.add(tag='%sLoss_Summary' % prefix, simple_value=loss_summary)
            self.file_writer.add_summary(summary_loss_summary, global_step=global_step)
        if scores is not None:
            summary_r1 = tf.Summary()
            summary_r1.value.add(tag='%sRouge-1' % prefix, simple_value=scores['rouge-1']['f'])
            self.file_writer.add_summary(summary_r1, global_step=global_step)
            summary_r2 = tf.Summary()
            summary_r2.value.add(tag='%sRouge-2' % prefix, simple_value=scores['rouge-2']['f'])
            self.file_writer.add_summary(summary_r2, global_step=global_step)
            summary_rl = tf.Summary()
            summary_rl.value.add(tag='%sRouge-L' % prefix, simple_value=scores['rouge-l']['f'])
            self.file_writer.add_summary(summary_rl, global_step=global_step)
        for k, v in kwargs.items():
            if v is None:
                continue
            summary = tf.Summary()
            summary.value.add(tag='%s%s' % (prefix, k), simple_value=v)
            self.file_writer.add_summary(summary, global_step=global_step)

    def save_summaries(self, cands, refs=None, step=None, suffix=None, sources=None, folder=None, sentis=None,ref_senti=None,my_type=''):
        """
        :param cands:
        :param refs:
        :param step:
        :param suffix:
        :param sources:
        :param folder:
        :return:
        """
        if folder:
            summary_dir = folder
        else:
            summary_dir = os.path.join(self.ckpt_dir, 'summary'+my_type)
        os.makedirs(summary_dir, exist_ok=True)
        suffix = '' if suffix is None else ('_{}'.format(suffix))
        if sources is not None:
            src_filepath = os.path.join(summary_dir, 'source' + suffix)
            if not os.path.exists(src_filepath):
                with open(src_filepath, 'w', encoding='utf8') as f:
                    for line in sources:
                        f.write(line)
                        f.write('\n')
        ref_filepath = os.path.join(summary_dir, 'ref' + suffix)
        cand_filepath = os.path.join(summary_dir, 'cand{}_{}'.format(suffix, step))
        if refs is not None:
            if not os.path.exists(ref_filepath):
                with open(ref_filepath, 'w', encoding='utf8') as f:
                    for line in refs:
                        f.write(line)
                        f.write('\n')

        if cands is not None:
            with open(cand_filepath, 'w', encoding='utf8') as f:
                for line in cands:
                    f.write(line)
                    f.write('\n')


    def save(self, sess, step, tag=None, loss=None, rg=None):
        if not tag:
            tag = 'best'
        self.saver.save(sess=sess, save_path=os.path.join(self.ckpt_dir, tag), global_step=step)
        if loss is not None or rg is not None:
            with open(os.path.join(self.ckpt_dir, tag) + ('-%d' % step), 'w', encoding='utf8') as f:
                if loss is not None:
                    f.write('%f\n' % loss)
                else:
                    f.write('999.9\n')
                if rg is not None:
                    f.write(json.dumps(rg))

    @staticmethod
    def _is_not_optimization(k, v=None):
        not_k = type(k) != str or not k.startswith('optimization/')
        if v is None:
            return not_k
        not_v = type(v) != str or not v.startswith('optimization/')
        return not_k and not_v

    def initialize_variables(self, ckpt_dir=None, ckpt_path=None, from_bert=False,from_roberta=False,from_pretrained=False,
                             layers_filter=tuple(range(12))):
        ckpt_path_backup=ckpt_path
        if ckpt_path is None and ckpt_dir is None:
            if self.ckpt_path is None:
                self.ckpt_path = ckpt_path = tf.train.latest_checkpoint(self.ckpt_dir)
            else:
                ckpt_path = self.ckpt_path
        elif ckpt_path is not None:
            ckpt_path = os.path.join(self.ckpt_dir, ckpt_path)
            if self.ckpt_path is None:
                self.ckpt_path = ckpt_path
        else:
            ckpt_path = tf.train.latest_checkpoint(ckpt_dir)

        start_epoch = ckpt_path.split('-')
        if len(start_epoch) > 1 and start_epoch[-1].isdigit():
            start_epoch = int(start_epoch[-1]) + 1
        else:
            start_epoch = 0

        tvars = tf.global_variables()
        if from_bert or from_roberta or from_pretrained:
            print("INITIALIZED FROM PRETRAINED_MODEL")
            maps, initialized_variable_names = create_assignment_map_from_bert_or_roberta(layers=layers_filter)
            print(ckpt_path_backup)
            for m in maps:
                tf.train.init_from_checkpoint(ckpt_path_backup, m)
        else:
            print("INITIALIZED FROM CHECKPOINT")
            (assignment_map,
             initialized_variable_names) = get_assignment_map_from_checkpoint(tvars, ckpt_path)
            tf.train.init_from_checkpoint(ckpt_path, assignment_map)

        self.print_variables(initialized_variable_names=initialized_variable_names)

        return start_epoch

    @staticmethod
    def print_variables(initialized_variable_names=()):
        print("**** Pre-train Variables ****")
        for var in tf.global_variables():
            init_string = ""
            trainable_string = '' if var.trainable == False else ' [Trainable]'
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
                txt = "name = %s, shape = %s%s%s" % (var.name, var.shape, init_string, trainable_string)
            else:
                txt = '[Not Initialized] name = %s, shape = %s%s%s' % (var.name, var.shape,
                                                                       init_string, trainable_string)
            print(txt)

    def restore(self, sess, ckpt_dir=None):
        if ckpt_dir is None:
            ckpt_dir = self.ckpt_dir
        ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
        self.saver.restore(sess=sess, save_path=ckpt_path)
