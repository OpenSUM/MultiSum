import gc
import os
import time
import traceback

import numpy as np
import tensorflow as tf

from constants import *
from model import *
from utils import calc_rouge, eprint, BertConfig, calc_acc
from utils.Batcher import get_batcher
from utils.Saver import Saver
from utils.data_loader import load_vocab, id2text
import sys


def test_batch(model, sess, eval_batcher, seq_length, word2id, id2word, use_pointer, substr_prefix, verbose=False, **kwargs):
    """
    :param model:
    :param sess:
    :param eval_batcher:
    :param seq_length:
    :param word2id:
    :param id2word:
    :param use_pointer:
    :param substr_prefix:
    :param verbose:
    :param kwargs:
    :return:
    """
    assert len(word2id) == len(id2word)
    vocab_size = len(word2id)
    outputs = []
    trues = []
    sources = []
    losses = []
    sentiments = []
    pred_sentiments = []
    substr_replacement = ' {}'.format(substr_prefix)

    start_time = time.time()
    last_time = start_time
    for batch_index, batch_data in enumerate(eval_batcher.batch(), 1):
        (y_token, y_ids, y_ids_loss, y_extend, y_mask,
         x_token, x_ids, x_extend, x_mask, oov_size, oovs, sentiment) = batch_data
        if not use_pointer:
            y_ids_loss[y_ids_loss >= vocab_size] = word2id[UNK_TOKEN]
        batch_sentis = []
        # run encoder
        fd = model.get_decode_encoder_feed_dict(batch_data)
        result = sess.run([model.encoder_output_for_decoder,model.pred_senti], feed_dict=fd)
        encoder_output,senti=result[0],result[1]
        batch_sentis.append(senti)
        encoder_output = model._split_encoder_output(encoder_output)

        # run decoder step by step
        prev_extend = np.zeros(shape=y_ids.shape, dtype=np.int32)
        prev_ids = np.zeros(shape=y_ids.shape, dtype=np.int32)
        batch_losses = []
        
        for i in range(seq_length):
            batch_data[1] = prev_ids
            fd = model.get_decode_decoder_feed_dict(batch_data=batch_data, split_encoder_output=encoder_output)
            result = sess.run([model.y_pred, model.loss_matrix_ml], feed_dict=fd)
            preds, loss = result[0], result[1]
            batch_losses.append(loss[:, i])
            prev_extend = np.concatenate((preds[:, :i + 1], prev_extend[:, i + 1:]), axis=-1)
            prev_ids = np.copy(prev_extend)
            prev_ids[prev_ids >= vocab_size] = word2id[UNK_TOKEN]

        batch_loss = np.vstack(batch_losses).T
        batch_loss = np.sum(batch_loss, axis=-1) / np.sum(y_mask, axis=-1, dtype=np.float32)
        losses.append(batch_loss)
        pred_sentiments.append(batch_sentis[0])
        sentiments.append(sentiment)
        for i, abs in enumerate(prev_extend.tolist()):
            output = []
            for w in abs:
                if w != word2id[SEP_TOKEN]:
                    if w > 0:
                        output.append(w)
                else:
                    break
            if len(output) == 0:
                output = word2id[SEP_TOKEN]
            outputs.append(id2text(ids=output, id2word=id2word, oov=oovs[i]).replace(substr_replacement, ''))
        tmp_trues = [' '.join(l).replace(substr_replacement, '') for l in y_token.tolist()]
        if verbose:
            for t in tmp_trues:
                print(t)
        trues.extend(tmp_trues)
        sources.extend([' '.join(l).replace(substr_replacement, '') for l in x_token.tolist()])
        t = time.time()
        print('Batch {}, time: {:.2f}s, total time: {:.2f}s'.format(batch_index, t - last_time, t - start_time))
        last_time = t

    print('Total Eval Time: {:.2f}s'.format(time.time() - start_time))
    scores = calc_rouge(outputs, trues, placeholder=substr_prefix)
    acc,f1= calc_acc(trues=sentiments, preds=pred_sentiments)

    return scores, np.mean(np.concatenate(losses)), dict(source=sources, ref=trues, cand=outputs,ref_senti=sentiments), acc,f1, pred_sentiments




def evaluate(FLAGS):
    """

    :param FLAGS:
    :return:
    """
    assert FLAGS.init_checkpoint is not None
    # ************************************************************************

    t = time.time()
    # load parameter from checkpoints file.
    ckpt_path = os.path.join(EXP_DIR, FLAGS.init_checkpoint)
    assert os.path.exists(ckpt_path)  or os.path.isfile(ckpt_path)
    saver = Saver(ckpt_dir=ckpt_path, max_to_keep=CHECKPOINTS_MAX_TO_KEEP)
    config = BertConfig.from_json_file(saver.hyper_parameter_filepath)
    merge_flags_config(FLAGS, config)

    print('\n******************** Hyper parameters: ********************')
    for k, v in config.__dict__.items():
        print('\t{}: {}'.format(k, v))
    print('***********************************************************\n')

    print('Loading data...')
    word2id, id2word = load_vocab(config.vocab_file, do_lower=config.do_lower)
    batcher = get_batcher(src_file=config.eval_src if config.mode.lower() == 'eval' else config.test_src,
                          dst_file=config.eval_dst if config.mode.lower() == 'eval' else config.test_dst,
                          senti_file=config.eval_senti if config.mode.lower() == 'eval' else config.test_senti,
                          word2id=word2id, src_seq_length=config.encoder_seq_length,
                          dst_seq_length=config.decoder_seq_length, batch_size=config.batch_size,
                          do_lower=config.do_lower,
                          substr_prefix=config.substr_prefix,
                          limit=EVAL_SAMPLE_LIMIT)
    print('Time: {:.1f}s'.format(time.time() - t))
    print('Finish loading data...')

    # build model
    #model =Model(config=config)
    model = MultiGPUModel(config=config, num_gpus=config.num_gpus)
    model.build(is_training=False)

    print('GC-ing...')
    gct = time.time()
    gc.collect()
    print('GC Finish! Time: %.1f' % (time.time() - gct))

    # train the model.
    print('Preparing...')
    saver.init_saver()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver.initialize_variables(ckpt_path=config.checkpoint_file)
        sess.run(tf.global_variables_initializer())
        print('Preparing Finish.\n')

        start_time = time.time()
        print('Evaluating...' if config.mode.lower() == 'eval' else 'Testing...')
        epoch_start = time.time()
        scores, loss, texts, acc,f1,sentiment = test_batch(model=model, sess=sess, eval_batcher=batcher,
                                                         seq_length=config.decoder_seq_length, word2id=word2id, id2word=id2word,
                                                         use_pointer=config.use_pointer, substr_prefix=config.substr_prefix,
                                                         beam_size=2)
        saver.save_summaries(sources=texts['source'], refs=texts['ref'], cands=texts['cand'],
                             step=saver.ckpt_path.split('/')[-1], suffix=config.mode, sentis=sentiment,ref_senti=texts['ref_senti'])
        o = ('Rouge-1:{r1:.8}, Rouge-2:{r2:.8}, '
             'Rouge-L:{rl:.8}, acc:{acc:.8}, f1:{f1:.8},time:{time:.1f}s').format(
            total=config.epochs, loss=loss,
            r1=scores['rouge-1']['f'], r2=scores['rouge-2']['f'],
            rl=scores['rouge-l']['f'], acc=acc, f1=f1,time=time.time() - epoch_start)
        print(o)

        print('Finish. total time:{time:.1f}s'.format(time=time.time() - start_time))




def train(FLAGS):
    """
    :param FLAGS:
    :return:
    """
    # load and configure hyper-parameters.
    t = time.time()
    if FLAGS.init_checkpoint:
        # load parameter from checkpoints file.
        ckpt = FLAGS.init_checkpoint
    else:
        ckpt = 'checkpoint_{time}'.format(time=time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()))

    ckpt_path = os.path.join(EXP_DIR, ckpt)
    ckpt_exist = os.path.exists(ckpt_path) and tf.train.latest_checkpoint(ckpt_path) is not None
    os.makedirs(ckpt_path, exist_ok=True)
    saver = Saver(ckpt_dir=ckpt_path, max_to_keep=CHECKPOINTS_MAX_TO_KEEP)  # get Saver object

    if FLAGS.init_checkpoint:
        config = BertConfig.from_json_file(saver.hyper_parameter_filepath)
    else:
        config = BertConfig.from_json_file(FLAGS.bert_config_file)
    merge_flags_config(FLAGS, config)

    if config.train_from_scratch:
        config.gradual_unfreezing = False
        config.discriminative_fine_tuning = False
        config.encoder_trainable_layers = -1
        config.trainable_layers = -1
        config.embedding_trainable = True
        config.pooler_layer_trainable = True
        config.masked_layer_trainable = True
        config.attention_layer_trainable = True

    saver.save_hyper_parameters(config.__dict__)

    print('****** Log content has been redirected to file %s ******' % saver.log_filepath)
    print('****** Please make sure you have save this checkpoint directory! ******')

    print('\n******************** Hyper parameters: ********************')
    for k, v in config.__dict__.items():
        print('\t{}: {}'.format(k, v))
    print('***********************************************************\n')

    # load dataset.
    print('Loading data...')
    word2id, id2word = load_vocab(config.vocab_file, do_lower=config.do_lower)
    train_batcher = get_batcher(src_file=config.train_src,
                                dst_file=config.train_dst,
                                senti_file=config.train_senti,
                                word2id=word2id, src_seq_length=config.encoder_seq_length,
                                dst_seq_length=config.decoder_seq_length, batch_size=config.batch_size,
                                do_lower=config.do_lower,
                                substr_prefix=config.substr_prefix,
                                limit=TRAIN_SAMPLE_LIMIT)
    setattr(config, 'steps_per_epoch', train_batcher.iterations)
    print('Time: {:.1f}s'.format(time.time() - t))
    print('Finish loading data...')

    # build model
    #model = Model(config)
    model = MultiGPUModel(config=config, num_gpus=config.num_gpus)
    model.build(is_training=True)
    if FLAGS.use_fgm:
        print("START TO PREPARE FGM")
        bert_grads_first,other_grads_first,action_list_dc=deep_copy_grads(model.averaged_bert_grad_and_vars,model.averaged_other_grad_and_vars)
        print('deepcopy finish')
        action_list_backup,embedding_variable_backup,embedding_value_backup=change_embedding_and_save_para(model.averaged_bert_grad_and_vars)
        print('change embedding finish')
        bert_grads_ultimate,other_grads_ultimate,action_list_merge=merge_grads(bert_grads_first,other_grads_first,model.averaged_bert_grad_and_vars,model.averaged_other_grad_and_vars)
        print('merge grads finish')
        action_list_restore = restore_embedding(embedding_variable_backup,embedding_value_backup)
        print('restore embedding finish')
        DIY_step,DIY_lr,action_list_backward = DIY_backward(config,model,bert_grads_ultimate,other_grads_ultimate)
        print('backward finish')

    print('GC-ing...')
    gct = time.time()
    gc.collect()
    print('GC Finish! Time: %.1f' % (time.time() - gct))

    # train the model.
    print('Preparing...')
    start_time = time.time()
    saver.init_saver()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        print('Start initialize from Saver.')
        if ckpt_exist:
            # load variables with interrupted training
            start_epoch = saver.initialize_variables(ckpt_path=config.checkpoint_file)
        elif FLAGS.train_from_scratch:
            # random initialized
            start_epoch = 0
            saver.print_variables()
        elif config.train_from_roberta:
            #load variables from roberta
            start_epoch = saver.initialize_variables(ckpt_path=config.roberta_checkpoint, from_roberta=True,
                                                     layers_filter=config.hidden_layers_filter)
        elif config.train_from_pretrained:
            #load variables from pretrained_parameter
            start_epoch = saver.initialize_variables(ckpt_path=config.pretrained_checkpoint, from_pretrained=True,
                                                     layers_filter=config.hidden_layers_filter)
        else:
            # load variables from bert
            start_epoch = saver.initialize_variables(ckpt_path=config.init_checkpoint, from_bert=True,
                                                     layers_filter=config.hidden_layers_filter)

        bert_learning_rate = config.bert_learning_rate
        other_learning_rate = config.other_learning_rate
        print('Run global_variables_initializer()')
        sess.run(tf.global_variables_initializer())
        saver.init_file_writer(verbose=True)
        print('Preparing Finish.')

        print('Start Training...')
        for epoch_index in range(start_epoch, config.epochs):
            epoch_start = time.time()
            batch_start = time.time()
            for batch_index, batch_data in enumerate(train_batcher.batch(), 1):
                if not config.use_pointer:
                    y_ids_loss = batch_data[2]
                    y_ids_loss[y_ids_loss >= len(word2id)] = word2id[UNK_TOKEN]
                fd = model.get_feed_dict(is_training=True, batch_data=batch_data)
                
                if FLAGS.use_fgm:
                    res=sess.run([model.summary_loss,model.senti_loss,saver.merged_op,model.averaged_bert_grad_and_vars,model.averaged_other_grad_and_vars,
                              bert_grads_first,other_grads_first,action_list_dc,
                              action_list_backup,embedding_variable_backup,embedding_value_backup],feed_dict=fd)
                    loss_summary=res[0]
                    loss_senti=res[1]
                    res=sess.run([model.summary_loss,model.senti_loss,saver.merged_op,model.averaged_bert_grad_and_vars,model.averaged_other_grad_and_vars,
                    bert_grads_ultimate,other_grads_ultimate,action_list_merge,
                    ],feed_dict=fd)
                    fgm_loss_summary=res[0]
                    fgm_loss_senti=res[1]
                    sess.run(action_list_restore)

                    res=sess.run([DIY_step,DIY_lr,action_list_backward])
                    global_step,lr=res[0],res[1]
                else:
                    res = sess.run([model.train_op,
                                    model.global_step,
                                    model.bert_optimizer.learning_rate,
                                    model.summary_loss,
                                    saver.merged_op],
                                    feed_dict=fd,
                                   )
                    global_step = res[1]
                    lr = res[2]
                    loss_summary = res[3]
                    
                if batch_index % PRINT_STEPS == 0:
                    saver.summary(loss_summary=loss_summary,prefix='train', global_step=global_step, bert_lr=lr)
                    print('batch {i},  Loss_Summary:{loss_summary:.8f},bert_lr={lr:.10f}, time:{time:.1f}s'.format(
                        i=batch_index, loss_summary=loss_summary,time=time.time() - batch_start, lr=lr if lr else 0.0))
                    if FLAGS.use_fgm:
                        print('batch {i}, Fgm_Loss_Senti:{fgm_loss_senti:.8f}, Fgm_Loss_Summary:{fgm_loss_summary:.8f},bert_lr={lr:.10f}, time:{time:.1f}s'.format(
                        i=batch_index, fgm_loss_senti=fgm_loss_senti, fgm_loss_summary=fgm_loss_summary,time=time.time() - batch_start, lr=lr if lr else 0.0))
                    batch_start = time.time()
                if global_step % CHECK_GLOBAL_STEPS == 0 and global_step != 0 and \
                        (HALVE_BERT_LR and bert_learning_rate >= MIN_LEARNING_RATE or
                         HALVE_OTHER_LR and other_learning_rate >= MIN_LEARNING_RATE):
                    print(('\t{} batches has been trained, scoring validation data set '
                           'for halving the learning rate...').format(CHECK_GLOBAL_STEPS))

            if not config.debug:
                saver.save(sess=sess, step=epoch_index)
            print('Epoch: {i}/{total}, time:{time:.1f}s\n'.format(
                i=epoch_index, total=config.epochs, time=time.time() - epoch_start))
        print('Finish. total time:{time:.1f}s'.format(time=time.time() - start_time))
        saver.close()


def merge_flags_config(flag, config):
    fields = [
        'debug',
        'train_src',
        'train_dst',
        'eval_src',
        'eval_dst',
        'test_src',
        'test_dst',
        'mode',
        'num_gpus',
        'batch_size',
        'learning_rate',
        'bert_learning_rate',
        'other_learning_rate',
        'theta',
        'init_checkpoint',
        'gradual_unfreezing',
        'discriminative_fine_tuning',
        'num_hidden_layers',
        'trainable_layers',
        # 'hidden_layers_filter',     this one needs to be process separately
        'encoder_trainable_layers',
        'embedding_trainable',
        'pooler_layer_trainable',
        'masked_layer_trainable',
        'attention_layer_trainable',
        'pointer_initializer',
        "use_pointer",
        'coverage',
        'trim_attention',
        'align_layers',
        'train_from_scratch',
        'name',
        'train_from_roberta',
        'roberta_checkpoint',
        'use_fgm',
        'use_pool_connected',
        'train_from_pretrained',
        'pretrained_checkpoint',
        'bert_config_file'
    ]
    for field in fields:
        if getattr(flag, field, None) is not None:
            setattr(config, field, getattr(flag, field))
    include_fileds = [
        'checkpoint_file',
    ]
    for field in include_fileds:
        setattr(config, field, getattr(flag, field))
    if getattr(flag, 'hidden_layers_filter', None) is not None:
        try:
            s = getattr(flag, 'hidden_layers_filter').split(',')
            s = map(int, s)
            setattr(config, 'hidden_layers_filter', tuple(s))
        except:
            traceback.print_exc()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.set_random_seed(4399)

    flags = tf.flags
    FLAGS = flags.FLAGS
 

    tf.flags.DEFINE_boolean('debug', False, 'whether save checkpoints or not.')

    tf.flags.DEFINE_string("train_src", None, "")
    tf.flags.DEFINE_string("train_dst", None, "")

    tf.flags.DEFINE_string("eval_src", None, "")
    tf.flags.DEFINE_string("eval_dst", None, "")

    tf.flags.DEFINE_string("test_src", None, "")
    tf.flags.DEFINE_string("test_dst", None, "")
    tf.flags.DEFINE_string('mode', 'train', 'train/eval/test/predict(not support yet)')
    tf.flags.DEFINE_integer("num_gpus", 4, "Number of GPUs")

    tf.flags.DEFINE_integer("batch_size", None, "batch size.")
    tf.flags.DEFINE_float("bert_learning_rate", None, "learning_rate of bert(encoder and decoder).")
    tf.flags.DEFINE_float("other_learning_rate", None, "learning_rate of parts except from bert(encoder and decoder)")
    tf.flags.DEFINE_float("learning_rate", None,
                       "[BACKUP PARAMETERS] learning_rate of parts except from bert(encoder and decoder)")
    tf.flags.DEFINE_float("theta", None, "weight of RL loss function.")
    tf.flags.DEFINE_string("init_checkpoint", None, "initial checkpoint directory.")
    tf.flags.DEFINE_string("roberta_checkpoint", None, "roberta checkpoint directory.")
    tf.flags.DEFINE_string("pretrained_checkpoint", None, "pretrained checkpoint directory.")
    tf.flags.DEFINE_string("checkpoint_file", None,
                        "checkpoint filename in folder ```init_checkpoint``` param.")

    tf.flags.DEFINE_boolean('gradual_unfreezing', None, 'whether use gradual unfreezing or not.')
    tf.flags.DEFINE_boolean('discriminative_fine_tuning', None, 'whether use discriminative fine-tuning or not.')

    tf.flags.DEFINE_integer("num_hidden_layers", None, "number of hidden layers in transformer model.")
    tf.flags.DEFINE_string("hidden_layers_filter", None, "layers of parameters which will be loaded to the model.")

    tf.flags.DEFINE_integer("trainable_layers", None,
                         "number of trainable layers in decoder, -1 means all layers are trainable.")
    tf.flags.DEFINE_integer("encoder_trainable_layers", None,
                         "number of trainable layers in encoder, -1 means all layers are trainable.")
    tf.flags.DEFINE_boolean("embedding_trainable", None, "embedding matrix trainable or not in encoder and decoder.")
    tf.flags.DEFINE_boolean("pooler_layer_trainable", None, "[Deprecated] pooler layer trainable or not in decoder.")
    tf.flags.DEFINE_boolean("masked_layer_trainable", None, "masked layer trainable or not in decoder.")
    tf.flags.DEFINE_boolean("attention_layer_trainable", None, "attention layer trainable or not in decoder.")
    tf.flags.DEFINE_string('pointer_initializer', None,
                        'one of [xavier/normal/truncated], initializer of parameters in Pointer.')
    tf.flags.DEFINE_boolean('use_pointer', None, 'use Pointer Generator Mechanism.')
    tf.flags.DEFINE_boolean('coverage', False, 'use Coverage Mechanism.')
    tf.flags.DEFINE_boolean('trim_attention', False, 'use Trim Relative Self-Attention.')
    tf.flags.DEFINE_boolean('align_layers', False, 'align encoder and decoder layers.')
    tf.flags.DEFINE_string('name', None, 'Name of the experiments')
    tf.flags.DEFINE_string('use_fgm', None, 'adversarial training or not')
    tf.flags.DEFINE_string('use_pool_connected', None, 'when predicting sentiments,using CLS directly or max pool other layers and connect it with CLS')
    tf.flags.DEFINE_string(
        "bert_config_file", None,
        "The config json file corresponding to the pre-trained BERT model. "
        "This specifies the model architecture.")

    tf.flags.DEFINE_boolean('train_from_scratch', None,
                         'train from scratch and don\'t use pre-trained BERT parameters.')
    tf.flags.DEFINE_boolean('train_from_roberta', None,
                         'train from roberta and  use pre-trained ROBERTa parameters.')
    tf.flags.DEFINE_boolean('train_from_pretrained', None,
                         'train from pretrained and  use pre-trained pretrained parameters.')


    if FLAGS.mode.lower() == 'train':
        try:
            train(FLAGS)
        except:
            traceback.print_exc()
    elif FLAGS.mode.lower() == 'eval' or FLAGS.mode.lower() == 'test':
        evaluate(FLAGS)
    else:
        eprint('[ERROR] Mode parameter should be train/eval')
