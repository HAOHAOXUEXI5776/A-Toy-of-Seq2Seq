# coding: utf-8
# @Author: Qin WT
# @Time: 2019/7/8 8:45
# model：2 layer rnn，unidirectional encoder

import os
import math
import opennmt
import logging
import numpy as np
import tensorflow as tf

import my_utils
import data_help


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# needed files
tf.app.flags.DEFINE_string('vocab', 'data/vocab', 'help')
tf.app.flags.DEFINE_string('train', 'data/train',
                                    "files to training segmented data, seperated by ';' ")
tf.app.flags.DEFINE_string('eval', 'data/eval', 'help')
tf.app.flags.DEFINE_string('test', 'data/test', 'help')
# directory to save important data
tf.app.flags.DEFINE_string('logdir', 'log/2/', 'help')
tf.app.flags.DEFINE_string('testdir', 'test/2/', 'help')
# module parameters
tf.app.flags.DEFINE_integer('vocab_size', 30, 'help')
tf.app.flags.DEFINE_integer('embed_size', 30, 'help')
tf.app.flags.DEFINE_integer('maxlen', 100, 'help')
tf.app.flags.DEFINE_integer('batch_size', 32, 'help')
# training parameters
tf.app.flags.DEFINE_float('lr', 0.0002, 'help')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 1, 'set to 1 means not decay')

tf.app.flags.DEFINE_integer('num_epochs', 100, 'help')
tf.app.flags.DEFINE_integer('print_period', 10, 'help')
# train or test
tf.app.flags.DEFINE_boolean('is_train', True, 'help')
# which checkpoint to use
tf.app.flags.DEFINE_string('ckpt_path', '', 'help')
# gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

FLAGS = tf.app.flags.FLAGS


token2idx, idx2token = data_help.load_vocab(FLAGS.vocab)
# module
embed = tf.get_variable('embed', [FLAGS.vocab_size, FLAGS.embed_size], tf.float32)

encoder = opennmt.encoders.UnidirectionalRNNEncoder(2, 512)
decoder = opennmt.decoders.RNNDecoder(2, 512, bridge=opennmt.layers.CopyBridge())


def train():
    mode = tf.estimator.ModeKeys.TRAIN
    os.makedirs(FLAGS.logdir, exist_ok=True)
    logging.info('Begin training...')

    # prepare training data
    logging.info('Load training data...')
    train_pairs = []
    train_files = FLAGS.train.split(';')
    for file in train_files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                s, t = line.strip().split('\t')
                s = s.split()
                if len(s) > FLAGS.maxlen: continue
                t = t.split()
                if len(t) > FLAGS.maxlen: continue
                train_pairs.append((s, t))
    train_batch = data_help.MyBatch(train_pairs, FLAGS.batch_size, token2idx)
    num_train_batches = train_batch.num_batch
    num_train_samples = train_batch.num_samples
    logging.info('Train batches: {}   Train samples: {}'.format(num_train_batches, num_train_samples))

    # model
    logging.info('Build model graph...')
    x = tf.placeholder(dtype=tf.int32, shape=(None, None), name='x')
    x_seqlens = tf.placeholder(dtype=tf.int32, shape=(None,), name='x_seqlens')
    y = tf.placeholder(dtype=tf.int32, shape=(None, None), name='y')
    decoder_inputs = tf.placeholder(dtype=tf.int32, shape=(None, None), name='decoder_inputs')
    y_seqlens = tf.placeholder(dtype=tf.int32, shape=(None,), name='y_seqlens')
    with tf.variable_scope('encoder'):
        enc_emb = tf.nn.embedding_lookup(embed, x)
        outputs, state, sequence_length = encoder.encode(enc_emb, x_seqlens, mode=mode)
    with tf.variable_scope('decoder'):
        dec_emb = tf.nn.embedding_lookup(embed, decoder_inputs)
        logits, _, _ = decoder.decode(dec_emb, y_seqlens, vocab_size=FLAGS.vocab_size, memory=outputs, mode=mode,
                                      memory_sequence_length=x_seqlens, initial_state=state)
    loss, normalizer, _ = opennmt.utils.losses.cross_entropy_sequence_loss(logits, y, y_seqlens,
                                                                           average_in_time=True, mode=mode)
    loss /= normalizer
    global_step = tf.train.get_or_create_global_step()
    lr = tf.Variable(FLAGS.lr, trainable=False)
    learning_rate_decay_op = lr.assign(
        lr * FLAGS.learning_rate_decay_factor)
    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.minimize(loss, global_step=global_step)

    # initialize model
    saver = tf.train.Saver(max_to_keep=FLAGS.num_epochs)
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
        ckpt = tf.train.latest_checkpoint(FLAGS.logdir)
        if ckpt is None:
            logging.info("Initializing from scratch")
            sess.run(tf.global_variables_initializer())
        else:
            logging.info("Restore from {}".format(ckpt))
            saver.restore(sess, ckpt)

        # train
        old_loss = []
        total_steps = FLAGS.num_epochs * num_train_batches
        _gs = sess.run(global_step)
        for i in range(_gs, total_steps + 1):
            # get feed data
            batch = train_batch.get_next()
            _x, _x_seqlens, _y, _decoder_inputs, _y_seqlens = train_batch.process_batch(batch)

            _, _gs = sess.run([train_op, global_step], feed_dict={
                x: _x, x_seqlens: _x_seqlens, y: _y, y_seqlens: _y_seqlens,
                decoder_inputs: _decoder_inputs
            })

            epoch = math.ceil(_gs / num_train_batches)

            _lr, _loss = sess.run([lr, loss], feed_dict={
                x: _x, x_seqlens: _x_seqlens, y: _y, y_seqlens: _y_seqlens,
                decoder_inputs: _decoder_inputs
            })

            if (_gs+1) % FLAGS.print_period == 0:
                logging.info("global step {}, lr {}, loss {}".format(_gs, _lr, _loss))
                # decay lr
                if len(old_loss) > 5 and _loss > max(old_loss[-5:]):
                    sess.run(learning_rate_decay_op)
                old_loss.append(_loss)

            if _gs and _gs % num_train_batches == 0:
                logging.info("epoch {} is done".format(epoch))
                ckpt_path = os.path.join(FLAGS.logdir, 'ckpt')
                saver.save(sess, ckpt_path, global_step=_gs)


def eval():
    os.makedirs(FLAGS.testdir, exist_ok=True)
    mode = tf.estimator.ModeKeys.PREDICT

    logging.info('Begin testing...')
    logging.info('Load test data...')
    test_pairs = []
    with open(FLAGS.test, 'r', encoding='utf-8') as f:
        for line in f:
            s, t = line.strip().split('\t')
            s = s.split()
            t = t.split()
            test_pairs.append((s, t))
    test_batch = data_help.MyBatch(test_pairs, FLAGS.batch_size, token2idx)
    num_test_batches = test_batch.num_batch
    num_test_samples = test_batch.num_samples

    logging.info('Test batches: {}   Test samples: {}'.format(num_test_batches, num_test_samples))

    # model
    logging.info('Build model graph')
    x = tf.placeholder(tf.int32, shape=(None, None), name='x')
    x_seqlens = tf.placeholder(tf.int32, shape=(None,), name='x_seqlens')
    y = tf.placeholder(dtype=tf.int32, shape=(None, None), name='y')
    decoder_inputs = tf.placeholder(dtype=tf.int32, shape=(None, None), name='decoder_inputs')
    y_seqlens = tf.placeholder(dtype=tf.int32, shape=(None,), name='y_seqlens')

    with tf.variable_scope('encoder'):
        enc_emb = tf.nn.embedding_lookup(embed, x)
        outputs, state, sequence_length = encoder.encode(enc_emb, x_seqlens, mode=mode)
    with tf.variable_scope('decoder'):
        start_tokens = tf.fill([FLAGS.batch_size], data_help.SOS_ID)
        end_token = data_help.EOS_ID
        target_ids, _, target_length, _ = decoder.dynamic_decode_and_search(
            embed,
            start_tokens,
            end_token,
            vocab_size=FLAGS.vocab_size,
            beam_width=1,
            memory=outputs,
            initial_state=state,
            memory_sequence_length=x_seqlens)

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
        # load ckpt
        logging.info('Loading checkpoint')
        if FLAGS.ckpt_path != '':
            ckpt_path = os.path.join(FLAGS.logdir, FLAGS.ckpt_path)
        else:
            ckpt_path = tf.train.latest_checkpoint(FLAGS.logdir)
        logging.info('Using checkpoint {}'.format(ckpt_path))
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_path)
        logging.info('Inference...')
        hypotheses = []
        for _ in range(num_test_batches):
            # get feed data
            batch = test_batch.get_next()
            _x, _x_seqlens, _y, _decoder_inputs, _y_seqlens = test_batch.process_batch(batch)

            h = sess.run(target_ids, feed_dict={
                x: _x, x_seqlens: _x_seqlens, y: _y, y_seqlens: _y_seqlens,
                decoder_inputs: _decoder_inputs
            })

            h = h[:, 0, :]
            hypotheses.extend(h.tolist())
        hypotheses = my_utils.postprocess(hypotheses, idx2token)
        hypotheses = hypotheses[:num_test_samples]
    logging.info('Writing to result...')
    with open(os.path.join(FLAGS.testdir, 'result'), 'w', encoding='utf-8') as fw:
        fw.write('\n'.join(hypotheses))
    logging.info('Inference Done.')

    # calculate bleu
    logging.info('Calculating bleu score')
    golden_file = os.path.join(FLAGS.testdir, 'gloden_temp')
    predict_file = os.path.join(FLAGS.testdir, 'result')
    bleu_file = os.path.join(FLAGS.testdir, 'bleu')
    with open(golden_file, 'w', encoding='utf-8') as fw:
        for s, t in test_pairs:
            fw.write(' '.join(t) + '\n')
    os.system('perl multi-bleu.perl {} < {} > {}'.format(golden_file, predict_file, bleu_file))
    logging.info('Bleu file is: {}'.format(bleu_file))
    with open(bleu_file) as f:
        logging.info(f.read())


if FLAGS.is_train:
    train()
else:
    eval()

