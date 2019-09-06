import os

import tensorflow as tf

from utils import (get_total_param_num,focal_loss)


class Model_CNN(object):
    def __init__(self, hparams):
        self.hparams = hparams

    def is_training(self):
        return self.hparams.mode == 'train'

    def build(self):
        self.setup_input_placeholders()
        self.setup_embedding()
        self.setup_conv_pooling()
        self.setup_clf()

        self.params = tf.trainable_variables()
        self.ema = tf.train.ExponentialMovingAverage(decay=0.9999)

        if self.hparams.mode in ['train', 'eval']:
            self.setup_loss()
        if self.hparams.mode == 'train':
            self.setup_training()
            self.setup_summary()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def init_model(self, sess, initializer=None):
        if initializer:
            sess.run(initializer)
        else:
            sess.run(tf.global_variables_initializer())

    def save_model(self, sess, global_step=None):
        return self.saver.save(sess, os.path.join(self.hparams.checkpoint_dir,
                                                  "model.ckpt"),
                               global_step=global_step if global_step else self.global_step)

    def restore_best_model(self, sess):
        self.saver.restore(sess, tf.train.latest_checkpoint(
            self.hparams.checkpoint_dir + '/best_dev'))

    def restore_ema_model(self, sess, path):
        shadow_vars = {self.ema.average_name(v): v for v in self.params}
        saver = tf.train.Saver(shadow_vars)
        saver.restore(sess, path)

    def restore_model(self, sess, epoch=None):
        if epoch is None:
            self.saver.restore(sess, tf.train.latest_checkpoint(
                self.hparams.checkpoint_dir))
        else:
            self.saver.restore(
                sess, os.path.join(self.hparams.checkpoint_dir, "model.ckpt" + ("-%d" % epoch)))
        print("restored model")

    def setup_input_placeholders(self):
        # self.source_tokens = tf.placeholder(tf.int32, shape=[None, None], name='source_tokens')
        self.input_x = tf.placeholder(tf.int32, [None, self.hparams.seq_len], name="input_x")
        # for training and evaluation
        if self.hparams.mode in ['train', 'eval']:
            self.target_labels = tf.placeholder(
                tf.float32, shape=[None, self.hparams.target_label_num], name='target_labels')

        self.batch_size = tf.shape(self.input_x, out_type=tf.int32)[0]

        self.sequence_length = tf.placeholder(
            tf.int32, shape=[None], name='sequence_length')

        self.global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        self.predict_token_num = tf.reduce_sum(self.sequence_length)
        self.embedding_dropout = tf.Variable(self.hparams.embedding_dropout, trainable=False)
        self.dropout_keep_prob = tf.Variable(self.hparams.dropout_keep_prob, trainable=False)

    def setup_embedding(self):
        # load pretrained embedding
        self.W = tf.Variable(
            tf.random_uniform([self.hparams.vocab_size, self.hparams.embedding_size], -1.0, 1.0))
        self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
        self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)


    def setup_conv_pooling(self):
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        filter_sizes = list(map(int, self.hparams.filter_sizes.split(",")))
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.hparams.embedding_size, 1, self.hparams.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.hparams.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.hparams.num_filters - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.hparams.num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.hparams.dropout_keep_prob)

    def setup_clf(self):
        with tf.variable_scope("classification", reuse=tf.AUTO_REUSE) as scope:
            l2_loss = tf.constant(0.0)
            with tf.name_scope("output"):
                W = tf.get_variable(
                    "W",
                    shape=[self.hparams.num_units, self.hparams.target_label_num],
                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[self.hparams.target_label_num]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
                print("scores shape:"+str(self.scores.shape))
                self.predictions = tf.argmax(self.scores, 1, name="predictions")
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.target_labels, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def setup_loss(self):
        l2_loss = tf.constant(0.0)
        # Calculate mean cross-entropy loss
        # with tf.name_scope("loss"):
        #     if self.hparams.focal_loss > 0:
        #         self.gamma = tf.Variable(self.hparams.focal_loss, dtype=tf.float32, trainable=False)
        #         label_losses = focal_loss(self.target_labels, self.final_logits, self.gamma)
        #     else:
        #         losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.target_labels, labels=self.target_labels)
        #         self.loss = tf.reduce_mean(losses) + self.hparams.l2_reg_lambda * l2_loss
        # self.losses = label_losses
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.target_labels)
        self.losses = tf.reduce_mean(losses) + self.hparams.l2_reg_lambda * l2_loss

    def setup_summary(self):
        self.summary_writer = tf.summary.FileWriter(
            self.hparams.checkpoint_dir, tf.get_default_graph())
        tf.summary.scalar("train_loss", self.losses)
        tf.summary.scalar("learning_rate", self.learning_rate)
        tf.summary.scalar("accuracy", self.accuracy)
        # tf.summary.scalar('gN', self.gradient_norm)
        # tf.summary.scalar('pN', self.param_norm)
        self.summary_op = tf.summary.merge_all()

    def setup_training(self):
        # learning rate decay
        if self.hparams.decay_schema == 'exp':
            self.learning_rate = tf.train.exponential_decay(self.hparams.learning_rate, self.global_step,
                                                            self.hparams.decay_steps, 0.96, staircase=True)
        else:
            self.learning_rate = tf.Variable(
                self.hparams.learning_rate, dtype=tf.float32, trainable=False)

        params = self.params
        if self.hparams.l2_reg_lambda > 0:
            l2_loss = self.hparams.l2_reg_lambda * tf.add_n(
                [tf.nn.l2_loss(p) for p in params if ('predict_clf' in p.name and 'bias' not in p.name)])
            self.losses += l2_loss

        get_total_param_num(params)

        self.param_norm = tf.global_norm(params)

        # gradients = tf.gradients(self.losses, params, colocate_gradients_with_ops=True)
        # clipped_gradients, _ = tf.clip_by_global_norm(
        #     gradients, self.hparams.max_gradient_norm)
        # self.gradient_norm = tf.global_norm(gradients)
        # opt = tf.train.RMSPropOptimizer(self.learning_rate)
        # train_op = opt.apply_gradients(
        #     zip(clipped_gradients, params), global_step=self.global_step)
        # with tf.control_dependencies([train_op]):
        #     train_op = self.ema.apply(params)
        # self.train_op = train_op
        # Define Training procedure
        # global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(self.losses)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        self.train_op = train_op

    def train_clf_one_step(self, sess, source, lengths, targets, add_summary=False, run_info=False):
        feed_dict = {}
        feed_dict[self.input_x] = source
        feed_dict[self.sequence_length] = lengths
        feed_dict[self.target_labels] = targets
        if run_info:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            _, batch_loss, summary, global_step, accuracy, token_num, batch_size = sess.run(
                [self.train_op, self.losses, self.summary_op, self.global_step, self.accuracy, self.predict_token_num,
                 self.batch_size],
                feed_dict=feed_dict,
                options=run_options,
                run_metadata=run_metadata)

        else:
            _, batch_loss, summary, global_step, accuracy, token_num, batch_size = sess.run(
                [self.train_op, self.losses, self.summary_op, self.global_step, self.accuracy, self.predict_token_num,
                 self.batch_size],
                feed_dict=feed_dict
            )
        if run_info:
            self.summary_writer.add_run_metadata(
                run_metadata, 'step%03d' % global_step)
            print("adding run meta for", global_step)

        if add_summary:
            self.summary_writer.add_summary(summary, global_step=global_step)
        return batch_loss, global_step, accuracy, token_num, batch_size

    def eval_clf_one_step(self, sess, input_x, lengths, targets):
        feed_dict = {}
        feed_dict[self.input_x] = input_x
        feed_dict[self.sequence_length] = lengths
        feed_dict[self.target_labels] = targets

        batch_loss, accuracy, batch_size, predict = sess.run(
            [self.losses, self.accuracy, self.batch_size, self.predictions],
            feed_dict=feed_dict
        )
        return batch_loss, accuracy, batch_size, predict

    def inference_clf_one_batch(self, sess, input_x, lengths):
        feed_dict = {}
        feed_dict[self.input_x] = input_x
        feed_dict[self.sequence_length] = lengths
        predict, logits = sess.run([self.predictions, tf.nn.softmax(self.final_logits)], feed_dict=feed_dict)
        return predict, logits