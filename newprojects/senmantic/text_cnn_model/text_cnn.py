import tensorflow as tf
import numpy as np
import os
# import time
import datetime
from text_cnn_model.data_helpers import batch_iter
from sklearn.metrics import f1_score

# Model Hyperparameters : 在这里用不到
# tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
# tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
# tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
# tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
# tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 2, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters : 这个参数不知道是什么意思
# tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
# tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    # 模型结构
    def model(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0,model_name="default"):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name=model_name + "W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(self.embedded_chars_expanded,W,strides=[1, 1, 1, 1],padding="VALID",name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h,ksize=[1, sequence_length - filter_size + 1, 1, 1],strides=[1, 1, 1, 1],padding='VALID',name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(model_name + "W",shape=[num_filters_total, num_classes],initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        with tf.name_scope("f1"):
            output_map = {0: -2, 1: -1, 2: 0, 3: 1}
            def getClassification(arr):
                if arr == 0:
                    return -2
                elif arr == 1:
                    return -1
                elif arr == 2:
                    return 0
                else:
                    return 1
            self.predict_output = tf.map_fn(getClassification,tf.cast(self.predictions,tf.int32))
            self.true_output = tf.map_fn(getClassification, tf.cast(self.input_y,tf.int32))

    def cnn_train(self, x_train, y_train, x_dev, y_dev, vocab_processor,name):
          with tf.Session().as_default() as sess:
              # Define Training procedure
              global_step = tf.Variable(0, name="global_step", trainable=False)
              optimizer = tf.train.AdamOptimizer(1e-3)
              grads_and_vars = optimizer.compute_gradients(self.loss)
              train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
              # Keep track of gradient values and sparsity (optional)
              grad_summaries = []
              for g, v in grads_and_vars:
                  if g is not None:
                      grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                      sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name),tf.nn.zero_fraction(g))
                      grad_summaries.append(grad_hist_summary)
                      grad_summaries.append(sparsity_summary)
              grad_summaries_merged = tf.summary.merge(grad_summaries)

              # Output directory for models and summaries
              # timestamp = str(int(time.time()))
              out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", name))
              print("Writing to {}\n".format(out_dir))

              # Summaries for loss and accuracy
              loss_summary = tf.summary.scalar("loss", self.loss)
              acc_summary = tf.summary.scalar("accuracy", self.accuracy)

              # Train Summaries
              train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
              train_summary_dir = os.path.join(out_dir, "summaries", "train")
              train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

              # Dev summaries
              dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
              dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
              dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

              # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
              checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
              checkpoint_prefix = os.path.join(checkpoint_dir, "model")
              if not os.path.exists(checkpoint_dir):
                  os.makedirs(checkpoint_dir)
              saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

              # Write vocabulary
              vocab_processor.save(os.path.join(out_dir, "vocab"))

              # Initialize all variables
              sess.run(tf.global_variables_initializer())

              def train_step(x_batch, y_batch):
                  """
                  A single training step
                  """
                  feed_dict = {
                      self.input_x: x_batch,
                      self.input_y: y_batch,
                      self.dropout_keep_prob: FLAGS.dropout_keep_prob,
                  }
                  _, step, summaries, loss, accuracy = sess.run(
                      [train_op, global_step, train_summary_op, self.loss, self.accuracy],
                      feed_dict)
                  time_str = datetime.datetime.now().isoformat()
                  print("{}:  step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                  train_summary_writer.add_summary(summaries, step)

              def dev_step(x_batch, y_batch, writer=None):
                  """
                  Evaluates model on a dev set
                  """
                  feed_dict = {
                      self.input_x: x_batch,
                      self.input_y: y_batch,
                      self.dropout_keep_prob: 1.0
                  }
                  step, summaries, loss, accuracy,predict_output,true_output = sess.run(
                      [global_step, dev_summary_op, self.loss, self.accuracy,self.predict_output,self.true_output],
                      feed_dict)
                  f1 = f1_score(list(self.true_output.eval()), list(self.predict_output.eval()), average='macro')
                  time_str = datetime.datetime.now().isoformat()
                  print("{}: step {}, loss {:g}, acc {:g},f1 {:g}".format(time_str, step, loss, accuracy,f1))
                  if writer:
                      writer.add_summary(summaries, step)

              # Generate batches
              batches = batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
              # Training loop. For each batch...
              for batch in batches:
                  x_batch, y_batch = zip(*batch)
                  train_step(x_batch, y_batch)
                  current_step = tf.train.global_step(sess, global_step)
                  # if current_step % FLAGS.evaluate_every == 0:
                  #     print("\nEvaluation:")
                  #     dev_step(x_dev, y_dev, writer=dev_summary_writer)
                  #     print("")
                  if current_step % FLAGS.checkpoint_every == 0:
                      path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                      print("Saved model checkpoint to {}\n".format(path))
              # 模型训练完成后validate
              print("\nEvaluation:")
              dev_step(x_dev, y_dev, writer=dev_summary_writer)

    def cnn_predict(self):
        print("cnn prediction...")