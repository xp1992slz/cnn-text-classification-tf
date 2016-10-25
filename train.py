#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
from tqdm import tqdm


# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 3.0, "L2 regularizaion lambda (default: 3.0)")
tf.flags.DEFINE_string("word2vec_path", None, "Path to word2vec file, no path then don't use")
tf.flags.DEFINE_boolean("word2vec_multi", False, "Whether to use Word2Vec multi channel")
tf.flags.DEFINE_boolean("embedding_static", False, "Whether word embedding is static or not")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 50)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_float("cross_validation", 0.3, "Percentage of data for validation/test (default: 0.3)")
tf.flags.DEFINE_float("generalization_loss_threshold", 5, "Stop training according to this threshold (default: 5)")
tf.flags.DEFINE_float("min_vocab_frequency", 2, "Minimum Frequency of a vocabulary to be considered (default: 2)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer("max_array_size", 1000, "Maximum Array Size (to prevent OOM)")

#Data Parameters
tf.flags.DEFINE_string("dataset_name", "MR", "Dataset Name")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(FLAGS.dataset_name)

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, min_frequency=FLAGS.min_vocab_frequency)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))

x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# Use FLAGS.cross_validation % of the training dataset as validation/test
n_test = int(len(x_shuffled) * FLAGS.cross_validation)
x_train, x_dev = x_shuffled[:-n_test], x_shuffled[-n_test:]
y_train, y_dev = y_shuffled[:-n_test], y_shuffled[-n_test:]

# Split validation/test set (50%/50%)

n_val = int(len(x_dev)/2)

x_valid, x_dev = x_dev[:-n_val], x_dev[-n_val:]
y_valid, y_dev = y_dev[:-n_val], y_dev[-n_val:]

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Valid/Dev split: {:d}/{:d}/{:d}".format(len(y_train), len(y_valid), len(y_dev)))
print("Number of Classes: %s" % len(y_train[0]))

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=len(y_train[0]),
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda,
            embedding_static=FLAGS.embedding_static,
            word2vec_multi=FLAGS.word2vec_multi)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", cnn.loss)
        acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

        # Validation summaries
        valid_summary_op = tf.merge_summary([loss_summary, acc_summary])
        valid_summary_dir = os.path.join(out_dir, "summaries", "valid")
        valid_summary_writer = tf.train.SummaryWriter(valid_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        # Word2Vec initiazliation
        # referenced https://github.com/dennybritz/cnn-text-classification-tf/issues/17
        if FLAGS.word2vec_path:
            # initial matrix with random uniform
            initW = np.random.uniform(-0.25,0.25,(len(vocab_processor.vocabulary_), FLAGS.embedding_dim))
            # load any vectors from the word2vec
            print("Load word2vec file {}\n".format(FLAGS.word2vec_path))
            with open(os.path.abspath(FLAGS.word2vec_path), "rb") as f:
                header = f.readline()
                vocab_size, layer1_size = map(int, header.split())
                binary_len = np.dtype('float32').itemsize * layer1_size
                for line in xrange(vocab_size):
                    word = []
                    while True:
                        ch = f.read(1)
                        if ch == ' ':
                            word = ''.join(word)
                            break
                        if ch != '\n':
                            word.append(ch)
                    idx = vocab_processor.vocabulary_.get(word)
                    if idx != 0:
                        initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                    else:
                        f.read(binary_len)

            if not FLAGS.word2vec_multi:
                print("Overriding embedding with word2vec")
                sess.run(cnn.W.assign(initW))
            else:
                print("Using 2 channel embedding with word2vec")
                sess.run(cnn.W2.assign(initW))


        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

            return loss

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            div_size = int(len(x_batch) / FLAGS.max_array_size)
            print "Dividing into %s arrays" % div_size
            x_batches = np.array_split(x_batch, div_size)
            y_batches = np.array_split(y_batch, div_size)

            total_loss = 0
            total_accuracy = 0
            for i in tqdm(range(len(x_batches))):
                x = x_batches[i]
                y = y_batches[i]

                feed_dict = {
                  cnn.input_x: x,
                  cnn.input_y: y,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)

                total_loss += loss * len(x)
                total_accuracy += accuracy * len(x)

            total_loss /= len(x_batch)
            total_accuracy /= len(x_batch)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, total_loss, total_accuracy))
            if writer:
                writer.add_summary(summaries, step)

            return loss

        def valid_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a validation set
            """
            return dev_step(x_batch, y_batch, writer)

        # Generate batches
        num_batches_per_epoch = int(len(x_train)/FLAGS.batch_size) + 1

        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        # Training loop. For each batch...

        min_valid_loss = -1

        for batch in batches:
            x_batch, y_batch = zip(*batch)

            training_loss = train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)

            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_loss = dev_step(x_dev, y_dev, writer=dev_summary_writer)

            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

            # L. Prechelt, "Early stopping-but when?" in Neural Networks: Tricks of the TradeAnonymous Springer, 1998, pp. 55-69.
            # Used Generalization Loss Method

            if current_step % num_batches_per_epoch == 0:
                print("\nValidation Set Evaluation:")
                valid_loss = valid_step(x_valid, y_valid, writer=valid_summary_writer)
                if min_valid_loss < 0 or min_valid_loss > valid_loss:
                    min_valid_loss = valid_loss

                generalization_loss = 100 * (valid_loss / min_valid_loss - 1)

                if min_valid_loss >= 0:
                    print("Generalization Loss is %s" % generalization_loss)

                if min_valid_loss >= 0 and generalization_loss > FLAGS.generalization_loss_threshold:
                    print("\nStop Training Here since generalization_loss exceeded threshold")
                    print("\nFinal Evaluation:")

                    dev_loss = dev_step(x_dev, y_dev, writer=dev_summary_writer)

                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                    break
