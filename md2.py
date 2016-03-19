# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import time
import os

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)


print(np.bincount(test_labels), np.bincount(test_labels).std())
print(np.bincount(valid_labels), np.bincount(valid_labels).std())

image_size = 28
num_labels = 10


def reformat(dataset, labels):
    reshaped_dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    one_hot_labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return reshaped_dataset, one_hot_labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

batch_size = 128

# hidden_layer_activation_fn = tf.nn.relu # achieved 84% test accuracy
hidden_layer_activation_fn = tf.nn.elu # achieved 89.2% test
identity_activation = lambda x: x


def get_batch(data, batch_num):
    count = data.shape[0]
    num_batches = count // batch_size
    offset = (batch_num * batch_size) % count
    indices = range(offset, offset + batch_size)
    ret = np.take(data, indices, axis=0, mode="wrap")
    # print(ret)
    return ret


def get_batches(data):
    count = data.shape[0]
    num_batches = count // batch_size
    for i in xrange(num_batches):
        yield get_batch(data, i)


def dense_layer(x, activation, size):
    weights = tf.Variable(
            tf.truncated_normal(size, name="weights", stddev=0.1))
    tf.histogram_summary(weights.op.name, weights)

    biases = tf.Variable(tf.zeros([size[-1]]), name="biases")
    tf.histogram_summary(biases.op.name, biases)

    dense_out = activation(tf.matmul(x, weights) + biases)
    tf.histogram_summary(dense_out.op.name + "/activation", dense_out)
    tf.scalar_summary(dense_out.op.name + "/sparsity", tf.nn.zero_fraction(dense_out))

    return dense_out


def inference(x, keep_probs):
    # Hidden Layers.
    with tf.variable_scope("dense1") as scope:
        dense1_out = dense_layer(x, hidden_layer_activation_fn, [image_size * image_size, 1024])
        dense1_drop = tf.nn.dropout(dense1_out, keep_probs[0])
    with tf.variable_scope("dense2") as scope:
        dense2_out = dense_layer(dense1_drop, hidden_layer_activation_fn, [1024, 512])
        dense2_drop = tf.nn.dropout(dense2_out, keep_probs[1])
    with tf.variable_scope("dense3") as scope:
        dense3_out = dense_layer(dense2_drop, hidden_layer_activation_fn, [512, 256])
        dense3_drop = tf.nn.dropout(dense3_out, keep_probs[2])
    with tf.variable_scope("dense4") as scope:
        logits = dense_layer(dense3_drop, identity_activation, [256, num_labels])

    return logits


def loss(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, tf.cast(labels, tf.float32))
    loss = tf.reduce_mean(cross_entropy)
    return loss


def add_loss_summary(loss):
    avg = tf.train.ExponentialMovingAverage(0.9)
    loss_avg_op = avg.apply([loss])
    tf.scalar_summary(loss.op.name + " (raw)", loss)
    tf.scalar_summary(loss.op.name + " (running avg)", avg.average(loss))
    return loss_avg_op


def training(loss):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Optimizer.
    # optimizer = tf.train.GradientDescentOptimizer(0.5)
    # optimizer = tf.train.AdamOptimizer(1e-4, global_step=global_step)
    loss_averages_op = add_loss_summary(loss_op)
    with tf.control_dependencies([loss_averages_op]):
        optimizer = tf.train.AdamOptimizer(1e-4)
    train_op = optimizer.minimize(loss)
    return train_op


def evaluation(logits, labels):
    predictions = tf.nn.softmax(logits)
    correct = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))


def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, keep_prob_placeholders, images, labels):
    correct_count = 0
    for img_batch, label_batch in zip(get_batches(images), get_batches(labels)):
        feed_dict = {
            images_placeholder: img_batch,
            labels_placeholder: label_batch,
            keep_prob_placeholders[0]: 1.0,
            keep_prob_placeholders[1]: 1.0,
            keep_prob_placeholders[2]: 1.0,
        }
        correct_count += sess.run(eval_correct, feed_dict=feed_dict)
    count = images.shape[0]
    accuracy = 100.0 * correct_count / count
    return accuracy


graph = tf.Graph()
with graph.as_default():

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    x = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size), name="inputs")
    y_ = tf.placeholder(tf.int32, shape=(batch_size, num_labels), name="labels")
    keep1_prob = tf.placeholder(tf.float32, name="keep_prob_1")
    keep2_prob = tf.placeholder(tf.float32, name="keep_prob_2")
    keep3_prob = tf.placeholder(tf.float32, name="keep_prob_3")
    keep_probs = [keep1_prob, keep2_prob, keep3_prob]

    logits_op = inference(x, keep_probs)

    loss_op = loss(logits_op, y_)

    train_op = training(loss_op)

    evaluate_op = evaluation(logits_op, y_)

    summary_op = tf.merge_all_summaries()

    saver = tf.train.Saver(tf.all_variables())


num_steps = 300001

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    wait_for_improvement = 20
    steps_without_improvement = 0
    print("Initialized")
    best_accuracy = -float("inf")

    savedir = "saved_models"
    savepath = os.path.join(savedir, "best.ckpt")
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    summary_writer = tf.train.SummaryWriter(savedir, graph_def=session.graph_def)

    starttime = time.time()
    for step in range(num_steps):
        # Generate a minibatch.
        batch_data = get_batch(train_dataset, step)
        batch_labels = get_batch(train_labels, step)
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {x : batch_data, y_ : batch_labels, keep1_prob: 0.9, keep2_prob: 0.8, keep3_prob: 0.7}
        _, l = session.run([train_op, loss_op], feed_dict=feed_dict)
        if step % 100 == 0:
            summary_str = session.run(summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
        if step % 500 == 0:
            validation_accuracy = do_eval(session, evaluate_op, x, y_, keep_probs, valid_dataset, valid_labels)
            print("Minibatch loss at step %d: %f in %.2fs" % (step, l, time.time() - starttime))
            print("Minibatch accuracy: %.1f%%" % do_eval(session, evaluate_op, x, y_, keep_probs, batch_data, batch_labels))
            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                steps_without_improvement = 0
                saver.save(session, savepath)
                print("Validation accuracy: %.1f%% (best)" % validation_accuracy)
            else:
                steps_without_improvement += 1
                print("Validation accuracy: %.1f%%" % validation_accuracy)
            if steps_without_improvement > wait_for_improvement:
                print("Validation accuracy not improved for %i evaluations. Stopping early!" % wait_for_improvement)
                break
    saver.restore(session, savepath)
    print("\nTest accuracy: %.1f%%" % do_eval(session, evaluate_op, x, y_, keep_probs, test_dataset, test_labels))
    print("Total time %.2fs" % (time.time() - starttime))

