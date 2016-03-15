# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

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


def inference(x):
    # Hidden Layer.
    hidden_layer_size = 1024
    weights_h = tf.Variable(
        tf.truncated_normal([image_size * image_size, hidden_layer_size]))
    biases_h = tf.Variable(tf.zeros([hidden_layer_size]))
    hidden_out = hidden_layer_activation_fn(tf.matmul(x, weights_h) + biases_h)

    # Output Layer.
    weights = tf.Variable(
        tf.truncated_normal([hidden_layer_size, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    logits = tf.matmul(hidden_out, weights) + biases
    return logits


def loss(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, tf.cast(labels, tf.float32))
    loss = tf.reduce_mean(cross_entropy)
    return loss


def training(loss):
    # global_step = tf.Variable(0, name='global_step', trainable=False)
    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train_op = optimizer.minimize(loss)
    return train_op


def evaluation(logits, labels):
    predictions = tf.nn.softmax(logits)
    correct = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))


def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, images, labels):
    correct_count = 0
    for img_batch, label_batch in zip(get_batches(images), get_batches(labels)):
        feed_dict = {
            images_placeholder: img_batch,
            labels_placeholder: label_batch,
        }
        correct_count += sess.run(eval_correct, feed_dict=feed_dict)
    count = images.shape[0]
    accuracy = 100.0 * correct_count / count
    return accuracy


graph = tf.Graph()
with graph.as_default():

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    x = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    y_ = tf.placeholder(tf.int32, shape=(batch_size, num_labels))

    logits_op = inference(x)

    loss_op = loss(logits_op, y_)

    train_op = training(loss_op)

    evaluate_op = evaluation(logits_op, y_)

    # Predictions for the training, validation, and test data.
    # train_prediction = tf.nn.softmax(logits)
    # valid_prediction = tf.nn.softmax(
    #     tf.matmul(hidden_layer_activation_fn(tf.matmul(tf_valid_dataset, weights_h) + biases_h), weights) + biases)
    # test_prediction = tf.nn.softmax(
    #     tf.matmul(hidden_layer_activation_fn(tf.matmul(tf_test_dataset, weights_h) + biases_h), weights) + biases)


num_steps = 3001

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized")
    for step in range(num_steps):
        # Generate a minibatch.
        batch_data = get_batch(train_dataset, step)
        batch_labels = get_batch(train_labels, step)
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {x : batch_data, y_ : batch_labels}
        _, l = session.run([train_op, loss_op], feed_dict=feed_dict)
        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % do_eval(session, evaluate_op, x, y_, batch_data, batch_labels))
            print("Validation accuracy: %.1f%%" % do_eval(session, evaluate_op, x, y_, valid_dataset, valid_labels))
            print("Test accuracy: %.1f%%" % do_eval(session, evaluate_op, x, y_, test_dataset, test_labels))