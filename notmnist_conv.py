import tensorflow as tf
from notmnist_dense import dense_layer, hidden_layer_activation_fn, identity_activation, training, evaluation, weight_decay, get_batches
from notmnist_input import image_size, num_labels, num_channels
import time

patch_size = 5
depth1 = 32
depth2 = 64
num_hidden = 1024
strides = [1, 1, 1, 1]
batch_size = 32


def conv_layer(x, activation, shape, strides, wd_rate=0.004):
    weights = tf.Variable(
            tf.truncated_normal(shape, name="weights", stddev=0.1))

    biases = tf.Variable(tf.zeros([shape[-1]]), name="biases")

    conv = tf.nn.conv2d(x, weights, strides, padding='SAME')
    conv_out = activation(conv + biases)

    return conv_out


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def inference(x, keep_probs):
    filter_width = 5
    filter_height = 5
    n_hidden_units = 1024

    n_features1 = 32  # n_output_channels1
    W_conv1 = weight_variable([filter_width, filter_height, num_channels, n_features1])
    b_conv1 = bias_variable([n_features1])
    x_image = tf.reshape(x, [-1, image_size, image_size, num_channels])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    n_features2 = 64
    W_conv2 = weight_variable([filter_width, filter_height, n_features1, n_features2])
    b_conv2 = bias_variable([n_features2])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_dense1 = weight_variable([7 * 7 * n_features2, n_hidden_units])  # 7 comes from applying max pooling of size 2 twice to the input image of size 28
    b_dense1 = bias_variable([n_hidden_units])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * n_features2])
    h_dense1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_dense1) + b_dense1)

    keep_prob = keep_probs[0]
    h_dense1_drop = tf.nn.dropout(h_dense1, keep_prob)

    W_dense2 = weight_variable([n_hidden_units, num_labels])
    b_dense2 = bias_variable([num_labels])
    logits = tf.nn.softmax(tf.matmul(h_dense1_drop, W_dense2) + b_dense2)

    return logits


def loss(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, tf.cast(labels, tf.float32))
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
    return cross_entropy_mean


def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, keep_prob_placeholders, images, labels):
    print("Started eval at {}".format(time.time()))
    correct_count = 0
    for img_batch, label_batch in zip(get_batches(images, batch_size), get_batches(labels, batch_size)):
        feed_dict = {
            images_placeholder: img_batch,
            labels_placeholder: label_batch,
            keep_prob_placeholders[0]: 1.0,
        }
        correct_count += sess.run(eval_correct, feed_dict=feed_dict)
    count = images.shape[0]
    accuracy = 100.0 * correct_count / count
    print("Ended eval at {}".format(time.time()))
    return accuracy