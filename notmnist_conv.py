import tensorflow as tf
from notmnist_dense import dense_layer, hidden_layer_activation_fn, identity_activation, training, evaluation, loss, weight_decay, get_batches
from notmnist_input import image_size, num_labels, num_channels

patch_size = 5
depth1 = 32
depth2 = 64
num_hidden = 1024
strides = [1, 1, 1, 1]
batch_size = 32


def conv_layer(x, activation, shape, strides, wd_rate=0.004):
    weights = tf.Variable(
            tf.truncated_normal(shape, name="weights", stddev=0.1))
    weight_decay(weights, rate=wd_rate)

    biases = tf.Variable(tf.zeros([shape[-1]]), name="biases")

    conv = tf.nn.conv2d(x, weights, strides, padding='SAME')
    conv_out = activation(conv + biases)

    return conv_out


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def inference(x, keep_probs):
    # Hidden Layers.
    out = x
    with tf.variable_scope("conv1") as scope:
        conv_out = conv_layer(out, hidden_layer_activation_fn, [patch_size, patch_size, num_channels, depth1], strides, wd_rate=0)
        out = max_pool_2x2(conv_out)
    with tf.variable_scope("conv2") as scope:
        conv_out = conv_layer(out, hidden_layer_activation_fn, [patch_size, patch_size, depth1, depth2], strides, wd_rate=0)
        out = max_pool_2x2(conv_out)
    with tf.variable_scope("dense1") as scope:
        flat = tf.reshape(out, [-1, 7 * 7 * depth2])
        dense_out = dense_layer(flat, hidden_layer_activation_fn, [7 * 7 * depth2, num_hidden], wd_rate=0)  # image_size // 4 == 7 (is this true?)
        out = tf.nn.dropout(dense_out, keep_probs[0])
    with tf.variable_scope("dense2") as scope:
        logits = dense_layer(out, identity_activation, [num_hidden, num_labels], wd_rate=0)

    return logits


def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, keep_prob_placeholders, images, labels):
    correct_count = 0
    for img_batch, label_batch in zip(get_batches(images), get_batches(labels)):
        feed_dict = {
            images_placeholder: img_batch,
            labels_placeholder: label_batch,
            keep_prob_placeholders[0]: 1.0,
        }
        correct_count += sess.run(eval_correct, feed_dict=feed_dict)
    count = images.shape[0]
    accuracy = 100.0 * correct_count / count
    return accuracy