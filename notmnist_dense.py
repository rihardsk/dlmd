import tensorflow as tf
import numpy as np
from notmnist_input import image_size, num_labels


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


def dense_layer(x, activation, size, wd_rate=0.004):
    weights = tf.Variable(
            tf.truncated_normal(size, name="weights", stddev=0.1))
    weight_decay(weights, rate=wd_rate)
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
        dense1_out = dense_layer(x, hidden_layer_activation_fn, [image_size * image_size, 1024], wd_rate=0)
        dense1_drop = tf.nn.dropout(dense1_out, keep_probs[0])
    with tf.variable_scope("dense2") as scope:
        dense2_out = dense_layer(dense1_drop, hidden_layer_activation_fn, [1024, 512], wd_rate=0)
        dense2_drop = tf.nn.dropout(dense2_out, keep_probs[1])
    with tf.variable_scope("dense3") as scope:
        dense3_out = dense_layer(dense2_drop, hidden_layer_activation_fn, [512, 256], wd_rate=0)
        dense3_drop = tf.nn.dropout(dense3_out, keep_probs[2])
    with tf.variable_scope("dense4") as scope:
        logits = dense_layer(dense3_drop, identity_activation, [256, num_labels], wd_rate=0)

    return logits


def weight_decay(variable, rate):
    wd_op = tf.mul(tf.nn.l2_loss(variable), rate, name='weight_decay')
    tf.add_to_collection('losses', wd_op)
    return wd_op


def loss(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, tf.cast(labels, tf.float32))
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def add_loss_summary(total_loss):
    avg_op = tf.train.ExponentialMovingAverage(0.99)
    losses = tf.get_collection('losses')
    loss_avg_op = avg_op.apply(losses + [total_loss])
    for l in losses + [total_loss]:
        tf.scalar_summary(l.op.name + "/raw", l)
        tf.scalar_summary(l.op.name + "/running_avg", avg_op.average(l))
    return loss_avg_op


def training(loss):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Optimizer.
    # optimizer = tf.train.GradientDescentOptimizer(0.5)
    # optimizer = tf.train.AdamOptimizer(1e-4, global_step=global_step)
    loss_averages_op = add_loss_summary(loss)
    with tf.control_dependencies([loss_averages_op]):
        optimizer = tf.train.AdamOptimizer(1e-4)
        train_op = optimizer.minimize(loss)
    return train_op


def evaluation(logits, labels):
    predictions = tf.nn.softmax(logits)
    correct = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))


def accuracy(logits, labels):
    predictions = tf.nn.softmax(logits)
    correct = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(correct, tf.float32)) * 100


def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, keep_prob_placeholders, images, labels):
    correct_count = 0
    for img_batch, label_batch in zip(get_batches(images), get_batches(labels)):
        img_batch = np.reshape(img_batch, (-1, image_size ** 2))
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