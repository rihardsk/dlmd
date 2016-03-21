# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import time
import os
import datetime
from notmnist_conv import inference, training, evaluation, do_eval, loss, batch_size
from notmnist_input import train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels


graph = tf.Graph()
with graph.as_default():

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    images_initializer = tf.placeholder(tf.float32, shape=train_dataset.shape, name="inputs")
    labels_initializer = tf.placeholder(tf.int32, shape=train_labels.shape, name="labels")

    input_images = tf.Variable(images_initializer, trainable=False, collections=[])
    input_labels = tf.Variable(labels_initializer, trainable=False, collections=[])

    image, label = tf.train.slice_input_producer([input_images, input_labels])
    # image = tf.reshape(image, (image_size, image_size, 1))
    # image = tf.image.per_image_whitening(image)
    # image = tf.reshape(image, (image_size * image_size,))
    image_batch, label_batch = tf.train.batch([image, label], batch_size, num_threads=4)

    keep1_prob = tf.placeholder(tf.float32, name="keep_prob_1")
    # keep2_prob = tf.placeholder(tf.float32, name="keep_prob_2")
    # keep3_prob = tf.placeholder(tf.float32, name="keep_prob_3")
    # keep_probs = [keep1_prob, keep2_prob, keep3_prob]
    keep_probs = [keep1_prob]

    logits_op = inference(image_batch, keep_probs)

    loss_op = loss(logits_op, label_batch)

    train_op = training(loss_op)

    evaluate_op = evaluation(logits_op, label_batch)

    summary_op = tf.merge_all_summaries()

    valid_accuracy_ph = tf.placeholder(tf.float32, name="validation_accuracy_ph")
    batch_accuracy_ph = tf.placeholder(tf.float32, name="batch_accuracy_ph")
    validation_summary_op = tf.scalar_summary("accuracy/validation", valid_accuracy_ph)
    batch_summary_op = tf.scalar_summary("accuracy/training", batch_accuracy_ph)
    accuracy_summary_op = tf.merge_summary([validation_summary_op, batch_summary_op])

    saver = tf.train.Saver(tf.all_variables())


num_steps = 300001

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    session.run(input_images.initializer, feed_dict={images_initializer: train_dataset})
    session.run(input_labels.initializer, feed_dict={labels_initializer: train_labels})

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coord)

    wait_for_improvement = 20
    steps_without_improvement = 0
    print("Initialized")
    best_accuracy = -float("inf")

    savedir = os.path.join("saved_models", str(datetime.datetime.now()))
    savepath = os.path.join(savedir, "best.ckpt")
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    summary_writer = tf.train.SummaryWriter(savedir, graph_def=session.graph_def)

    starttime = time.time()
    stepstart = starttime
    for step in range(num_steps):
        # Generate a minibatch.
        # batch_data = get_batch(train_dataset, step)
        # batch_labels = get_batch(train_labels, step)
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        # feed_dict = {images_initializer: batch_data, labels_initializer: batch_labels, keep1_prob: 0.9, keep2_prob: 0.8, keep3_prob: 0.7}
        feed_dict = {keep1_prob: 0.5}
        # feed_dict = {input_images : batch_data, input_labels : batch_labels, keep1_prob: 1, keep2_prob: 1, keep3_prob: 1}
        _, l = session.run([train_op, loss_op], feed_dict=feed_dict)
        if step % 100 == 0:
            summary_str = session.run(summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
        if step % 500 == 0:
            validation_accuracy = do_eval(session, evaluate_op, image_batch, label_batch, keep_probs, valid_dataset, valid_labels)
            train_accuracy = do_eval(session, evaluate_op, image_batch, label_batch, keep_probs, train_dataset, train_labels)

            summary_str = session.run(accuracy_summary_op, feed_dict={valid_accuracy_ph: validation_accuracy,
                                                                      batch_accuracy_ph: train_accuracy})
            summary_writer.add_summary(summary_str, step)

            stepend = time.time()
            print("Minibatch loss at step %d: %f in %.2fs" % (step, l, stepend - stepstart))
            print("Minibatch accuracy: %.1f%%" % train_accuracy)
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
            stepstart = stepend

    saver.restore(session, savepath)
    print("\nTest accuracy: %.1f%%" % do_eval(session, evaluate_op, image_batch, label_batch, keep_probs, test_dataset, test_labels))
    print("Total time %.2fs" % (time.time() - starttime))

    coord.request_stop()
    # Wait for threads to finish.
    coord.join(threads)
    session.close()