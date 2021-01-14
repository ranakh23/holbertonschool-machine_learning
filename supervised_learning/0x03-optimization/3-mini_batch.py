#!/usr/bin/env python3
""" Mini-Batch """
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """ trains a loaded neural network model using mini-batch GD """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(sess, load_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        loss = tf.get_collection("loss")[0]
        accuracy = tf.get_collection("accuracy")[0]
        train_op = tf.get_collection("train_op")[0]

        feed_dict_t = {x: X_train, y: Y_train}
        feed_dict_v = {x: X_valid, y: Y_valid}
        float_iterations = X_train.shape[0]/batch_size
        iterations = int(float_iterations)
        if float_iterations > iterations:
            iterations = int(float_iterations) + 1
            extra = True
        else:
            extra = False
        for epoch in range(epochs + 1):
            cost_t = sess.run(loss, feed_dict_t)
            acc_t = sess.run(accuracy, feed_dict_t)
            cost_v = sess.run(loss, feed_dict_v)
            acc_v = sess.run(accuracy, feed_dict_v)
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(cost_t))
            print("\tTraining Accuracy: {}".format(acc_t))
            print("\tValidation Cost: {}".format(cost_v))
            print("\tValidation Accuracy: {}".format(acc_v))
            if epoch < epochs:
                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)
                for step in range(iterations):
                    start = step*batch_size
                    if step == iterations - 1 and extra:
                        end = (int(step*batch_size +
                               (float_iterations - iterations + 1) *
                                batch_size))
                    else:
                        end = step*batch_size + batch_size
                    feed_dict_mini = {x: X_shuffled[start: end],
                                      y: Y_shuffled[start: end]}
                    sess.run(train_op, feed_dict_mini)
                    if step != 0 and (step + 1) % 100 == 0:
                        print("\tStep {}:".format(step + 1))
                        cost_mini = sess.run(loss, feed_dict_mini)
                        print("\t\tCost: {}".format(cost_mini))
                        acc_mini = sess.run(accuracy, feed_dict_mini)
                        print("\t\tAccuracy: {}".format(acc_mini))
        save_path = saver.save(sess, save_path)
    return save_path
