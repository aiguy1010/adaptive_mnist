import tensorflow as tf
import numpy as np
import sys
from tensorflow.examples.tutorials.mnist import input_data


import argparse
FLAGS = None


def main(_):
    # Import data
    mnist_data = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    print(type(mnist_data))

    # Create a network with one hidden FC layer
    x = tf.placeholder(tf.float32, [None, 784])

    N1 = 1000
    W1 = tf.Variable(tf.zeros([784, N1]))
    b1 = tf.Variable(tf.zeros([N1]))
    h1 = tf.matmul(x, W1) + b1

    Wf = tf.Variable(tf.zeros([N1, 10]))
    bf = tf.Variable(tf.zeros([10]))
    y = tf.matmul(h1, Wf)

    # The target
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Training ops
    cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y) )
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # Eval
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # Optimize!
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    for i in range(10000):
        batch_xs, batch_ys = mnist_data.train.next_batch(10)
        sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})

        if i%100 == 0:
            test_xs, test_ys = mnist_data.test.next_batch(100)
            acc = sess.run(accuracy, feed_dict={x:test_xs, y_:test_ys})
            print('Accuracy after iteration {}: {}'.format(i, acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( '--data_dir',
                         type=str,
                         default='/tmp/tensorflow/mnist/input_data',
                         help='Directory to store input data' )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
