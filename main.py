import tensorflow as tf
import numpy as np
import sys
from tensorflow.examples.tutorials.mnist import input_data


import argparse
FLAGS = None

class LayerInfo:
    def __init__(self, layer_type, layer_size, previous_layer_size, weights=None, biases=None):
        self.layer_type = layer_type
        self.layer_size = layer_size
        self.previous_layer_size = previous_layer_size

        if weights is None:
            weights = np.random.normal(size=[previous_layer_size, layer_size]).astype(np.float32)
        self.weights = weights

        if biases is None:
            biases = np.random.normal(size=[layer_size]).astype(np.float32)
        self.biases = biases

        self.weights_node = None
        self.biases_node = None
        self.output_node = None

    def build(self, last_output):
        """Updates values of self.output_node and returns the new value"""
        self.weights_node = tf.Variable(self.weights)
        self.biases_node = tf.Variable(self.biases)
        self.output_node = tf.matmul(last_output, self.weights_node) + self.biases_node
        return self.output_node

    def save_wieghts(self):
        self.weights = self.weights_node.eval()
        self.biases = self.biases_node.eval()


class AdaptiveNet:
    def __init__(self, input, target):
        # Adaption Settings
        initial_size = 100

        self.key_nodes = {
            'input': input,
            'target': target # <-- One Hot target
        }

        input_size = input.shape[1]

        # Layer format: [W(input_size, layer_size), b(layer_size)]
        self.hidden_layers = [
            LayerInfo('fc', 100, input_size)
        ]

        # Ouput Layer
        self.output_layer = LayerInfo('fc', 10 , self.hidden_layers[-1].layer_size)

    def build(self):
        """Populates self.key_nodes with tf.Ops"""
        # Build hidden layers
        last_output = self.key_nodes['input']
        for layer in self.hidden_layers:
            last_output = layer.build(last_output)

        # Build output layer
        self.key_nodes['output'] = self.output_layer.build(last_output)
        self.key_nodes['output_softmax'] = tf.nn.softmax(self.key_nodes['output'])
        self.key_nodes['loss'] = tf.nn.softmax_cross_entropy_with_logits(logits=self.key_nodes['output'], labels=self.key_nodes['target'])

        return self.key_nodes








def main(_):
    # Import data
    mnist_data = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    print(type(mnist_data))

    # Create a network with one hidden FC layer
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    net = AdaptiveNet(input=x, target=y_)
    key_nodes = net.build()

    # Training ops
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(key_nodes['loss'])

    # Eval
    correct = tf.equal(tf.argmax(key_nodes['output'], 1), tf.argmax(y_, 1))
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
