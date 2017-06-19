import tensorflow as tf
import numpy as np
import sys
from sklearn import cross_validation

def random_data(samples, std, m=0, b=0):
    noise = np.random.normal(0, std, size=[samples])
    X = np.arange(samples)
    Y = (X * m) + b + noise
    return (X, Y)

# Generate data to fit
m = 5
b = 500
std_dev = 10
X, y_ = random_data(1000, std_dev, m, b)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y_, test_size=0.2)
print(y_train.shape)

# Build the simple op graph for y=m*x+b
m_op = tf.Variable(np.random.normal(0, 5), name='m')
b_op = tf.Variable(np.random.normal(0, 10), name='b')
x_op = tf.placeholder(tf.float32, name='x')
y_op = m_op * x_op + b_op

# loss + training
target_op = tf.placeholder(tf.float32, name='target')
loss = tf.reduce_mean((target_op-y_op)**2, name='loss')
train_step = tf.train.AdamOptimizer(learning_rate=1).minimize(loss)

# Train!
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
print('======== Actual Values ========')
print('m={}, b={}, std_dev={}'.format(m, b, std_dev))
print('')
print('======== Learned Values ========')
for i in range(5000):
    # Eval
    if i % 50 == 0:
        print("{}: loss={}, m={}, b={}".format(i, loss.eval(feed_dict={x_op:X_test, target_op:y_test}), m_op.eval(), b_op.eval()))

    # Fit
    sess.run(train_step, feed_dict={x_op:X_train, target_op:y_train})
