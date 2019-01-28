import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/")

input_size = 28 * 28

''' define number of neurons for each layers '''
n_hidden1 = 30
n_hidden2 = 10
n_outputs = 10  # number of class to classify

''' place-holder to store input data and label to be used in training '''
x = tf.placeholder(dtype=tf.float32, shape=(None, input_size), name='x')
y = tf.placeholder(dtype=tf.int32, shape=(None), name='y')


''' define structure of layer '''
class layer:
    def __init__(self, n_inputs, n_neurons, activation=None, initializer=None):
        if initializer is not None:
            init = initializer(shape=(n_inputs, n_neurons))
        else:
            init = self.default_initializer(shape=(n_inputs, n_neurons))

        self.W = tf.Variable(initial_value=init)
        self.b = tf.Variable(initial_value=tf.zeros(shape=[n_neurons], dtype=tf.float32))

        if activation is not None:
            self.activation = activation
        else:
            self.activation = None

    def default_initializer(self, shape):
        stddev = 2 / np.sqrt(shape[0] + shape[1])

        return tf.truncated_normal(shape=shape, stddev=stddev)

    def get_output(self, input):
        if self.activation is not None:
            return self.activation(tf.matmul(input, self.W) + self.b)

        else:
            return tf.matmul(input, self.W) + self.b


''' create simple dnn structure '''
hidden1 = layer(input_size, n_hidden1, activation=tf.nn.relu)
hidden2 = layer(n_hidden1, n_hidden2, activation=tf.nn.relu)
logits = layer(n_hidden2, n_outputs)

h1_output = hidden1.get_output(x)
h2_output = hidden2.get_output(h1_output)
output = logits.get_output(h2_output)

''' define loss function '''
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=output)
loss = tf.reduce_mean(cross_entropy)

''' define hyper-parameters '''
learning_rate = 0.01

''' define optimizer '''
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

''' define accuracy '''
correct = tf.nn.in_top_k(output, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

''' define initializer '''
init = tf.global_variables_initializer()

n_epochs = 40
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for interation in range(mnist.train.num_examples // batch_size):
            x_batch, y_batch = mnist.train.next_batch(batch_size)

            sess.run(training_op, feed_dict={x: x_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={x: x_batch, y: y_batch})
        acc_val = accuracy.eval(feed_dict={x: mnist.validation.images, y: mnist.validation.labels})

        print(epoch, acc_train, acc_val)

    print("Optimization is Done!")
    
    acc_test = accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels})
    print("test accuracy : ", acc_test)