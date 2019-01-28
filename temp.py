import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data


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


class conv_layer:
    def __init__(self, filter_size, n_channels, filter_num, stride=1, initializer=None, activation=None):
        self.stride = stride

        W_shape = [filter_size, filter_size, n_channels, filter_num]

        if initializer is not None:
            init = initializer(shape=W_shape)
        else:
            init = tf.truncated_normal(shape=W_shape, stddev=0.1)

        self.W = tf.Variable(initial_value=init, dtype=tf.float32)
        self.b = tf.Variable(tf.constant(value=0.1, shape=[filter_num]))

        if activation is not None:
            self.activation = activation
        else:
            self.activation = tf.nn.relu

    def get_output(self, input):
        # shape of output : [batch_size, output_h, output_w, filter_num]
        return self.activation(tf.nn.conv2d(input, self.W, strides=[1, self.stride, self.stride, 1], padding='SAME') + self.b)


class pool_layer:
    def __init__(self, filter_size, stride=1):
        self.kernel_size = [1, filter_size, filter_size, 1]
        self.stride = [1, stride, stride, 1]

    def max_pooling(self, input):
        return tf.nn.max_pool(value=input, ksize=self.kernel_size, strides=self.stride, padding="VALID")

    def avg_pooling(self, input):
        return tf.nn.avg_pool(value=input, ksize=self.kernel_size, strides=self.stride, padding="VALID")

''' configuration '''
width = 28
height = 28
channel_num = 1

input_size = width * height * channel_num
filter_size = 3
output_size = 10

c1 = 10
c2 = 20
c3 = 10

batch_size = 100

''' read data '''
mnist = input_data.read_data_sets("/tmp/data/")

''' place-holder to store input data and label to be used in training '''
x = tf.placeholder(dtype=tf.float32, shape=[None, input_size])
y = tf.placeholder(dtype=tf.int32, shape=[None])

x_image = tf.reshape(x, [-1, width, height, channel_num])  # shape of one batch : [batch size, width, height, channel num]

''' define convolution neural network structure '''
h1 = conv_layer(filter_size=filter_size, n_channels=channel_num, filter_num=c1)
p1 = pool_layer(filter_size=2)

h2 = conv_layer(filter_size=filter_size, n_channels=c1, filter_num=c2)
p2 = pool_layer(filter_size=2)

h3 = conv_layer(filter_size=filter_size, n_channels=c2, filter_num=c3)
p3 = pool_layer(filter_size=2)

full1 = layer(n_inputs=25 * 25 * c3, n_neurons=10)

''' get output'''
h1_output = h1.get_output(x_image)  # [batch_size, 28, 28, c1]
p1_output = p1.max_pooling(h1_output)  #
# p1_dropout = tf.nn.dropout(p1_output, keep_prob=keep_prob)  # drop-out

h2_output = h2.get_output(p1_output)
p2_output = p2.max_pooling(h2_output)

h3_output = h3.get_output(p2_output)
p3_output = p3.max_pooling(h3_output)  # [batch_size, 25, 25, c3]

full1_output = full1.get_output(tf.reshape(p3_output, [-1, 25 * 25 * c3]))

''' define hyper-parameters '''
learning_rate = 0.01
n_epochs = 40
batch_size = 10

''' define loss function '''
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=full1_output)
loss = tf.reduce_mean(cross_entropy)

''' define optimizer '''
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
training_op = optimizer.minimize(cross_entropy)

''' define accuracy '''
correct = tf.nn.in_top_k(full1_output, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

''' define initializer '''
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()

    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            x_batch, y_batch = mnist.train.next_batch(batch_size)

            sess.run(training_op, feed_dict={x: x_batch, y: y_batch})

        acc_train = accuracy.eval(feed_dict={x: x_batch, y: y_batch})
        acc_val = accuracy.eval(feed_dict={x: mnist.validation.images, y: mnist.validation.labels})

        print(epoch, acc_train, acc_val)

    print("Optimization is Done!")

    acc_test = accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels})
    print("test accuracy : ", acc_test)