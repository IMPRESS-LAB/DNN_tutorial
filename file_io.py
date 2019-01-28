from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmmp/data/")


n_ephocs = 40
batch_size = 100

for epoch in range(n_ephocs):
    for iteration in range(mnist.train.num_examples // batch_size):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        # sess.run(training_op, feed_dict={})

