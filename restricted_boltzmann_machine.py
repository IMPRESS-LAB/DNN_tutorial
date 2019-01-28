import tensorflow as tf

""" download MNIST data """
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp', one_hot=True)


