import numpy as np
import tensorflow as tf

from discriminator import Discriminator

from experiments.utils import attach_debug_op

class GenericImageCNN(Discriminator):
    """Generic CNN architecture from TensorFlow CIFAR10 tutorial.
    https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/
    cifar10.py
    """
    def _variable_with_weight_decay(self, name, shape, stddev, wd):
        """This is a simplified version of the one in the tutorial."""
        dtype = tf.float32
        init  = tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
        return tf.get_variable(name, shape, initializer=init, dtype=dtype)
      
    def _get_logits_op(self, images, n_classes=10, **kwargs):
        """Build the CIFAR-10 model"""
        # conv1
        with tf.variable_scope('conv1') as scope:
            num_channels = self.dims[2]
            kernel_size = [5,5,num_channels,64]
            kernel = self._variable_with_weight_decay(
                'weights', shape=kernel_size, stddev=5e-2, wd=0.0
            )
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            b = tf.get_variable('biases', [64],
                initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            pre_activation = tf.nn.bias_add(conv, b)
            conv1 = tf.nn.relu(pre_activation)

        # pool1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
            padding='SAME', name='pool1')

        # norm1
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
            name='norm1')

        # conv2
        with tf.variable_scope('conv2') as scope:
            kernel = self._variable_with_weight_decay(
                'weights', shape=[5, 5, 64, 64], stddev=5e-2, wd=0.0
            )
            conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
            b = tf.get_variable('biases', [64],
                initializer=tf.constant_initializer(0.1), dtype=tf.float32)
            pre_activation = tf.nn.bias_add(conv, b)
            conv2 = tf.nn.relu(pre_activation)

        # norm2
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
            name='norm2')

        # pool2
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
            padding='SAME', name='pool2')

        # local3
        with tf.variable_scope('local3') as scope:
            dim = np.prod([pool2.get_shape()[i].value for i in [1,2,3]])
            reshape = tf.reshape(pool2, [-1, dim])
            weights = self._variable_with_weight_decay(
                'weights', shape=[dim, 384], stddev=0.04, wd=0.004
            )
            b = tf.get_variable('biases', [384],
                initializer=tf.constant_initializer(0.1), dtype=tf.float32)
            local3 = tf.nn.relu(
                tf.nn.bias_add(tf.matmul(reshape, weights), b)
            )

        # local4
        with tf.variable_scope('local4') as scope:
            weights = self._variable_with_weight_decay(
                'weights', shape=[384, 192], stddev=0.04, wd=0.004
            )
            b = tf.get_variable('biases', [192],
                initializer=tf.constant_initializer(0.1), dtype=tf.float32)
            local4 = tf.nn.relu(
                tf.nn.bias_add(tf.matmul(local3, weights), b)
            )

        # linear layer(WX + b),
        with tf.variable_scope('softmax_linear') as scope:
            weights = self._variable_with_weight_decay(
                'weights', [192, n_classes], stddev=1./192., wd=0.
            )
            b = tf.get_variable('biases', [n_classes],
                initializer=tf.constant_initializer(0.), dtype=tf.float32)
            softmax_linear = tf.nn.bias_add(
                tf.matmul(local4, weights), b
            )
        return softmax_linear
