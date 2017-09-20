# coding:utf-8

import tensorflow as tf

a = tf.constant(123)
b = tf.constant(456)

with tf.Session() as sess:
    print (sess.run(a + b))

