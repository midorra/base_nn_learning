# coding:utf-8

import tensorflow as tf

## 定义常量
A = tf.constant(123)
B = tf.constant(456)

## 定义变量
a = tf.placeholder(tf.int32)
b = tf.placeholder(tf.int32)

## 定义加法操作
add = tf.add(a, b)
## 定义乘法操作
mul = tf.multiply(a, b)

with tf.Session() as sess:
    print sess.run(A + B)
    print sess.run(add, feed_dict={a:11, b:22})
    print sess.run(mul, feed_dict={a:11, b:22})

## 定义常量矩阵
matrix_a = tf.constant([[5.,5.]])
matrix_b = tf.constant([[7.],[9.]])

## 定义矩阵乘法
product = tf.matmul(matrix_a, matrix_b)

with tf.Session() as sess:
    print sess.run(matrix_a)
    print sess.run(matrix_b)
    print sess.run(product)

