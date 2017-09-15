from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np

learning_rate = 0.01
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10

n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 128 # 2nd layer num features
n_input = 784 # MNIST data input (img shape: 28*28)

X=tf.placeholder("float",[None,n_input])

weights={
    'encoder_h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'encoder_h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    'decoder_h1':tf.Variable(tf.random_normal([n_hidden_2,n_hidden_1])),
    'decoder_h2':tf.Variable(tf.random_normal([n_hidden_1,n_input])),
}

biases={
    'encoder_h1':tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_h2':tf.Variable(tf.random_normal([n_hidden_2]))
    'decoder_h1':tf.Variable(tf.random_normal([n_hidden_1]))
    'decoder_h2':tf.Variable(tf.random_normal([n_input]))
}

def encoder(x):
    layer_1=tf.nn.sigmoid(tf.add(tf.mul(x,weights['encoder_h1']),biases['encoder_h1']))
    layer_2=tf.nn.sigmoid(tf.add(tf.mul(layer_1,weights['encoder_h2']),biases['encoder_h2']))
    return layer_2
def decoder(x):
    layer_1=tf.nn.sigmoid(tf.add(tf.mul(x,weights['decoder_h1']),biases['decoder_h1']))
    layer_2=tf.nn.sigmoid(tf.add(tf.mul(layer_1,weights['decoder_h2']),biases['decoder_h2']))
    return layer_2

encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

y_pred=decoder_op
y_true=X

cost=tf.reduce_mean(tf.pow(y_pred - y_true,2))
optimizer=tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

init=tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    total_batch=int(mnist.train.num_examples/batch_size)
    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _,c = sess,run([optimizer,cost],feed_dict={X:batch_xs})
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
            "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")
    encode_decode = sess.run(y_pred,feed_dict={X: mnist.test.images[:examples_to_show]})
