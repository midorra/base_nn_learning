#from __future__ import print_function

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
LEARNING_RATE = 0.01
TRAINING_EPOCHS = 25
BATCH_SIZE = 100
DISPLAY_STEP = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    print 'Debug ===========> Session start'
    # Training cycle
    for epoch in range(TRAINING_EPOCHS):

        print 'DEBUG ===> '
        print ' *** epoch = ' + str(epoch)
        
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/BATCH_SIZE)
        print 'DEBUG ===> '
        print ' *** mnist.train.num_examples = ' + str(mnist.train.num_examples)
        print ' *** BATCH_SIZE               = ' + str(BATCH_SIZE)
        print ' *** total_batch              = ' + str(total_batch)

        # Loop over all batches
        for i in range(total_batch):
            print 'DEBUG ===> '
            print ' *** batch_i        = ' + str(i)

            batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
            print 'DEBUG ===> '
            #print ' *** batch_xs       = ' + str(batch_xs)
            #print ' *** batch_ys       = ' + str(batch_ys)
            print ' *** type(batch_xs) = ' + str(type(batch_xs))
            print ' *** type(batch_ys) = ' + str(type(batch_ys))

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={ x: batch_xs, y: batch_ys})
            print 'DEBUG ===> '
            #print ' *** _       = ' + str(_)
            print ' *** c       = ' + str(c)
            #print ' *** type(_) = ' + str(_)
            print ' *** type(c) = ' + str(c)

            # Compute average loss
            avg_cost += c / total_batch
            print 'DEBUG ===> '
            print ' *** avg_cost = ' + str(avg_cost)

        # Display logs per epoch step
        if (epoch+1) % DISPLAY_STEP == 0:
            print("-------  Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
