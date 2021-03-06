import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # images, 28*28
    x = tf.placeholder(tf.float32, [None, 784])
    # labels of images
    y_ = tf.placeholder(tf.float32, [None, 10])

    # reshape images from 784 to 28*28
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # convolution layer 1
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # convolution layer 2
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # Fully Convolutional layer 1, input 28/2/2 * 28/2/2 * 64, output 1024
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    # Dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Fully Convolutional layer 2, input 1024, output 10
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # loss, cross entropy
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    
    # train operation
    #train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
    train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
    # accuracy
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # session
    sess = tf.InteractiveSession()
    # init variables
    sess.run(tf.global_variables_initializer())

    # train
    for step in range(201):
        # using mnist.train.images
        batch = mnist.train.next_batch(100)
        
        sess.run(train_op, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        
        # show learning process and accuracy
        if step % 20 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, mnist.train.images(100) accuracy %g" % (step, train_accuracy))
    
    # show accuracy using mnist.test.images
    test_batch = mnist.test.next_batch(100)
    print("final, mnist.test.images(100) accuracy %g" % sess.run(accuracy, feed_dict={x: test_batch[0], y_: test_batch[1], keep_prob: 1.0}))
    #print("final, mnist.test.images accuracy %g" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))