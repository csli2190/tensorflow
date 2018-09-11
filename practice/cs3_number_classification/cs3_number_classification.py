import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# images, 28*28
x = tf.placeholder(tf.float32, [None, 784])

# Variables
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# softmax, prediction of images
y = tf.nn.softmax(tf.matmul(x, W) + b)

# labels of images
y_ = tf.placeholder(tf.float32, [None, 10])

# loss, cross entropy
cross_entropy = tf.reduce_mean( -tf.reduce_sum(y_ * tf.log(y)) )

# # train operation
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

# session
sess = tf.InteractiveSession()

# init variables
tf.global_variables_initializer().run()

for step in range(4000 + 1):
    # 100 images for a batch within mnist.train
    # images : batch_xs shape(100, 784), labels : batch_ys shape(100, 10)
    batch_xs, batch_ys = mnist.train.next_batch(100)
    
    sess.run(train_op, feed_dict={x: batch_xs, y_: batch_ys})

    # show learning process and accuracy
    if step % 400 == 0:
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(step, sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
