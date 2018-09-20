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
#cross_entropy = tf.reduce_mean( -tf.reduce_sum(y_ * tf.log(y)) )
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# train operation
#train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
# session
sess = tf.InteractiveSession()
# init variables
tf.global_variables_initializer().run()

for step in range(4001):
    # 100 images for a batch within mnist.train
    # images : batch[0] shape(100, 784), labels : batch[1] shape(100, 10)
    batch = mnist.train.next_batch(100)
    
    sess.run(train_op, feed_dict={x: batch[0], y_: batch[1]})

    # show learning process and accuracy
    if step % 400 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1]})
        print("step %d, mnist.train.images(100) accuracy %g" % (step, train_accuracy))
        
# show accuracy using mnist.test.images
test_batch = mnist.test.next_batch(100)
print("final, mnist.test.images(100) accuracy %g" % sess.run(accuracy, feed_dict={x: test_batch[0], y_: test_batch[1]}))
print("final, mnist.test.images accuracy %g" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))