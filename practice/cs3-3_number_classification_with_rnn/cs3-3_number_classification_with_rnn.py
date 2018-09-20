import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

n_inputs = 28               # image shape: 28*28
n_max_steps = 28            # time steps
n_lstm_hidden_units = 100   # neurons in hidden layer
n_classes = 10              # classes (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_max_steps * n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = tf.Variable(tf.truncated_normal([n_lstm_hidden_units, n_classes], stddev=0.1))
biases = tf.Variable(tf.constant(0.1, shape=[n_classes]))

def RNN(X, weights, biases):
    inputs = tf.reshape(X, [-1, n_max_steps, n_inputs])
    # def LSTM cell
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_lstm_hidden_units)
    
    # final_state[0]: cell state.
    # final_state[1]: hidden state
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1], weights) + biases)
    return results
    
prediction = RNN(x, weights, biases)

# loss, cross entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# train operation
#train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# accuracy
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
# session
sess = tf.InteractiveSession()
# init variables
tf.global_variables_initializer().run()

for step in range(4001):
    batch = mnist.train.next_batch(100)
    
    sess.run(train_op, feed_dict={x: batch[0], y: batch[1]})

    # show learning process and accuracy
    if step % 400 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y: batch[1]})
        print("step %d, mnist.train.images(100) accuracy %g" % (step, train_accuracy))
        
# show accuracy using mnist.test.images
test_batch = mnist.test.next_batch(100)
print("final, mnist.test.images(100) accuracy %g" % sess.run(accuracy, feed_dict={x: test_batch[0], y: test_batch[1]}))
print("final, mnist.test.images accuracy %g" % sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))