import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# fake data
x = np.linspace(-1, 1, 200)[:, np.newaxis]          # shape (200, 1)
noise = np.random.normal(0, 0.1, size=x.shape)
y = np.power(x, 2) + noise                          # y = x^2 + noise, shape (200, 1)

# plot data
plt.scatter(x, y)
plt.pause(1)

# placeholder
tf_x = tf.placeholder(tf.float32, x.shape)     # input x
tf_y = tf.placeholder(tf.float32, y.shape)     # input y

# neural network layers
l1 = tf.layers.dense(tf_x, 10, tf.nn.relu)      # hidden layer
output = tf.layers.dense(l1, 1)                 # output layer

# train operation
loss = tf.losses.mean_squared_error(tf_y, output)                   # calculate cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.2)    # using GradientDescentOptimizer
train_op = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())     # initialize var in graph
    
    for step in range(200):
        
        # train and output
        _, prediction_loss, prediction_output = sess.run([train_op, loss, output], feed_dict={tf_x: x, tf_y: y})
        
        # plot and show learning process
        if step % 10 == 0:
            plt.cla()
            plt.scatter(x, y)
            plt.plot(x, prediction_output, 'r-', lw=5)
            plt.text(0.5, -0.2, 'Loss=%.4f' % prediction_loss, fontdict={'size': 16, 'color': 'red'})
            plt.pause(0.2)

plt.show()