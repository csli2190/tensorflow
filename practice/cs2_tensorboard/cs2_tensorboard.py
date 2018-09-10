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
with tf.variable_scope('Inputs'):
    tf_x = tf.placeholder(tf.float32, x.shape, name='tf_x')     # input x
    tf_y = tf.placeholder(tf.float32, y.shape, name='tf_y')     # input y

# neural network layers
with tf.variable_scope('Net'):
    l1 = tf.layers.dense(tf_x, 10, tf.nn.relu)      # hidden layer
    output = tf.layers.dense(l1, 1)                 # output layer
    
    # add to histogram summary
    tf.summary.histogram('hidden_out', l1)
    tf.summary.histogram('prediction', output)

# train operation
loss = tf.losses.mean_squared_error(tf_y, output)                   # calculate cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.2)    # using GradientDescentOptimizer
train_op = optimizer.minimize(loss)

tf.summary.scalar('loss', loss)     # add loss to scalar summary

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())     # initialize var in graph
    
    writer = tf.summary.FileWriter('./log', sess.graph)     # write to file
    merge_op = tf.summary.merge_all()                       # operation to merge all summary
    
    for step in range(200):
        
        # train and output
        _, prediction_loss, prediction_output, result = sess.run([train_op, loss, output, merge_op], feed_dict={tf_x: x, tf_y: y})
        writer.add_summary(result, step)
        
        # plot and show learning process
        if step % 10 == 0:
            plt.cla()
            plt.scatter(x, y)
            plt.plot(x, prediction_output, 'r-', lw=5)
            plt.text(0.5, -0.2, 'Loss=%.4f' % prediction_loss, fontdict={'size': 16, 'color': 'red'})
            plt.pause(0.2)

#plt.show()

# -----
# in your terminal, type this :
# $ tensorboard --logdir [your log dir]
# $ tensorboard --logdir [your log dir] --host=127.0.0.1
# $ tensorboard --logdir==training:your_log_dir --host=127.0.0.1
# open you google chrome, type the link shown on your terminal. (something like this: http://localhost:6006)