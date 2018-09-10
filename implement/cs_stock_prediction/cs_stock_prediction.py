# -*- coding: utf-8 -*-
from cs_taiwan_stock_crawler.cs_taiwan_stock_crawler import cs_taiwan_stock_crawler
import tensorflow as tf
import numpy as np

_cs_taiwan_stock_crawler = cs_taiwan_stock_crawler()
labels, samples = _cs_taiwan_stock_crawler.crawl_samples()

x = tf.placeholder(tf.float32, [None, 2])
y = tf.placeholder(tf.float32, [None, 1])
#x = tf.placeholder(tf.float32, samples.shape)
#y = tf.placeholder(tf.float32, labels.shape)

W = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1]))
prediction = tf.matmul(x, W) + b

#loss = tf.reduce_mean(tf.square(y-prediction))
loss = tf.reduce_mean(tf.abs(y-prediction))

#train_op = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
train_op = tf.train.AdadeltaOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for step in range(10):
        if step % 1 == 0:
            print(step)
            print("W", sess.run(W))
            print("b", sess.run(b))
            print("labels", labels)
            print("prediction", sess.run(prediction, feed_dict={x:samples}))
            print("loss", sess.run(loss, feed_dict={x:samples, y:labels}))

        sess.run(train_op, feed_dict={x:samples, y:labels})

print("done")
#print(labels)