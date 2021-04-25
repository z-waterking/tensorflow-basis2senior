# -*- coding: utf-8 -*-#
'''
@Project    :   tensorflow-basis2senior
@File       :   8.TF网络格式参考.py
@USER       :   ZZZZZ
@TIME       :   2021/4/25 17:08
'''
# !/usr/bin/env python
# coding: utf-8
# # TF网络格式参考
# 
# 前面介绍了许多tf的基本用法，包括张量的概念、tensorboard等，本节就做一套tf搭建DNN，对MNIST数据集进行分类的参考流程

import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data
# ## 数据集加载

# 加载MNIST数据集
mnist = input_data.read_data_sets("../DataSet/MNIST/")
# ## 超参定义

# 训练几轮
epochs = 10
# 一个batch的数据大小
batch_size = 50
# 迭代次数
iterations = mnist.train.num_examples // batch_size
# 学习率
lr = 0.1
# ## 网络结构参数定义

# MNIST图像输入维度
n_inputs = 28 * 28
# 隐藏层1的维度
n_hidden1 = 300
# 隐藏层2的维度
n_hidden2 = 100
# 输出层的维度
n_outputs = 10
# ## 搭建网络结构

# 输入数据的placeholder
x = tf.placeholder(tf.float32, shape=(None, n_inputs), name='x')
# label的placeholder
y = tf.placeholder(tf.int64, shape=(None), name='y')

# 搭建三层神经网络
with tf.name_scope('dnn'):
    hidden1 = tf.contrib.layers.fully_connected(x,
                                                n_hidden1,
                                                activation_fn=tf.nn.relu,
                                                scope='hidden1')
    hidden2 = tf.contrib.layers.fully_connected(hidden1,
                                                n_hidden2,
                                                activation_fn=tf.nn.relu,
                                                scope='hidden2')
    logits = tf.contrib.layers.fully_connected(hidden2,
                                               n_outputs,
                                               activation_fn=None,
                                               scope='logits')
# ## 构建loss

with tf.name_scope('loss'):
    # 计算交叉熵loss
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                             logits=logits)
    # 对一个batch内的loss求平均
    loss = tf.reduce_mean(entropy, name='loss')
    tf.summary.scalar('loss', loss)
# ## 构建训练算子

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train = optimizer.minimize(loss)
# ## 计算准确率

with tf.name_scope('metrics'):
    correct = tf.nn.in_top_k(logits, y, 1)
    acc = tf.reduce_mean(tf.cast(correct, tf.float32), name='acc')
# ## 模型训练及保存

init = tf.global_variables_initializer()
# Saver-1: 模型保存，默认保存所有参数
saver = tf.train.Saver()
# Summary-1: 将所有的summary进行merge操作
summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    # Summary-2: 定义summary_writer
    summary_writer = tf.summary.FileWriter('./logs', sess.graph)
    sess.run(init)
    for epoch in range(epochs):
        acc_train = 0.0
        for iteration in range(iterations):
            # 产生下一个batch的数据
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            # Summary-3: 运算summary_op
            _, acc_batch, summary_str = sess.run([train, acc, summary_op], feed_dict={x: x_batch, y: y_batch})
            # Summary-4: 将summary_str写出去
            summary_writer.add_summary(summary_str, iteration)
            acc_train += acc_batch
        # 验证集
        acc_eval = sess.run(acc, feed_dict={x: mnist.validation.images, y: mnist.validation.labels})
        print(epoch + 1,
              'Train_acc:',
              acc_train / iterations,
              'Eval_acc',
              acc_eval)
    # Saver-2: 将模型保存在此路径下
    saver.save(sess, './model/dnn.ckpt')
# ## 加载模型

with tf.Session() as sess:
    saver.restore(sess, 'model/dnn.ckpt')
    acc_test = acc.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels})
    print("Acc_test:{0}".format(acc_test))







