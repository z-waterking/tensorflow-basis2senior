# -*- coding: utf-8 -*-#
'''
@Project    :   tensorflow-basis2senior
@File       :   7.TensorBoard使用.py
@USER       :   ZZZZZ
@TIME       :   2021/4/25 16:19
'''

# !/usr/bin/env python
# coding: utf-8

# # TensorBoard使用
# 
# 在tf运行过程中，有大量的参数需要我们去观测。
# 
# 例如模型运行的loss是否在收敛，学习率的大小变化，甚至网络层的参数变化。
# 
# tensorboard可以将这些指标进行可视化。
# 
# 借用之前的线性函数来进行说明。

# ## 在图中将需要看的数据加入tf.summary

import tensorflow as tf
import numpy as np

# ## 构造数据
# 
# 直接随机生成训练数据
# 先随机创建一条向量，做出一个线性函数: y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype("float32")
y_data = x_data * 0.1 + 0.3

with tf.name_scope("weights"):
    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    # 将W输出
    tf.summary.histogram("weights", W)

with tf.name_scope("bias"):
    b = tf.Variable(tf.zeros([1]))
    # 将b输出
    tf.summary.histogram("bias", b)

y = W * x_data + b

# 最小化均方误差
loss = tf.reduce_mean(tf.square(y - y_data))

# 梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(0.5)

# 训练朝使得loss下降的方向进行
train = optimizer.minimize(loss)

# 开始训练前，初始化所有参数
init = tf.global_variables_initializer()

tf.summary.scalar("loss", loss)

# 将所有的summary进行merge操作
summary_op = tf.summary.merge_all()

# 构建图
with tf.Session() as sess:
    # 第一个参数指定生成文件的目录。
    summary_writer = tf.summary.FileWriter("logs/", sess.graph)
    # 初始化变量
    sess.run(init)

    # 拟合这条线
    for step in range(201):
        sess.run(train)
        # 将summary加入运行图中
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

        if step % 20 == 0:
            print(step, sess.run(W), sess.run(b))

# ## 本地启动tensorboard进行数据的查看

# 在命令行中输入: 
# 
# > tensorboard –-logdir 'logs/'
# 
# 打开：http://localhost:6006 即可看到刚才打印出来的log



