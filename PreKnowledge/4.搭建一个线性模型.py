# -*- coding: utf-8 -*-#
'''
@Project    :   tensorflow-basis2senior
@File       :   4.搭建一个线性模型.py
@USER       :   ZZZZZ
@TIME       :   2021/4/23 16:21
'''

# # 搭建一个线性模型
# 
# 自己造一份假数据，搭建tensorflow的主要流程。
# 
# 前面已经讲了如何对tf中的主角：tensor进行一些操作，接下来带大家搭一个最最简单的线性模型
# 
# 来看看tf的运行流程


import tensorflow as tf
import numpy as np
# ## 构造数据
# 
# 直接随机生成训练数据


# 先随机创建一条向量，做出一个线性函数: y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype("float32")
y_data = x_data * 0.1 + 0.3

print('x_data', x_data)
print('y_data', y_data)
# ## 构造线性表达式
# 
# 用随机生成的数据构造线性表达式


# 通过tf去计算公式 y_data = W * x_data + b 里的 W 和 b
# 已经提前知道 W = 0.1，b = 0.3
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

b = tf.Variable(tf.zeros([1]))

y = W * x_data + b
# ## 构造误差和优化器
# 
# 使用均方误差作为损失函数
# 
# 用梯度下降法进行优化


# 最小化均方误差
loss = tf.reduce_mean(tf.square(y - y_data))

# 梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(0.5)

# 训练朝使得loss下降的方向进行
train = optimizer.minimize(loss)
# ## 构造session
# 
# 构造init算子，使得所有的变量进行初始化
# 
# 创建session


# 开始训练前，初始化所有参数
init = tf.global_variables_initializer()

# 构建图
sess = tf.Session()
sess.run(init)
# ## 进行训练
# 
# 训练的时候可以在中间打印出指标的值
# 
# **注意：**这里的train，就是那张图的对loss进行最小化的算子


# 拟合这条线
for step in range(201):
  sess.run(train)
  if step % 20 == 0:
    print(step, sess.run(W), sess.run(b))


