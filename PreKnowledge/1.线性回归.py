# -*- coding: utf-8 -*-#
'''
@Project    :   DeepLearning
@File       :   1.线性回归.py
@USER       :   ZZZZZ
@TIME       :   2021/4/23 10:27
'''
import tensorflow as tf
import numpy as np

# 先随机创建一条向量，做出一个线性函数: y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype("float32")
y_data = x_data * 0.1 + 0.3

print('x_data', x_data)
print('y_data', y_data)

# 通过tf去计算公式 y_data = W * x_data + b 里的 W 和 b
# 已经提前知道 W = 0.1，b = 0.3
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# 最小化均方误差
loss = tf.reduce_mean(tf.square(y - y_data))
# 梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 训练朝使得loss下降的方向进行
train = optimizer.minimize(loss)


# 开始前，初始化所有参数
init = tf.global_variables_initializer()

# 构建图
sess = tf.Session()
sess.run(init)

# 拟合这条线
for step in range(201):
  sess.run(train)
  if step % 20 == 0:
    print(step, sess.run(W), sess.run(b))
