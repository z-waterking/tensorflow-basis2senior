# -*- coding: utf-8 -*-#
'''
@Project    :   tensorflow-basis2senior
@File       :   2.Tensor的构建.py
@USER       :   ZZZZZ
@TIME       :   2021/4/23 14:10
'''

# # Tensor的构建
# 
# tf中的基本单元是tensor，我们在构建整个计算图的时候，都是与tensor打交道。
# 
# 每个tensor都可以有一个**名称**，作为这个tensor的标识。
# 
# 你可以将tensor简单理解为向量（张量），与深度学习一致，你要操作的就是向量。
# 
# **矩阵**也是向量的一种！！！
# 
# 考虑以下情形：
# 
# 1. 普通向量a = [1, 2, 3]，我们说它是一个向量。
# 2. 嵌套向量b = [[1], [2], [3]],我们说它是一个3 * 1的矩阵。
# 
# 以上两种在tf中没有任何区别，统一称作tensor。只不过形状不同而已。
# 
# 既然大家都是向量，那么一些基本运算都是可以支持的。
# 
# 本节就带大家看看如何在tf中使用这些招式。


import tensorflow as tf

sess = tf.Session()
# # 常量
# tensorflow 如何构建常量呢？有以下几种方式：
# 
# 1. 从列表构建常量
# 2. 构建指定形状0常量
# 3. 构建指定形状1变量
# 4. 从一定范围内生成一个等差序列
# 5. 创建随机常量
# 6. 创建单位矩阵
# 7. 从已有常量创建一个常量
# 8. 填充张量
# 
# 总之，有什么需求直接百度就行了，这里仅列出几种常用的。


# 1.构建常量
constant = tf.constant([1, 2, 3])

# 2.构建10 * 10的所有元素为0的常量
zeros = tf.zeros([10, 10])

# 3.构建10 * 10的所有元素为1的常量
ones = tf.ones([10, 10])

# 4.在一定范围内生成一个从初值到终值等差排布的序列
lines = tf.linspace(2.0, 5.0, 5)

# 5.创建满足分布的随机量
randoms = tf.random_normal([2, 3], mean = 2.0, stddev = 4, seed = 12)

# 6.创建一个单位矩阵
eye = tf.eye(5)

# 7.从已有常量创建一个常量
randoms_copy = tf.zeros_like(randoms)

# 8.填充张量
fill_tensor = tf.fill([2, 3], 5.0)

sess.run(constant)
sess.run(zeros)
sess.run(ones)
sess.run(lines)
sess.run(randoms)
sess.run(eye)
sess.run(randoms_copy)
sess.run(fill_tensor)
# # 变量
# 
# tensorflow也可以构造变量，变量顾名思议，就是在图的运行过程中可以改变的量
# 
# 变量构造时，需要对其进行初始化，一般都随机初始化。
# 
# 变量可以用作tf计算图中的权重，最终通过梯度下降，将其更新到合适的值
# 


# 构造权重
weights = tf.Variable(tf.random_normal([100, 100], stddev = 2), name = 'weights')
# 构造偏置
bias = tf.Variable(tf.zeros(100), name = 'bias')
# 变量虽然定义了，且指明了初始化方法，但实际上还没有进行初始化。
# 
# 需要显示地对所有变量进行初始化


# 可以直接对变量本身进行操作,操作后就可以看到这个变量的值
sess.run(weights.initializer)
sess.run(weights)

# 当然也可以一把将所有的变量进行初始化
initial_op = tf.global_variables_initializer()
sess.run(initial_op)
# **只有初始化之后才能打印出变量的值**


sess.run(weights)
sess.run(bias)
# # 占位符
# 
# 想象一下，整张图里面的数据怎么流入？
# 
# 由于要先搭建图形，而这时候我们根本就没有获得数据，那么应该让谁来充当这个数据角色呢？
# 
# 这时候就要用到占位符


# 定义一个占位符
input_data = tf.placeholder(dtype = tf.float32,shape=None,name=None)

# 构造一个图
y = 2 * input_data

# 随机一些数据,并将它的值拿出来
data = tf.random_uniform([4, 5], 10)
data = sess.run(data)

# 现在如何run出y的值呢？
# 直接去run(y),而由于y依赖了一个placeholder，因此需要对它进行数据填充
# 用feed_dict = {'placeholder变量':'数据'} 参数对数据进行输入
sess.run(y, feed_dict = {input_data:data})