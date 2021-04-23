# -*- coding: utf-8 -*-#
'''
@Project    :   tensorflow-basis2senior
@File       :   5.名称空间与变量空间.py
@USER       :   ZZZZZ
@TIME       :   2021/4/23 17:21
'''
# # 名称空间和变量
#
# ## 名称空间
#
# 在模型简单的时候，所有的问题都不会暴露出来。
#
# 想象一个有N(>=100)个参数的模型，如果不用名称空间，那么你起的名字大概率会是难以分辨，最终导致难以维护
#
# 还记得之前提过的变量名字吗？这里的名称空间等于给每一个变量名字加上一个前缀（扣上一个帽子），使得它们可以分门别类。
#
# ## 变量空间
#
# 其次，对于许多变量，是需要重复使用的。
#
# 因此，变量空间除了起到区别变量的作用，在同一个变量空间下的变量是可以根据name重复/跨函数使用的。
#
# ## 区分
# 1. tf.Variable
# 2. tf.get_variable
# 3. tf.name_scope
# 4. tf.variable_scope

import tensorflow as tf
# ## 创建一个变量
#
# 当大家都从0创建时，tf.Variable和tf.get_variable是完全一样的。
#
# 只不过get_variable需要提供名称。

# 这里命名冲突时，tf.Variable自动处理冲突问题
v1 = tf.Variable(tf.constant(1.0, shape = [1]), name = 'v')
v2 = tf.Variable(tf.constant(1.0, shape = [1]), name = 'v')

print('v1', v1)
print('v2', v2)
# tf.get_variable在没有设置命名空间reuse的情况下变量命名冲突时报错

# 这里第一次运行时，等于是新建一个变量，不会报错
# 你再运行一次，就会报错。原因是'v2'这个变量名缩在的名称空间并不是可重用的
v3 = tf.get_variable(name = 'a', shape = [1], initializer=tf.constant_initializer(1.0))
v4 = tf.get_variable(name = 'a', shape = [1], initializer=tf.constant_initializer(1.0))

print('v3', v3)
print('v4', v4)
# ## name_scope
#
# tf.name_scope没有**reuse**功能
#
# tf.get_variable命名不受它影响，并且命名冲突时报错
#
# tf.Variable命名受它影响

s1 = tf.Variable(tf.constant(1.0, shape = [1]), name = 's')
with tf.name_scope('layer0'):
    s2 = tf.Variable(tf.constant(1.0, shape = [1]), name = 'ns_s')
    s3 = tf.Variable(tf.constant(1.0, shape = [1]), name = 'ns_s')
    s4 = tf.get_variable(name = 'ns_s', shape = [1], initializer=tf.constant_initializer(1.0))

# 可以看到这里的 s2 和 s3 的名字是加上了前缀的，
print('s1', s1)
print('s2', s2)
print('s3', s3)
print('s4', s4)
# ## variable_scope
#
# tf.variable_scope可以配tf.get_variable实现变量共享；reuse默认为None，有False/True/tf.AUTO_REUSE可选：
#
# * 设置reuse = None/False 时，tf.get_variable创建新变量，变量存在则报错
# * 设置reuse = True 时tf.get_variable只获取已存在的变量，变量不存在时报错
# * 设置reuse = tf.AUTO_REUSE 时tf.get_variable在变量已存在则自动复用，不存在则创建

with tf.variable_scope('layer1',reuse=tf.AUTO_REUSE):
    # Variable会自动新建变量
    a1 = tf.Variable(tf.constant(1.0, shape=[1]),name="vs_a")
    a2 = tf.Variable(tf.constant(1.0, shape=[1]),name="vs_a")
    # 新建一个名字为 a 的变量
    a3 = tf.get_variable("vs_a", shape=[1], initializer=tf.constant_initializer(1.0))
    # 复用了a3,因为a3是由 get_variable 创建的
    a4 = tf.get_variable("vs_a", shape=[1], initializer=tf.constant_initializer(1.0))
    print('a1', a1)
    print('a2', a2)
    print('a1==a2?',a1==a2)
    print('a3', a3)
    print('a4', a4)
    print('a3==a4?',a3==a4)


