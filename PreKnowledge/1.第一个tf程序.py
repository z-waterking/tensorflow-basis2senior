# -*- coding: utf-8 -*-#
'''
@Project    :   tensorflow-basis2senior
@File       :   1.第一个tf程序.py
@USER       :   ZZZZZ
@TIME       :   2021/4/23 11:34
'''
# # 第一个tf程序
# 
# 在这里，我们开启第一个tf程序
# 
# tf的运行方式是先构造计算图，因此，主要流程为：
# 
# 1. 将整个计算流程图完全搭建好。
# 2. 构建session
# 3. 通过session运行流程图
# 
# **清注意：** 在构建图的过程中，所有的内容对你都是不可见的，只有用session运行之后，才能看到整个数据流真实的过程。
# 
# **建议：** 在写tf代码时，尽可能将关键结点写上注释，例如变量的形状、数据类型之类的，不然回头看时会很痛苦。

import tensorflow as tf


# 通过tf.constant创建一个常量，这就是我们的图
message = tf.constant('Hello World!')


# 构建session
sess = tf.Session()


# run，看一下message结点的结果
sess.run(message)

