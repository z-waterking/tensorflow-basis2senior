# -*- coding: utf-8 -*-#
'''
@Project    :   tensorflow-basis2senior
@File       :   3.Tensor的基本运算.py
@USER       :   ZZZZZ
@TIME       :   2021/4/23 14:40
'''

# # 基本数学操作
# 首先是对向量的 +,-,*,/,乘方，根号 等操作
# 
# 如果想对向量与标量（单个值）进行操作，直接用python的语法操作即可


import tensorflow as tf

sess = tf.Session()

v_1 = tf.constant([1, 2, 3, 4], dtype = tf.float32)
v_2 = tf.constant([5, 6, 7, 8], dtype = tf.float32)

# 看一下v_1的内容：这里标识了它是一个常量及它的shape和数据类型
print(v_1)

# 向量相加
v_add = tf.add(v_1, v_2)

# 向量相减
v_sub = tf.subtract(v_1, v_2)

# 向量对应元素相乘
v_mul = tf.multiply(v_1, v_2)

# 向量相除
v_div = tf.div(v_2, v_1)

# 向量的乘方
v_pow = tf.pow(v_1, 2)

# 向量的根号，这里必须得是浮点数
v_sqrt = tf.sqrt(v_2)

# 可以一把将所有的向量全部打印
need_print_tensors = [v_add, v_sub, v_mul, v_div, v_pow, v_sqrt]

print(sess.run(need_print_tensors))
# # 矩阵运算
# 
# 向量的基本运算告一段落。接下里进入矩阵运算。
# 
# 矩阵的按元素操作与向量是一致的。可参考上文。
# 
# 不同点在于矩阵乘法：
# 
# * **multiply** 为点乘，是矩阵中的逐元素相乘
# * **matmul** 为矩阵乘法，与线性代数中的矩阵乘法一致，要求 左 矩阵的 列 等于 右 矩阵的 行


# 先建立两个最基本的矩阵
# 为了方便，直接指定比较简单的值
matrix_a = tf.constant([
    [1, 2],
    [3, 4]
])

matrix_b = tf.constant([
    [0, 1],
    [1, 0]
])

# 注意这里是 **matmul** 而不是 multiply，因此这里是矩阵乘法
matrix_mul = tf.matmul(matrix_a, matrix_b)
sess.run(matrix_mul)
# # 矩阵变换
# 
# 矩阵变换是在运算过程中，对矩阵的形状、数据类型等进行变换的操作。
# 
# 矩阵变换一般发生在两个需要做运算的矩阵不匹配了，需要对其中一个进行变换，以使得满足运算条件。
# 
# **注意：** 虽然tf提供了一定的补充（兜底）操作，使得你的矩阵稍有一些不匹配它也帮你补全，正常进行运算。
# 
# 但是我们在编程时不要用，而应该显式地将每个操作都处理清楚。
# 
# 变换主要有以下几类：
# 
# 1. 矩阵的形状变换
# 2. 矩阵的转置
# 3. 矩阵的逆
# 4. 矩阵切片
# 5. 矩阵增加维度
# 6. 矩阵降低维度
# 


# 矩阵形状变换
matrix_a = tf.constant([
    [1, 2],
    [3, 4]
], dtype = tf.float32)

# shape = (2, 2)
print(matrix_a.shape)

# 1. 矩阵的形状变换
# tf.reshape操作，将一个矩阵的形状进行变换
# -1 代表这一维是根据其他维度来确定的
matrix_reshape = tf.reshape(matrix_a, (-1, 1))
print(matrix_reshape.shape)

# 2. 矩阵的转置
matrix_transpose = tf.transpose(matrix_a)
sess.run(matrix_transpose)

# 3. 矩阵的逆
inverse_matrix = tf.matrix_inverse(matrix_a)
sess.run(inverse_matrix)

# 4. 矩阵切片
# 主要用到tf.slice函数

matrix_b = tf.constant([
    [1, 2, 3, 4],
    [2, 3, 4, 5],
    [5, 6, 7, 8]
], dtype = tf.float32)

# 从x[0,1],即元素2开始抽取
begin = [0,1]  
# 从x[0,1]开始，
# 对x的第一个维度（行）抽取3个元素，再对x的第二个维度（列）抽取2个元素
# 即最后想变成 3 * 2的矩阵
size = [3,2]

matrix_slice = tf.slice(matrix_b, begin, size)
sess.run(matrix_slice)
# ## 关于维度的说明
# 
# tf中的维度想要理解，需要层层将它剥开，我们从最容易理解的二维形式开始。
# 
# 若一个tensor的值为：
# 
# ```
# [
#     [1, 2],
#     [3, 4]
# ]
# ```
# 
# 则它的维度为（2,2)，前一个2表示有两个列表，后一个2表示每个列表中有两个元素。
# 
# 若变成了
# 
# ```
# [
#     [
#         [1, 2],
#         [3, 4]
#     ]
# ]
# ```
# 
# 则它的维度为（1, 2, 2), 最前面加了个1，表示最外层有1个列表，这个列表中有两个列表，最底层的每个列表中有2个元素


# 5. 矩阵增加维度:tf.expand_dims
matrix_c = tf.constant([[1, 2]])
# 1 * 2
print('before expand',matrix_c.shape)

# 增加一个维度
matrix_expand_dim = tf.expand_dims(matrix_c, 0)
print('after expand', matrix_expand_dim.shape)

# 6. 矩阵降低维度，指定axis时会将指定维度删掉，若不指定，则会将所有为1的维度全部删掉
matrix_squeeze_axis0 = tf.squeeze(matrix_expand_dim, axis = 0)
matrix_squeeze_all = tf.squeeze(matrix_expand_dim)
print('after squeeze axis0', matrix_squeeze_axis0.shape)
print('after squeeze all', matrix_squeeze_all.shape)


