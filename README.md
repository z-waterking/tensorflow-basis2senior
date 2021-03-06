# tensorflow-basis2senior

## 说明

本代码库所用的python版本为Anaconda包中的2.7

tensorflow环境为：1.10

本教程假设你已经了解了关于**python**和**深度学习**的基本知识，例如numpy的使用，矩阵运算、向量，多层感知机等

## 什么是好代码？

本库的目标在于写"笨代码"。

其间有两个要素，清晰的代码结构与完善的代码文档。

写最显式的代码，~~吹最狠的NB~~

## 深度学习一般套路

1. 加载数据
2. 构造输入
3. 搭建网络
4. 计算输出
5. 计算loss
6. 反向传播
7. 打完收工

# 1. Tensorflow基础

基础部分做一个参考。随着Tensorflow版本的提升，许多方法、变量都会废弃。

| Content    | .ipynb文件  |  .py 文件 |
| ------------------ | :--------------------- | :--------------------------: |
| 1.第一个tf程序 | [第一个tf程序.ipynb](PreKnowledge/1.第一个tf程序.ipynb) | [第一个tf程序.py](PreKnowledge/1.第一个tf程序.py) |
| 2.Tensor的构建 | [Tensor的构建.ipynb](PreKnowledge/2.Tensor的构建.ipynb) | [Tensor的构建.py](PreKnowledge/2.Tensor的构建.py) |
| 3.Tensor的基本运算 | [Tensor的基本运算.ipynb](PreKnowledge/3.Tensor的基本运算.ipynb) | [Tensor的构建.py](PreKnowledge/3.Tensor的基本运算.py) |
| 4.搭建一个线性模型 | [搭建一个线性模型.ipynb](PreKnowledge/4.搭建一个线性模型.ipynb) | [搭建一个线性模型.py](PreKnowledge/4.搭建一个线性模型.py) |
| 5.名称空间与变量空间 | [名称空间与变量空间.ipynb](PreKnowledge/5.名称空间与变量空间.ipynb) | [名称空间与变量空间.py](PreKnowledge/5.名称空间与变量空间.py) |
| 6.tf读取csv数据 | [tf读取csv数据.ipynb](PreKnowledge/6.tf读取csv数据.ipynb) | [tf读取csv数据.py](PreKnowledge/6.tf读取csv数据.py) |
| 7.TensorBoard使用 | [TensorBoard使用.ipynb](PreKnowledge/7.TensorBoard使用.ipynb) | [TensorBoard使用.py](PreKnowledge/7.TensorBoard使用.py) |
| 8.TF网络格式参考使用 | [TF网络格式参考.ipynb](PreKnowledge/8.TF网络格式参考.ipynb) | [TF网络格式参考.py](PreKnowledge/8.TF网络格式参考.py) |


# 参考文献

* Tensorflow官方文档：https://www.tensorflow.org/learn

* 简单粗暴Tensorflow：https://tf.wiki/zh_hans

* 动手学深度学习：https://zh.d2l.ai/index.html

* 线性变换: https://zhuanlan.zhihu.com/p/139551097