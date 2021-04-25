# -*- coding: utf-8 -*-#
'''
@Project    :   tensorflow-basis2senior
@File       :   6.tf读取csv数据.py
@USER       :   ZZZZZ
@TIME       :   2021/4/25 15:37
'''
# !/usr/bin/env python
# coding: utf-8

# # tf读取csv数据
#
# 以[1978 年收集的波士顿房价数据集](http://lib.stat.cmu.edu/datasets/boston)为例
#
# 该数据集包括 506 个样本场景，每个房屋含 14 个特征：
# 1. CRIM：城镇人均犯罪率
# 2. ZN：占地 25000 平方英尺（1 英尺=0.3048 米）以上的住宅用地比例
# 3. INDUS：每个城镇的非零售商业用地比例
# 4. CHAS：查尔斯河（Charles River）变量（若土地位于河流边界，则为 1；否则为 0）
# 5. NOX：一氧化氮浓度（每千万）
# 6. RM：每个寓所的平均房间数量
# 7. AGE：1940 年以前建成的自住单元比例
# 8. DIS：到 5 个波士顿就业中心的加权距离
# 9. RAD：径向高速公路可达性指数
# 10. TAX：每万美元的全价值物业税税率
# 11. PTRATIO：镇小学老师比例
# 12. B：1000(Bk-0.63)2，其中 Bk 是城镇黑人的比例
# 13. LSTAT：低地位人口的百分比
# 14. MEDV：1000 美元自有住房的中位值

import tensorflow as tf

sess = tf.Session()

# 这里先定义好需要读的文件名以及对应的batch大小

DataFile = '../DataSet/boston_housing.csv'
BatchSize = 10
NumFeatures = 14

def data_generator(filenames):
    # 数据队列，用一个列表来表示
    f_queue = tf.train.string_input_producer(filenames)
    reader = tf.TextLineReader()

    # 读取其中的值
    key, value = reader.read(f_queue)

    # 指定一些默认数据
    record_defaults = [[0.0] for _ in range(NumFeatures)]

    # 用tf.decode_csv方法来解析前面读到的值
    data = tf.decode_csv(value, record_defaults=record_defaults)

    # 将5，10，12列的数据聚合起来
    features = tf.stack(tf.gather_nd(data, [[5], [10], [12]]))

    # 最后一列作为它的label
    label = data[-1]

    min_after_dequeue = 10 * BatchSize
    # 队列中的最大样本数量
    capacity = 20 * BatchSize

    # 产生BatchSize数量的样本，并进行打乱操作
    feature_batch, label_batch = tf.train.shuffle_batch([features, label],
                                                        batch_size=BatchSize,
                                                        capacity=capacity,
                                                        min_after_dequeue=min_after_dequeue)
    # 将做好的batch返回
    return feature_batch, label_batch


feature_batch, label_batch = data_generator([DataFile])

# 以上就把csv数据读入了
for i in range(5):
    print(sess.run(label_batch))