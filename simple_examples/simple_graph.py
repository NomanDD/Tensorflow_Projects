import os

import tensorflow as tf
import numpy as np

# import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def add_layer(inputs, in_size, out_size, n_layer, activation_funtion=None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('weight'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name + '/Weights', Weights)  # 各层网络权重，偏置的分布，用histogram_summary函数
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases  # inputs与weight 顺序不能换
        if activation_funtion is None:
            output = Wx_plus_b
        else:
            output = activation_funtion(Wx_plus_b)
        tf.summary.histogram(layer_name + '/output', output)
    return output


x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputs'):
    xs = tf.compat.v1.placeholder(tf.float32, [None, 1], name='x_in')
    ys = tf.compat.v1.placeholder(tf.float32, [None, 1], name='y_in')

l1 = add_layer(xs, 1, 10, n_layer=1, activation_funtion=tf.nn.relu)
prediction = add_layer(l1, 10, 1, n_layer=2, activation_funtion=None)

# loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
#                     reduction_indices=[1]))
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(ys - prediction))
    tf.summary.scalar('loss', loss)  # 数值如学习率，损失函数用scalar_summary函数，tf.scalar_summary(节点名称，获取的数据)
optimizer = tf.train.GradientDescentOptimizer(0.1)
with tf.name_scope('train'):
    train = optimizer.minimize(loss)

init = tf.global_variables_initializer()  # 替换成这样就好
sess = tf.Session()
sess.run(init)

# 整个图经常需要检测许许多多的值，也就是许多值需要summary operation，一个个去run来启动太麻烦了，所以就合并所有获得的值
merged = tf.compat.v1.summary.merge_all()  # 合并所有的summary data的获取函数，merge_all 可以将所有summary全部保存到磁盘，以便tensorboard显示。如果没有特殊要求，一般用这一句就可一显示训练时的各种信息了。
writer = tf.compat.v1.summary.FileWriter("logs/", sess.graph)  # 把图保存到一个路径，FileWriter从tensorflow获取summary data，然后保存到指定路径的日志文件中
for i in range(1000):
    sess.run(train, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # summary的操作对于整个图来说相当于是外设，因为tensorflow是由结果驱动的，而图的结果并不依赖于summary操作，所以summary操作需要被run
        rs = sess.run(merged,
                      feed_dict={xs: x_data, ys: y_data})  # 运行所有合并所有的图，获取summary data函数节点和graph是独立的，调用的时候也需要运行session
        writer.add_summary(rs, i)  # 把数据添加到文件中，每一次run的信息和得到的数据加到writer里面，主要是描述数据变化，所以要这样，若是只有流图，就不需要这样

        # print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
