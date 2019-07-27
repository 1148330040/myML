# encoding:gbk

from __future__ import absolute_import, division, print_function
# 这种文件必须在开头导入

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# 显示所有列

pd.set_option('display.max_columns', None)
from time import time

# Deep Learning
import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 防止深度学习过拟合的方法和ML基本一致
# 增加数据量,调小参数,添加丢弃层,正则化,数据增强,批次归一化

# DeepL分类练习
"""
close_images = keras.datasets.fashion_mnist
(X_train,train_label),(X_test,test_label) = close_images.load_data()


X_train = X_train // 55
X_test = X_test // 55


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), #将二维数组->一维数组 扁平化处理
    keras.layers.Dense(128, activation=tf.nn.relu), #tf.nn 是一个激活函数 relu是修正线性单元,数字指神经元
    keras.layers.Dense(10, activation=tf.nn.softmax) #softmax 交叉熵函数 多分类问题 一般情况需要进行独热编码
])

model.compile(
    optimizer=tf.train.AdamOptimizer(), #optimizer 优化器
    loss='sparse_categorical_crossentropy', #loss损失函数 即我所理解的平衡性函数
    metrics=['accuracy'] #评估指标 准确率(precision,recall等)
)

model.fit(X_train,train_label,epochs=5)

test_loss , test_score = model.evaluate(X_test,test_label)
print("scores",test_loss,test_score)
"""

# DeepL回归练习
'''
dataset_path = keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.dropna(inplace=True)
origin = dataset.pop('Origin')
#pop很有用 获取指定列并从dataset中删除，可以节省三行代码

dataset['USA'] = (origin==1) // 1.0
dataset['Europe'] = (origin==2) // 1.0
dataset['Japan'] = (origin==3) // 1.0

from sklearn.model_selection import train_test_split

#sns.pairplot(dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
#diag_kind将单变量图换为线形图其他为散点
train_stats = dataset.describe()

dataset_label = dataset.pop('MPG')
X_train, X_test, y_train, y_test = train_test_split(dataset, dataset_label,
                                                    test_size=0.3, random_state=123)

train_stats.pop('MPG')
train_stats = train_stats.transpose() #转置

def norm_datas(datas):
    #根据名称分别进行运算
    return (datas - train_stats['mean']) / train_stats['std']

norm_train = norm_datas(X_train)
norm_test = norm_datas(X_test)

def build_mode():
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[len(X_train.keys())]),
        keras.layers.Dense(64, activation=tf.nn.relu),  #Dense相当于添加一个连接层
        keras.layers.Dense(1) #Dense构造函数参数实例化
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001) #优化器

    #编译模型
    model.compile(loss='mean_squared_error', #损失函数
                  optimizer=optimizer, #优化器
                  metrics=['mean_absolute_error', 'mean_squared_error'] #评估指标
                  )
    return model

model = build_mode()

class get_ends(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0:
            print('')
        print('.',end='')

epoch = 1000 #迭代次数

history = model.fit(
    norm_train, y_train, batch_size=50,
    epochs=epoch, validation_split=0.2,
    verbose=0, callbacks=[get_ends()]
)


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Val Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()
# 由此图可以看到一定训练周期之后,分数基本没有变化
# 因此设定在一段周期内不产生变化就自动停止
model = build_mode()
stop_fit = tf.keras.callbacks.EarlyStopping(monitor='val_loss', #指标
                                            patience=10 #阈值
                                            )
history_2 = model.fit(
    norm_train, y_train, epochs=epoch,
    validation_split=0.2, verbose=0,
    callbacks=[stop_fit,get_ends()]
)
loss, mae, mse = model.evaluate(norm_test, y_test, verbose=0)
print("Testing set Mean Abs Error: {:5.2f} MPG,MSE : {:f},Loss : {:f}".format(mae,mse,loss))
test_predictions = model.predict(norm_test).flatten()
'''

# Model于Weight 保存获取
'''
(train, train_label), (test, test_label) = keras.datasets.mnist.load_data()

train = train[:1000].reshape(-1, 28*28) / 55
test = test[:1000].reshape(-1, 28*28) / 55

train_label = train_label[:1000]
test_label = test_label[:1000]

def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu,
                           input_shape=(784,)
                           ),
                           #kernel_regularizer=keras.regularizers.l2(0.001)),
                           #l2正则化 0.001 * value**2
        keras.layers.Dropout(0.2), #丢弃层丢弃括号内的数字比率
        keras.layers.Dense(16, activation=tf.nn.softmax, input_shape=(728,),)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    return model

model = build_model()

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

callback_check = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1 #, period=5 #每隔五个周期保存一次,可以不设置
                                                 )

model.fit(train, train_label, epochs=10,
          validation_data=(test, test_label),
          callbacks=[callback_check])

model = build_model() #定义一个新的模型不进行权重设置
#进行测试分数
loss, acc = model.evaluate(test, test_label)
#print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

model.load_weights(checkpoint_path) #继承以训练好模型的权重
#可以再次进行测试分数
#和sklearn的提取模型函数一致

model.save_weights('./check_weights')
#手动保存权值,更直接
model = build_model()
model.load_weights('./check_weights')

#直接保存模型
model.save('end_model.h5')
model = keras.models.load_model('end_model.h5')
'''

# Tensor 张量 是TensorFlow的基本数据表达方式和传输方式
'''
tf.enable_eager_execution()

#张量可以看做是一个结构包含名字(name)、维度(shape)、类型(type)
#张量与ndarray类似但是它具有以下两个特点
#张量可以由加速器内存(如GPU,TPU)支持. 这个TensorFlow会自行决定
#张量是不可改变的.
print(tf.add([1, 2, 3], [1, 2, 3]))
print(tf.add([[1, 2], [1, 2], [1, 2]], [[1, 2], [1, 2], [1, 2]])) #列表长度要求一致
#看最边缘有几层中括号就是几阶张量
print(tf.square([5]))
print(tf.reduce_sum([1, 2, 3]))
print(tf.encode_base64("hello world"))

# Operator overloading is also supported
print(tf.square(2) + tf.square(3))
#张量的形状判定和行列排列很奇怪

ndarrays = np.ones([3,3])
#ndarray转张量
tensor = tf.multiply(ndarrays, 42)
#tensor = tf.multiply(ndarrays, 42)+3
#tensor = tf.add(tensor, 3) 针对tensor本身两种相加方法
#tensor = np.add(tensor, 3) 通过numpy相加
tensor = tensor.numpy()
#tensor和ndarry之间的运算是可以直接进行,这意味着想改变tensor可以想让其转为ndarray后进行改变
a = tf.constant([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24], shape=[2,3,4])
b = tf.constant([1,2,3,4,5,6,7,8,9,10,11,12], shape=[4,3])
c = tf.tensordot(a, b, axes=1)
d = tf.tensordot(a, b, axes=2)
print(a)
print(b)
print(c)

ds_tensor = tf.data.Dataset.from_tensor_slices(np.arange(1,6))

import tempfile
_, filename = tempfile.mkstemp()
with open(filename,'w') as f:
    f.write("""line1
               line2
               line3""")

ds_file = tf.data.TextLineDataset(filename)
ds_tensor = ds_tensor.map(tf.square).shuffle(2).batch(2)
#map 数据映射 shuffle 清洗(类似洗牌) batch 数据批量读取
ds_file = ds_file.batch(2)
print("Element")

for data in ds_tensor:2
    print(data)
for data in ds_file:
    print(data)
'''

# 关于初始化神经网络权重值的函数
'''
t = tf.truncated_normal_initializer(stddev=0.01, seed=1)
v = tf.get_variable('v', [1], initializer=t)

with tf.Session() as sess:
    for i in range(1, 10, 1):
        sess.run(tf.global_variables_initializer())
        print(sess.run(v))
'''

from tensorflow import feature_column

# categorical_column_with_vocabulary_list
'''
color_data = {'color': [['R', 'R', 'G', 'CZ'], ['G', 'R', 'R', 'CX']]}  # 4行样本

builder = _LazyBuilder(color_data)
color_column = feature_column.categorical_column_with_vocabulary_list(
    'color', ['R', 'G', 'B'], dtype=tf.string, default_value=-1
)

color_column_tensor = color_column._get_sparse_tensors(builder)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    # 初始化模型参数
    session.run(tf.tables_initializer())
    # 返回初始化的操作
    print(session.run([color_column_tensor.id_tensor]))
'''

# embedding_column与share_embedding_column的内容
'''
color_data = {'color': [['G'], ['B'], ['B'], ['R'], ['ZZZ']]}  # 4行样本

color_column = feature_column.categorical_column_with_vocabulary_list(
    'color', ['R', 'G', 'B'], dtype=tf.string, default_value=-1
)

color_embeding = feature_column.embedding_column(color_column, 6, combiner='mean')
color_embeding_dense_tensor = feature_column.input_layer(color_data, [color_embeding])
builder = _LazyBuilder(color_data)
color_column_tensor = color_column._get_sparse_tensors(builder)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    print(session.run([color_column_tensor.id_tensor]))

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    print('embeding' + '_' * 40)
    print(session.run([color_embeding_dense_tensor]))
'''

# 前向传播算法的神经网络小例子
'''
batch_size = 8

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

x = tf.placeholder(tf.float32, shape=(None, 2), name='x_input')
y = tf.placeholder(np.float, shape=(None, 1), name='y_input')
# placeholder占位符 有数字就填没数字就先占着位置

a = tf.matmul(x, w1)
b = tf.matmul(a, w2)
# 使用numpy的矩阵乘法dot和matmul会因为无法和tf的占位符函数合作而直接报错
# 使用matmul的原因是前向传播算法属于全连接它的权重和输入值全都互相交错
# 产生矩阵乘法格式的传播过程因此使用矩阵乘法函数matmul
# with tf.Session() as sess:
#     end = tf.global_variables_initializer()
#     sess.run(end)
#     print(sess.run(a, feed_dict={x: [[0.2, 0.5]]}))
#     print(sess.run(w2))
#     print(sess.run(b, feed_dict={x: [[0.7, 0.9], [0.1, 0.2], [0.3, 0.5]]}))
b = tf.sigmoid(b)
# 交叉熵 分类问题的一种常用损失函数
cross_entropy = -tf.reduce_mean(
    y * tf.log(tf.clip_by_value(b, 1e-10, 1.0)) +
    (1-y) * tf.log(tf.clip_by_value(1-b, 1e-10, 1.0))
)
# clip_by_value按值裁剪, 小于1e-10的让其变为1e-10, 大于1.0的让其变为1.0
# 反向传播的优化算法
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

rdm = np.random.RandomState(seed=1) # 随机数生成器
dataset_size = 128
X_train = rdm.rand(dataset_size, 2) # 2列
print(X_train)
y_train = [[int(train_1+train_2)] for (train_1, train_2) in X_train]
# y_train = [[int(train.sum())] for train in X_train] 效果一致

with tf.Session() as sess:
    init_all = tf.global_variables_initializer()
    sess.run(init_all)
    Steps = 5000
    for i in range(Steps):
        start = (i*batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)
        sess.run(
            train_step,
            feed_dict={x: X_train[start:end], y: y_train[start:end]} # 必须指明赋予给那个变量
        )
        # X_train[start:end] 可以看到是n行两列矩阵所以相当于是x传递给a前向传播变量的
        # y_train[start:end] 可以看到是n行一列矩阵所以相当于是y传递给b前向传播变量的
        if i % 1000 == 0:
            total_cross_entropy = sess.run(
                cross_entropy, feed_dict={x: X_train, y: y_train}
            )
            # 每个1000step在所有数据上面计算交叉熵
            print("After %d steps, the predict result is %f" % (i, total_cross_entropy))
'''


# 第四章 深层神经网络
'''
x = tf.placeholder(dtype=np.float, shape=(None, 2), name='x_input')
y = tf.placeholder(dtype=np.float, shape=(None, 1), name='y_label')

w = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y_p = tf.matmul(x, w)
loss_less = 10
loss_more = 1
# loss = tf.reduce_sum(tf.where(tf.greater(y_p, y),
#                               (y_p - y) * loss_more,
#                               (y - y_p) * loss_less ))
loss = tf.reduce_mean(tf.square(y - y_p))
# 这里的意思是假如预测的值大于真实值了那么选择 (y-y_p)*loss_less查看最终利润的多寡
# 反之假如预测的值小于真实值那么选择(y_p-y)*loss_more查看最终利润的多少
# 这样要比直接通过mse均方根误差更能获取最大化利润的选择
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
rdm = np.random.RandomState(1)
data_size = 128

X_train = rdm.rand(data_size, 2)
y_train = [[x.sum() + rdm.rand()/10.0 - 0.05] for x in X_train]

batch_size = 8
with tf.Session() as sess:
    init_all = tf.global_variables_initializer()
    sess.run(init_all)
    Steps = 5000
    for i in range(Steps):
        start = (i * batch_size) % data_size
        end = min(start+batch_size, data_size)
        sess.run(
            train_step,
            feed_dict={x: X_train[start:end], y: y_train[start:end]}
        )
        if i==4999:
            print(sess.run(w))
# [[1.019347 ]
#  [1.0428089]]
'''

'''
# 正则化样例
def get_weights(shape, lambdas):
    # 先根据shape定义变量
    var = tf.Variable(tf.random_normal(shape), dtype=np.float)
    # 将正则化后的值(权重值)加入到列表内
    tf.add_to_collections(
        'losses', tf.contrib.layers.l2_regularizer(lambdas)(var)
    )
    return var


x = tf.placeholder(np.float, shape=(None, 2))
y = tf.placeholder(np.float, shape=(None, 1))

batch_size = 8
layers_dimension = [2, 10, 10, 10, 1]
layers_nums = len(layers_dimension)

cur_layer = x
# 当前层的节点数
in_dimension = layers_dimension[0]
for i in range(1, layers_nums):
    out_dimension = layers_dimension[i]  # 注意此时i为1

    # 获取当前层的正则化权重值
    weights = get_weights(shape=[in_dimension, out_dimension], lambdas=0.001)
    # 获取偏置项
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    # 设置前向传播算法
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weights) + bias)
    # 将获取当前节点个数的变量移动到下一层神经网络节点个数
    in_dimension = layers_dimension[i]

mse_loss = tf.reduce_mean(tf.square(y - cur_layer))  # 损失函数 mse均方根误差
tf.add_to_collections(
    'losses', mse_loss
)
loss = tf.add_n(tf.get_collection('losses'))
'''

'''
# 平均滑动模型
v1 = tf.Variable(10, dtype=np.float)
step = tf.Variable(10, trainable=False)

# 定义滑动平均的类,初始化衰减率为0.99以及控制述案件屡的变量step
ema = tf.train.ExponentialMovingAverage(decay=0.99, num_updates=step)
#定义列表保存更新后的滑动变量即shadow_variable
maintain_averages_op = ema.apply([v1])
with tf.Session() as sess:
    init_av = tf.global_variables_initializer()
    sess.run(init_av)
    # sess.run(tf.assign(v1, 0))
    sess.run(maintain_averages_op)
    # 只有当初始变量发生改变时,才可以对shadow_variable进行改变
    # 不然无论num_updates初始值设为多少他的值始终就是初始变量的值
    # 因为shadow_variable是为了变量服务的
    #查看初始变量和v1的shadow_variable
    print(sess.run([v1, ema.average(v1)]))
    sess.run(tf.assign(v1, 5))
    # 更新shadow_variable
    sess.run(maintain_averages_op)
    # 查看v1和shadow_variable, ema.average()
    print(sess.run([v1, ema.average(v1)]))

    sess.run(tf.assign(step, 10000))
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))

    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))
'''

# 第五章 实例:MNIST数字识别
from tensorflow.examples.tutorials.mnist import input_data
"""
'''
# mnist = input_data.read_data_sets("/path/to/MNIST_data/", one_hot=True)
# print("train_data size: ", mnist.train.num_examples)
# batch_size = 100
# train, label = mnist.train.next_batch(batch_size=batch_size)
# 输入层节点数
Input_node = 784
# 输出层节点数
Output_node = 10
# 设置神经网络的参数
# 隐藏层节点数
Layer_node = 500
Batch_size = 100 # batch
Learning_rate = 0.8 # 学习率
Learning_decay_rate = 0.99 # 学习率的衰减速率
Regularization_rate = 0.0001 # 正则化中用于限制模型复杂度(即限制模型去拟合任意点的数据)的参数
Train_steps = 2000
Moving_average_decay = 0.99 #滑动平均衰减速率
'''
'''    
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 首先判断是否存在滑动平均类
    if avg_class:
        layer1 = tf.nn.relu(
            tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1)
        )
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)
    else:
        # 说明没有滑动平均类出现直接使用现有的参数
        layer1 = tf.nn.relu(
            tf.matmul(input_tensor, weights1) + biases1
        )
        return tf.matmul(layer1, weights2) + biases2
'''
'''
def inference(input_tensor, reuse, variable_average):
    with tf.variable_scope('layer1', reuse=reuse):
        weights = tf.get_variable('weights', shape=[Input_node, Layer_node],
                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', shape=[Layer_node],
                                  initializer=tf.constant_initializer(0.0))
        if variable_average!=None:
            layer1 = tf.nn.relu(
                tf.matmul(input_tensor, variable_average.average(weights)) + variable_average.average(biases)
            )
        else:
            layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
    with tf.variable_scope('layer2', reuse=reuse):
        weights = tf.get_variable('weights', shape=[Layer_node, Output_node],
                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', shape=[Output_node],
                                  initializer=tf.truncated_normal_initializer(0.0))

        if variable_average!=None:
            layer2 = tf.matmul(layer1, variable_average.average(weights)) + variable_average.average(biases)
        else:
            layer2 = tf.matmul(layer1, weights) + biases
    return layer2

def train(mnist):
    x = tf.placeholder(tf.float32, shape=[None, Input_node], name='x-input')

    y_ = tf.placeholder(tf.float32, shape=[None, Output_node], name='y-input')

    weights1 = tf.Variable(
        tf.truncated_normal([Input_node, Layer_node], stddev=0.1)
    )
    biases1 = tf.Variable(
        tf.constant(0.1, shape=[Layer_node])
    )
    weights2 = tf.Variable(
        tf.truncated_normal([Layer_node, Output_node], stddev=0.1)
    )
    biases2 = tf.Variable(
        tf.constant(0.1, shape=[Output_node])
    )
    y = inference(x, False, None)
    # 初始条件下无需滑动平均类

    # 定义存储训练轮数的变量,trainable=False表示该变量不被训练
    global_steps = tf.Variable(2, trainable=False)

    variable_average = tf.train.ExponentialMovingAverage(
        Moving_average_decay, global_steps
    )
    varuables_averages_op = variable_average.apply(
        tf.trainable_variables() # 这个返回的是更新后的shadow_variable
    )
    averages_y = inference(x, True, variable_average)

    # 刻画损失函数-》用交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularizer = tf.contrib.layers.l2_regularizer(Regularization_rate)
    regularaztion = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularaztion
    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        Learning_rate, # 基础学习率
        global_steps,
        mnist.train.num_examples / Batch_size,
        Learning_decay_rate # 学习率衰减速度
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_steps)
    # 使用梯度下降算法作为优化器更新参数, 使用交叉熵和正则项作为损失函数, 使用global_step作为计时器
    # 模型
    with tf.control_dependencies([train_step, varuables_averages_op]):
        # control_dependencies上下文控制器将优化器和滑动平均类(用于更新shadow_variable)按先后顺序进行
        train_op = tf.no_op(name='train') # no_op意思是之行为train_step和滑动平均类之后什么都不做

    correct_prediction = tf.equal(tf.argmax(averages_y, axis=1), tf.argmax(y_, axis=1))
    # argmax() 脱胎于np.argmax() 返回axis方向的最大值的索引
    accuracy = tf.reduce_mean(tf.cast(
        correct_prediction, dtype=np.float
    ))
    # tf.cast() 用于张量的数据类型转换
    saver = tf.train.Saver() # 真的傻逼设定非得放到会话管理的上面才行太JB傻逼了
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        for i in range(Train_steps):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("validate :", validate_acc)
            xs, ys = mnist.train.next_batch(Batch_size)
            sess.run(train_op, feed_dict={x: xs, y_: ys})
        saver.save(sess, save_path="DL/model.ckpt")

        saver.restore(sess, "DL/model.ckpt")
        test_auc = sess.run(accuracy, feed_dict=test_feed)
        print("test :", test_auc)

def main(argv=None):
    mnist = input_data.read_data_sets("/path/to/MNIST_data/", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()

'''

'''
with tf.variable_scope("foo"):
    v1 = tf.get_variable('v', [1],
                         initializer=tf.constant_initializer(1.0))
with tf.variable_scope("foo", reuse=True):
    # 当reuse=True时,目的是获取变量而无法创建变量
    v2 = tf.get_variable('v', [1])
    print(v2 == v1)

with tf.variable_scope("bar"):
    v3 = tf.get_variable('x', [1],
                         initializer=tf.constant_initializer(1.0))
with tf.variable_scope("bar", reuse=True):
    v4 = tf.get_variable('x', [1])
    print(v4==v3)
with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v5 = tf.get_variable('x', [1],
                             initializer=tf.constant_initializer(1.0))
        # 当处在其他variable_scope的reuse=True时无法获取原本位置下创建的变量
        # 原因是此时变量的命名空间不一致 v4的命名空间是bar而v5的命名空间是foo/bar
        # 所以无法获取
with tf.variable_scope("", reuse=True):
    v6 = tf.get_variable("foo/bar/x", [1]) # 通过直接索引命名空间的位置来直接获取变量
    print(v6==v5)
'''

'''
v = tf.Variable(0, dtype=np.float, name='v')

ema = tf.train.ExponentialMovingAverage(0.99)
ema_allop = ema.apply(tf.global_variables())

saver = tf.train.Saver()

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(tf.assign(v, 10))
    sess.run(ema_allop) # 更新shadow_variable
    saver.save(sess, "DL/zxc.ckpt")
    print(sess.run([v, ema.average(v)]))


x = tf.Variable(0, dtype=np.float, name='x')

ema = tf.train.ExponentialMovingAverage(0.99)

saver = tf.train.Saver({'v/ExponentialMovingAverage': x})
# saver = tf.train.Saver(ema.variables_to_restore(v)) 
# 这种方法不推荐，无法将值赋给其他变量
# 且当上文中存在saver容易造成混乱
with tf.Session() as sess:
    saver.restore(sess, "DL/zxc.ckpt")
    print(sess.run(x))
'''

'''
from tensorflow.python.framework import graph_util

v1 = tf.Variable(1.0, dtype=np.float, name='v1')
v2 = tf.Variable(2.0, dtype=np.float, name='v2')
result = v1 + v2

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    graph_def = tf.get_default_graph().as_graph_def() # 序列化图,目的是可以将其导入到另外的图中
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph_def, ['add']
    )
    with tf.gfile.GFile("DL/model.pb", "wb") as f:
        f.write(output_graph_def.SerializeToString()) # 序列化字符串 目的是将序列化图转变为字符串

from tensorflow.python.platform import gfile
with tf.Session() as sess:
    with gfile.FastGFile("DL/model.pb", "rb") as f:
        graph_defs = tf.GraphDef()
        graph_defs.ParseFromString(f.read()) # 将序列化的字符串(注意是由序列化图转为的序列化字符串)转为图
    results = tf.import_graph_def(graph_defs, return_elements=['add:0']) # 将graph_defs导入到当前图中
    print(sess.run(results))
'''

# 整个流程大致可以表示为 建立图->序列化图->序列化字符串->储存->获取序列化字符串->图
'''
# 获取保存模型内的变量 
# 无法获取保存图的内容，因为图内保存的是运算过程即如何从输入层经过传播过程后得到输出层即可
reader = tf.train.NewCheckpointReader('DL/zxc.ckpt')
# 获取所有变量的列表 是一个字典
global_variables = reader.get_variable_to_shape_map()
for name in global_variables:
    print(name, global_variables[name])
    print(reader.get_tensor(name))
'''


# 完整的mnist实践
'''
Input_node = 784
Output_node = 10
Layer_node = 500

def get_weights_variable(shape, regularizer):
    weights = tf.get_variable(
        dtype=np.float, shape=shape, name='weights',
        initializer=tf.truncated_normal_initializer(stddev=0.1)
    )

    if regularizer!=None:
        tf.add_to_collections(
            'losses', regularizer(weights)
        )
    return weights

def interence(input_tensor, regularizer):
    with tf.variable_scope("layer1"):
        weights = get_weights_variable(
            [Input_node, Layer_node], regularizer
        )
        biases = tf.get_variable(
            'biases', [Layer_node],
            initializer=tf.truncated_normal_initializer(stddev=0.0)
        )
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
    with tf.variable_scope("layer2"):
        weights = get_weights_variable([Layer_node, Output_node], regularizer)
        biases = tf.get_variable(
            'biases', [Output_node],
            initializer=tf.truncated_normal_initializer(stddev=0.0)
        )
        layer2 = tf.matmul(layer1, weights) + biases
    return layer2
# 神经网络的前向传播过程和参数

Batch_size = 100
Learning_rate_base = 0.8
Learning_rate_decay = 0.99
Regularization_rate = 0.0001
Train_step = 20000
Moving_average_decay = 0.99
model_path = 'DL/'
model_name = 'end_model.ckpt'

def train(mnist):

    x = tf.placeholder(
        dtype=np.float, shape=[None, Input_node],
        name='x_input'
    )
    y_ = tf.placeholder(
        dtype=np.float, shape=[None, Output_node],
        name='y_input'
    )
    regularizer = tf.contrib.layers.l2_regularizer(Regularization_rate)
    y = interence(x, regularizer)
    # 获取预测值，为接下来交叉熵进行数据提供

    # ---------------------------
    # 开始参数的调整
    # 定义不可训练变量,每一轮batch_size记录一次
    global_step = tf.Variable(0, trainable=False)

    # 定义滑动平均类用来在训练模型之后更新shadow_variable
    # 目的是在初期加快更新后期接近最佳值时缓慢更新参数
    variable_average = tf.train.ExponentialMovingAverage(Moving_average_decay, global_step)
    variable_average_op = variable_average.apply(
        tf.trainable_variables()
    )

    # 交叉熵损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1)  # 返回最大值
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))  # 列表求和
    learing_rate = tf.train.exponential_decay(
        # 针对于学习率的指数衰减函数
        learning_rate=Learning_rate_base,
        global_step=global_step,  # 每个batch_size记录一次
        decay_steps=mnist.train.num_examples / Batch_size,  # 衰减速度
        decay_rate=Learning_rate_decay
    )
    # 定义模型
    train_step = tf.train.GradientDescentOptimizer(learning_rate=learing_rate).minimize(
        loss, global_step=global_step
    )
    with tf.control_dependencies([train_step, variable_average_op]):
        # 每轮训练过程后进行shadow_variable的更新
        train_op = tf.no_op(name='train')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for i in range(Train_step):
            x_train, y_train = mnist.train.next_batch(Batch_size)
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: x_train, y_: y_train})
            if i%1000 == 0:
                print("after %d training steps, loss on training batch is %g"
                      ", get score is %g" % (step, loss_value, (1-loss_value)))

                saver.save(
                    sess=sess,
                    save_path=os.path.join(model_path, model_name),
                    global_step=global_step
                )

# def main(argv=None):
#     mnist = input_data.read_data_sets("/path/to/MNIST_data/", one_hot=True)
#     train(mnist)
# if __name__ == '__main__':
#     tf.app.run()

import time
Eval_interval_secs = 10


# noinspection PyTypeChecker
def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(
            dtype=np.float, shape=[None, Input_node], name='x_input'
        )
        y_ = tf.placeholder(
            dtype=np.float, shape=[None, Output_node], name='y_label'
        )

        validate_data = {
            x: mnist.validation.images,
            y_: mnist.validation.labels
        }

        # 因为验证数据, 不关注正则化损失的值所以regularizer=None
        y = interence(input_tensor=x, regularizer=None)

        correct_y_ylabel = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuray = tf.reduce_mean(tf.cast(correct_y_ylabel, np.float))
        # 通过变量重命名的方法加载滑动平均模型 需要用到重命名字典
        variable_average = tf.train.ExponentialMovingAverage(
            Moving_average_decay
        )
        # variables_to_restore函数来生成tf.train.Saver类所需要的变量重命名字典
        variable_average_restore = variable_average.variables_to_restore()
        saver = tf.train.Saver(variable_average_restore)

        while True:
            with tf.Session() as sess:
                # 自动查找模型文件
                ckpt = tf.train.get_checkpoint_state(
                    'DL/'
                )
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    # 注意这里只能够获取最后一轮模型的global_step
                    accuray_Score= sess.run(accuray, feed_dict=validate_data)
                    print("After %s training steps, validate score is %g" % (global_step, accuray_Score))
                    x = accuray_Score
                else:
                    print("file not found")
            time.sleep(Eval_interval_secs)

def main(argv=None):
    mnist = input_data.read_data_sets("/path/to/MNIST_data/", one_hot=True)
    evaluate(mnist)
if __name__ == '__main__':
    tf.app.run()
'''

"""

# 第六章 卷积神经网络
"""
# 卷积层的构建
'''
Convolutions_weight = tf.get_variable(
    'weights', [5, 5, 3, 16], # 5,5 是过滤器的尺寸 3是当前层的深度 16是过滤器深度
    initializer=tf.truncated_normal_initializer(stddev=0.1)
)
# 注意过滤器尺寸指的是当前层的子节点矩阵的尺寸 过滤器深度指的是输出单位子节点矩阵的深度

biases = tf.get_variable(
    'biases', [16], # 和上面的16意义一样都是下一层的深度
    initializer=tf.constant_initializer(0.1)
)

conv = tf.nn.conv2d(
    input=input, filter=Convolutions_weight,
    strides=[1, 1, 1, 1], # strides[0]和strides[3]必须为1
    padding='SAME' # SAME指进行全0填充
)
bias = tf.nn.bias_add(conv, biases) # 需要将biases添加到矩阵的每一个节点上面

actived_conv = tf.nn.relu(bias)
'''

# 卷积网络LeNet-5模型进行mnist识别
'''
# 定义神经网络的参数
Input_node = 784; Output_node = 10

Image_size = 28; Num_channels = 1 # 颜色波段 因为是黑白色所以为1
Num_labels = 10
# 定义第一次卷积层
Conv1_deep = 32
Conv1_size = 5
# 定义第二层卷积层
Conv2_deep = 64
Conv2_size = 5
# 定义全连接层的节点数
Fulink_size = 512

def interfence(input_tensor, train, regularizer):
    with tf.variable_scope('conv1_layer'):
        conv1_weights = tf.get_variable(
            'weights', [Conv1_size, Conv1_size, Num_channels, Conv1_deep],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv1_bias = tf.get_variable(
            'biases', [Conv1_deep],
            initializer=tf.constant_initializer(0.1)
        )
        conv1_conv = tf.nn.conv2d(
            input=input_tensor, filter=conv1_weights,
            strides=[1, 1, 1, 1], padding='SAME'
        )
        conv1_biases = tf.nn.bias_add(conv1_conv, conv1_bias)
        conv1_relu = tf.nn.relu(conv1_biases)
    with tf.variable_scope('sub1_layer'):
        pond1 = tf.nn.max_pool(
            conv1_relu, ksize=[1, 2, 2, 1], # 中间两值为过滤器尺寸,ksize[0]和ksize[3]必须为1
            strides=[1, 2, 2, 1], padding='SAME'
        )
    with tf.variable_scope('conv2_layer'):
        conv2_weights = tf.get_variable(
            'weights', [Conv2_size, Conv2_size, Conv1_deep, Conv2_deep],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv2_bias = tf.get_variable(
            'biases', [Conv2_deep],
            initializer=tf.constant_initializer(0.1)
        )
        conv2_conv = tf.nn.conv2d(
            pond1, filter=conv2_weights,
            strides=[1, 1, 1, 1], padding='SAME'
        )
        conv2_biases = tf.nn.bias_add(conv2_conv, conv2_bias)
        conv2_relu = tf.nn.relu(conv2_biases)
    with tf.variable_scope('sub2_layer'):
        pond2 = tf.nn.max_pool(
            conv2_relu, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME'
        )

    # 构建全连接层
    # 首先转化最后一层池化层的输入格式
    pool_shape = pond2.get_shape().as_list()
    # pool_shape[0]是一个bathc中的数据个数
    print(pool_shape)
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

    # 将pond2池化层转变为符合全连接层的输入格式类似于第五章的layer1
    pool_reshape = tf.reshape(pond2, [pool_shape[0], nodes])

    # 声明第一层全连接层
    with tf.variable_scope('fc1_layer'):
        fc1_weights = tf.get_variable(
            'weights', shape=[nodes, Fulink_size],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        fc1_bias = tf.get_variable(
            'biases', shape=[Fulink_size],
            initializer=tf.constant_initializer(0.1)
        )
        if regularizer!=None:
            # 如果有正则化则将其加入到losses中
            tf.add_to_collections(
                'losses', regularizer(fc1_weights)
            )
        fc1 = tf.nn.relu(tf.matmul(pool_reshape, fc1_weights) + fc1_bias)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)
    with tf.variable_scope('fc2_layer'):
        fc2_weights = tf.get_variable(
            'weights', shape=[Fulink_size, Num_labels],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        fc2_bias = tf.get_variable(
            'biases', shape=[Num_labels],
            initializer=tf.constant_initializer(0.1)
        )
        if regularizer!=None:
            tf.add_to_collections('losses', regularizer(fc2_weights))
        fc2 = tf.matmul(fc1, fc2_weights) + fc2_bias
    return fc2

Batch_size = 100
Moving_decay = 0.99
Regularization_rate = 0.0001
Learning_rate_base = 0.01
Learning_rate_decay = 0.99
Training_steps = 30000
def train(mnist):
    x = tf.placeholder(
        dtype=np.float,
        shape=[Batch_size, Image_size, Image_size, Num_channels],
        name='x_input'
    )
    y_ = tf.placeholder(
        dtype=np.float, shape=[None, Output_node], name='y_label'
    )
    regularizer = tf.contrib.layers.l2_regularizer(Regularization_rate)
    y = interfence(x, False, regularizer)
    global_steps = tf.Variable(
        0.1, trainable=False
    )
    variable_average = tf.train.ExponentialMovingAverage(
        decay=Moving_decay, num_updates=global_steps
    )
    variable_average_op = variable_average.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1)
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    learning_rate = tf.train.exponential_decay(
        learning_rate=Learning_rate_base,
        global_step=global_steps,
        decay_steps=mnist.train.num_examples / Batch_size,
        decay_rate=Learning_rate_decay
    )
    losses = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss=losses, global_step=global_steps
    )
    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name='train')

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(Training_steps):
            xs, ys = mnist.train.next_batch(Batch_size)

            reshape_xs = np.reshape(xs,
                (Batch_size,
                 Image_size,
                 Image_size,
                 Num_channels)
            )
            _, loss_value, step = sess.run([train_op, losses, global_steps],
                                           feed_dict={x: reshape_xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), "
                      "loss on training batch is %g." % (step, loss_value))

def main(argv=None):
    mnist = input_data.read_data_sets("/path/to/MNIST_data/", one_hot=True)
    train(mnist)
if __name__ == '__main__':
    tf.app.run()
'''

# 卷积网络Inception-v3模型构建
'''
def Inception():
    silm = tf.contrib.silm
    with silm.arg_scope([silm.conv2d, silm.max_pool2d, silm.avg_pool2d], stride=1, padding='VALLD'):
        net = "上层节点的输出矩阵"
        with tf.variable_scope('mixed_7'):
            with tf.variable_scope('branch_0'):
                branch_0 = silm.conv2d(net, 320, # 深度
                                       kshp=[1, 1], scope='conv2d_0')
            with tf.variable_scope('branch_1'):
                branch_1 = silm.conv2d(net, 384,
                                       kshp=[1, 1], scope='conv2d_1')
                branch_1 = tf.concat(
                    3, # 深度
                    [
                        silm.conv2d(branch_1, 384, kshp=[1, 3], scope='conv2d_1a'),
                        silm.conv2d(branch_1, 384, kshp=[3, 1], scope='conv2d_1b')
                    ]
                )
            with tf.variable_scope('branch_2'):
                branch_2 = silm.conv2d(net, 448,
                                       kshp=[1, 1], scope='conv2d_2')
                branch_2 = silm.conv2d(branch_2, 384,
                                       kshp=[3, 3], scope='conv2d_2a')
                branch_2 = tf.concat(
                    3,
                    [
                        silm.conv2d(branch_2, 384, kshp=[1, 3], scope='conv2d_2b'),
                        silm.conv2d(branch_2, 384, kshp=[3, 1], scope='conv2d_2c')
                    ]
                )
            with tf.variable_scope('branch_3'):
                branch_3 = silm.avg_pool2d(net, [3, 3],
                                           scope='conv2d_3')
                branch_3 = silm.conv2d(branch_3, 192, [1, 1], scope='conv2d_3a')
            net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
        return net
'''

# TensorFlow实现迁移学习
'''
import glob
from tensorflow.python.platform import gfile
from sklearn.model_selection import train_test_split

def creat_image_data(path, sess, validate_data_rate, test_data_rate):
    datas_file_path = [x[0] for x in os.walk(path)]
    current_lable = 0
    # 每经历一个类别current_label+1 相当于将五个类别划分成五个数字当做它们的label
    # 比如说第一类别是rose,current_label=0,当第一个类别结束后current_label+1 = 1
    # 那么第二个类别就是1
    is_root_dir = True
    image_kinds = ['jpg', 'jpeg', 'JPG', 'JPEG']
    image_values = []
    image_labels = []
    for data_path in datas_file_path:
        if is_root_dir:
            is_root_dir=False
            continue
        file_list = []
        data_name = os.path.basename(data_path)
        # 获取到每张图片的地址
        for image_kind in image_kinds:
            date_file_glob = os.path.join(Input_datas_path, data_name, '*.' + image_kind)
            file_list.extend(glob.glob(date_file_glob))
        # print(file_list)

        # 处理图片数据
        for image_path_name in file_list[:3]:
            image_data = gfile.FastGFile(image_path_name, 'rb').read()
            image = tf.image.decode_jpeg(image_data)
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.image.resize_images(image, [299, 299])
            image_value = sess.run(image)
            image_values.append(image_value)
            image_labels.append(current_lable)
        current_lable += 1
    image_datas = pd.DataFrame({
        'image_values':image_values,
        'image_labels':image_labels
    })
    test_data = image_datas[:len(image_datas)//10]
    test_label = test_data.pop('image_labels')
    data = image_datas[len(image_datas)//10:]
    label = data.pop('image_labels')
    train_data, validate_data, train_label, validate_label = train_test_split(data, label,
                                                                     test_size=0.2, random_state=1)
    return np.asarray(
        [list(train_data), list(train_label),
         list(validate_data), list(validate_label),
         list(test_data), list(test_label)]
    )
'''
'''
            # 这样划分有点麻烦我把他转化为DataFrame划分数据
            chance = np.random.randint(100)
            # 这里说白了就是通过多次随机正整数的值来侧面构建一个百分比为
            # 8:1:1的train:validate:test的数据
            # 比如说第7张图片,chance=15那么这张图片就归到test数据内
            if chance < validate_data_rate:
                validate_data.append(image_value)
                validate_label.append(current_lable)
            elif chance < (validate_data_rate + test_data_rate):
                test_data.append(image_value)
                test_label.append(current_lable)
            else:
                train_data.append(image_value)
                train_label.append(current_lable)
        current_lable += 1
    state = np.random.get_state()
    np.random.shuffle(train_data) # state记录train_data的shuffle的状态
    np.random.set_state(state)
    np.random.shuffle(train_label) # 是train_label和train_date依旧一一对应
    print(test_data, len(test_data))
'''
'''
# 和np.array的区别是array会copy出一个新的ndarray, asarray可以理解为不copy直接使用本身
Input_datas_path = 'DL\\fdatas'
validate_rate = 10; test_rate = 10
def get_data():
    with tf.Session() as sess:
        prcess_data = creat_image_data(
            path=Input_datas_path, sess=sess,
            validate_data_rate=validate_rate,
            test_data_rate=test_rate
        )
        #print(prcess_data)
        np.save('DL/fdatas/proceedata/p_datas.npy', prcess_data)
#get_data()

import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3

datas_file_path = 'DL/fdatas/proceedata/'
model_save_file_path = 'DL/..'
model_file_path = 'DL/..'
Learning_rate = 0.0001
Steps = 300
Batch = 32
N_class = 5


import tensorflow.contrib.slim as slim

# 获取模型需要的参数
def get_turned_variables():
    flinklayer_noneed_scopes = ['Inception_v3/Logits', 'Inception_v3/AuxLogits']
    variabels_to_restore = [] # 此列表用于获取迁移模型需要的参数
    for var in slim.get_model_variables():
        excluded = False
        # 枚举所有Inception模型中的参数然后判断是否需要
        for scope in flinklayer_noneed_scopes:
            if var.op.name.startswith(scope):
                # 判断var是否是scope的开头
                # 即判断此参数是否是以Inception_v3/Logits或者Inception_v3/AuxLogits开头的参数
                excluded=True
                break
        if not excluded:
            # 此处说明参数不是以Inception_v3/Logits或者Inception_v3/AuxLogits开头的
            # 所以添加到列表内部
            variabels_to_restore.append(var)
    return variabels_to_restore

# 获取模型需要重新训练的参数
def get_trainable_variable():
    flinklayer_need_scopes = ['Inception_v3/Logits', 'Inception_v3/AuxLogits']
    variables_to_train = [] # 此列表用于获取需要重新训练的参数
    for var in flinklayer_need_scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, var)
        # 过滤可训练变量以scope=var为标准 最终生成一个列表因此使用get_collection()
        variables_to_train.extend(variables)
        # .extend() 若variables 为一个列表 则功能类似于两个列表相加(也就是链接)
        # 若variables是一个字符串则相当于拆分比如说.extend('zxc')->['z','x','c']
    return variables_to_train
def main():
    datas = np.load('DL/fdatas/proceedata/p_datas.npy')
    train_datas = datas[0]; train_label = datas[1]
    train_example = len(train_datas)
    validate_datas = datas[2]; validate_label = datas[3]
    test_datas = datas[4]; test_label = datas[5]

    images = tf.placeholder(
        dtype=np.float, shape=[None, 299, 299, 3], name='input'
        # 彩色图片因此为 3
    )
    labels = tf.placeholder(
        dtype=np.float, shape=[None], name='label'
    )
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(
            images, num_classes=train_example
        )
    # 获取模型需要重新训练的参数
    train_variable = get_trainable_variable()
    tf.losses.softmax_cross_entropy(
        onehot_labels=(labels, N_class), logits=logits, weights=1
    )
    # 定义训练过程
    train_steps = tf.train.RMSPropOptimizer(Learning_rate).minimize(
        loss=tf.losses.get_total_loss() # 获取损失值配合tf.losses使用
    )
    # 计算正确率
    with tf.variable_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
        #
        correct_prediction_mean = tf.reduce_mean(tf.cast(correct_prediction, np.float))
    load_model_fn = slim.assign_from_checkpoint_fn(
        model_file_path,
        get_turned_variables(), # 获取模型需要的参数
        ignore_missing_vars=True
    )
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        load_model_fn(sess) # 将模型加入到sess中
        start = 0
        end = Batch
        for i in range(Steps):
            sess.run(train_steps, feed_dict={
                images : train_datas[start:end],
                labels : train_label[start:end]
            })
            if i % 30 == 0 or i + 1 == Steps:
                saver.save(sess=sess, save_path=model_save_file_path, global_step=i)
                validate_accuracy = sess.run(correct_prediction_mean, feed_dict={
                    images : validate_datas,
                    labels : validate_label
                })
                print("step : %d, validate_accuracy : %g" % (i, validate_accuracy*100))
                start = end
                if start == train_example:
                    start = 0
                end = start + Batch
                if end > train_example:
                    end = train_example
        test_accuracy = sess.run(correct_prediction_mean, feed_dict={
            images : test_datas,
            labels : test_label
        })
        print("the test accuracy : %g" % (test_accuracy*100))
'''
"""

# 第七章 图像数据的处理

def int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))
def string_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))
'''
# TFRecord 输入数据格式
TFRecord_file_path = 'C:/Users/dell--pc/Desktop/py与贝叶斯/DL/TFRecord/'
TFRecord_data_name_train = 'outputss.tfrecords'
TFRecord_data_name_test  = 'output_test.tfrecords'

mnist = input_data.read_data_sets("../../datasets/MNIST_data",
                                  dtype=tf.uint8,
                                  one_hot=True)
images = mnist.train.images
labels = mnist.train.labels
pixels = images.shape[1]
num_example = mnist.train.num_examples
# images = mnist.test.images
# labels = mnist.test.labels
# pixels = images.shape[1]
# num_example = mnist.test.num_examples
if not os.path.exists(TFRecord_file_path):
    os.makedirs(TFRecord_file_path)
# 创建TFRecord写入函数
# write = tf.python_io.TFRecordWriter(TFRecord_file_path+TFRecord_data_name_train)
write = tf.python_io.TFRecordWriter(TFRecord_file_path+TFRecord_data_name_train)
for index in range(num_example):
    image_raw = images[index].tostring()
    example = tf.train.Example(
        # 注意是两个features 有s
        features=tf.train.Features(
            # tf.train.Features用于构造每个样本的信息键值对
            feature={
                'pixels': int64_feature(pixels),
                # 他会自动将获取到的信息分配给三种类型(bytes(string), int, float)
                'label': int64_feature(np.argmax(labels[index])),
                'image_raw': string_feature(image_raw)
            }
        )
    )
    write.write(example.SerializeToString())
write.close()
print("数据生成完毕")
'''

'''
reader = tf.TFRecordReader()

TFRecord_get_path = tf.train.string_input_producer(
    [TFRecord_file_path+TFRecord_data_name]
)
# 获取一个样例, 多个样例可以使用read_up_to()
_, example_tfrecord = reader.read(TFRecord_get_path)

# 解析样例, 多个样例可以使用parse_example()
features = tf.parse_single_example(
    example_tfrecord,
    features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
        'pixels': tf.FixedLenFeature([], tf.int64)
    }
)
# tf.decode_raw可以将字符串解析成图像
image = tf.decode_raw(features['image_raw'], tf.uint8)
label = tf.cast(features['label'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)

sess = tf.Session()
# 多线程
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
for i in range(10):
    print(sess.run([image, label, pixels]))
'''
# 图像数据预处理
'''
image = tf.gfile.GFile('C:/Users/dell--pc/Desktop/py与贝叶斯/DL/test.jpeg', 'rb').read()
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image)
    plt.imshow(img_data.eval())
    plt.show()
    # 改变图片的大小需要在Session()内操作

    img_float = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
    for i in range(1,5):
        new_image_1 = tf.image.resize_images(img_float, [150, 150], method=i)
        plt.imshow(new_image_1.eval())
        plt.show()

    # 改变图片的尺寸

    new_image_2 = tf.image.resize_image_with_crop_or_pad(img_data, 100, 100)
    plt.imshow(new_image_2.eval()); plt.show()
    new_image_2 = tf.image.resize_image_with_crop_or_pad(img_data, 400, 500)
    plt.imshow(new_image_2.eval()); plt.show()
    new_image_2 = tf.image.central_crop(img_data, 0.5)
    plt.imshow(new_image_2.eval()); plt.show()
  
    # 图像翻转
    fliped = tf.image.flip_left_right(img_data)
    updown = tf.image.flip_up_down(img_data)
    plt.imshow(fliped.eval()); plt.show()
    plt.imshow(updown.eval()); plt.show()
    # 随机翻转(50%)
    fliped_r = tf.image.random_flip_left_right(img_data)
    updown_r = tf.image.random_flip_up_down(img_data)
'''

def handle_color(image, var):
    if var==0:
        image = tf.image.random_brightness(image=image, max_delta= 32 / 255)
        image = tf.image.random_contrast(image=image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image=image, max_delta=0.2)
        image = tf.image.random_saturation(image=image, lower=0.5, upper=1.5)
    elif var==1:
        image = tf.image.random_contrast(image=image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image=image, max_delta=0.2)
        image = tf.image.random_saturation(image=image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image=image, max_delta=32 / 255)

    return tf.clip_by_value(image, 0.0, 1.0)
    # 截断操作,让小于0.0的等于0.0大于1.0的等于1.0

def preprocess_image(image, height, width, bbox):
    if bbox==None:
        bbox = tf.constant([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]], dtype=np.float)
        # 简单来说 rank是几就有几层中括号
    if image.dtype != tf.float32:
        # image = tf.shape(image)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        image_size=tf.shape(image), bounding_boxes=bbox, min_object_covered=0.4
    )
    # 从bbox_begin开始切一块大小为bbox_size的数据
    distorted_image = tf.slice(image, bbox_begin, bbox_size)
    # 调整大小
    distorted_image = tf.image.resize_images(
        distorted_image, [height, width], method=np.random.randint(4)
    )
    # 翻转图像
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    # 处理色彩
    distorted_image = handle_color(distorted_image, np.random.randint(1,2))
    images = tf.cast(distorted_image, tf.float32)
    images = tf.reshape(images, [784])
    return images


'''
image = tf.gfile.GFile('C:/Users/dell--pc/Desktop/py与贝叶斯/DL/test.jpeg', 'rb').read()
with tf.Session() as sess:
    image_data = tf.image.decode_jpeg(image)
    bbox = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
    plt.subplots(2, 3)
    for i in range(6):
        plt.subplot(2, 3, i+1)
        result = preprocess_image(image_data, 250, 250, bbox)
        plt.imshow(result.eval())
    plt.show()
'''
# 多线程输入数据处理框架

'''
#队列与多线程
q = tf.FIFOQueue(2, dtypes=tf.int32)
queue_init = q.enqueue_many(([0, 10], ))
# z = q.enqueue_many(([0, 10], ))
x = q.dequeue()
y = x + 1
new_queue = q.enqueue([y])
with tf.Session() as sess:
    # z.run() 你也可以直接初始化对应变量名
    # sess.run(q.enqueue_many(([0, 10], )))
    # 你可以把队列初始化放到这里
    # init = tf.global_variables_initializer()
    # 但是注意无法通过init初始化队列哪怕把队列初始化的任务放到一个变量上面也不可以
    # sess.run(init)
    sess.run(queue_init)
    for i in range(5):
        z, _ = sess.run([x, new_queue])
        print(z)
'''

# 简单线程
'''
import time
def myloop(coord, coordid):
    while not coord.should_stop():
        if np.random.rand() < 0.1:
            print("Stop coord from id is : %d" % coordid)
            coord.request_stop()
        else:
            print("walking coord id is : %d" % coordid)
        time.sleep(1)
coord = tf.train.Coordinator()
import threading
threads = [
    threading.Thread(target=myloop, args=(coord, i, )) for i in range(5)
]
for t in threads:
    t.start()
coord.join(threads)

# 线程控制队列

# 声明队列
queue = tf.FIFOQueue(100, dtypes=tf.float32)
# 初始化队列
init_queue = queue.enqueue([tf.random_normal([1])])
#tf.random_normal()和np.random.normal()两者的区别可查看以下代码

for i in range(3):
    np.random.seed(3)
    print(np.random.normal([3]))
with tf.Session() as sess:
    for i in range(3):
        np.random.seed(3)
        print(sess.run(tf.random_normal([3])))

# 我推测可能是会话的原因导致tf的random_normal和np的random.normal()产生如上的区别
queue_runner = tf.train.QueueRunner(queue, [init_queue] * 5)
tf.train.add_queue_runner(queue_runner)
x = queue.dequeue()

with tf.Session() as sess:
    # 声明多线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for _ in range(5):
        print(sess.run(x)[0])
    coord.request_stop()
    coord.join(threads)
    
    
# 输入文件队列
file_nums = 2
file_data_nums = 2
file_data_name = "data.tfrecords"
for i in range(file_nums):
    if not os.path.exists(TFRecord_file_path):
        os.makedirs(TFRecord_file_path)
    file_data_names = TFRecord_file_path + file_data_name+"-%.5d-of%.5d" % (i, file_data_nums)
    writer = tf.python_io.TFRecordWriter(file_data_names)
    for j in range(file_data_nums):
        examples = tf.train.Example(
            features=tf.train.Features(
                # tf.train.Features用于构造每个样本的信息键值对
                feature={
                    'i': int64_feature(i),
                    # 将获取到的信息分配给三种类型(bytes(string), int, float)
                    'j': int64_feature(j)
                }
            )
        )
        #writer.write(examples.SerializeToString())
    #writer.close()

files_path_names= TFRecord_file_path+file_data_name+"-*"
# 先通过match_filenames_once的正则式表述获取文件
files_list = tf.train.match_filenames_once(files_path_names)
# 再通过string_input_producer来进行管理
manage_file = tf.train.string_input_producer(files_list, shuffle=False)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(manage_file)
features = tf.parse_single_example(
      serialized_example,
      features={
          'i': tf.FixedLenFeature([], tf.int64),
          'j': tf.FixedLenFeature([], tf.int64),
      }
)
with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    print(sess.run(files_list))
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(6):
        print(sess.run([features['i'], features['j']]))
    coord.request_stop()
    coord.join(threads)
'''

# 文件队列和组合batch数据输入
"""
file_nums = 2
file_data_nums = 2
for i in range(file_nums):
    if not os.path.exists(TFRecord_file_path):
        os.makedirs(TFRecord_file_path)
    file_data_names = (TFRecord_file_path + "data.tfrecords-%.5d-of-%.5d" % (i, file_data_nums))
    writer = tf.python_io.TFRecordWriter(file_data_names)
    for j in range(file_data_nums):
        examples = tf.train.Example(
            features = tf.train.Features(
                feature={
                    'i': int64_feature(i),
                    'j': int64_feature(j)
                }
            )
        )
        writer.write(examples.SerializeToString())
    writer.close()

file_list_names = tf.train.match_filenames_once(TFRecord_file_path+"data.tfrecords-*")
manager_file = tf.train.string_input_producer(file_list_names)
reader = tf.TFRecordReader()
_, example = reader.read(manager_file)
features = tf.parse_single_example(
    example,
    features={
        'i': tf.FixedLenFeature([], tf.int64),
        'j': tf.FixedLenFeature([], tf.int64)
    }
)

'''
with tf.Session() as sess:
    tf.local_variables_initializer().run()
    print(sess.run(file_list_names))
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    for i in range(6):
        print(sess.run([features['i'], features['j']]))
    coord.request_stop()
    coord.join()
'''

train, label = features['i'], features['j']
batch_size = 3
capacity = 1000 + 3 * batch_size
# train_batch, label_batch = tf.train.batch(
#     [train, label], batch_size=batch_size, capacity=capacity
# )
train_batch, label_batch = tf.train.shuffle_batch(
    [train, label], batch_size=batch_size, capacity=capacity,
    min_after_dequeue=30 # 出队元素小于该值会等待更多的元素入队
)
with tf.Session() as sess:
    tf.local_variables_initializer().run()
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(2):
        train_, label_ = sess.run([train_batch, label_batch])
        print(train_, label_)
    coord.request_stop()
    coord.join(threads=threads)
"""

# 完整样例
"""
learning_rate_base = 0.01
input_node = 784
output_node = 10
layer_node = 500
regularaztion_rate = 0.0001
train_steps = 5001

def get_weight_variables(shape, r_rate=None):
    weights = tf.get_variable(
        'weights', shape,
        initializer=tf.truncated_normal_initializer(stddev=0.1)
    )
    if r_rate!=None:
        tf.add_to_collections('losses', r_rate(weights))
    return weights

def inference(i_tensor, regularazter):
    with tf.variable_scope('layer_1'):
        weights1 = get_weight_variables(
            shape=[input_node, layer_node], r_rate=regularazter
        )
        biases1 = tf.get_variable(
            'biases', [layer_node],
            initializer=tf.constant_initializer(0.0)
        )
        layer_1 = tf.nn.relu(tf.matmul(i_tensor, weights1) + biases1)

    with tf.variable_scope('layer_2'):
        weights2 = get_weight_variables(
            shape=[layer_node, output_node], r_rate=regularazter,
        )
        biases2 = tf.get_variable(
            'biases', [output_node],
            initializer=tf.constant_initializer(0.0)
        )
        layer_2 = tf.matmul(layer_1, weights2) + biases2
    r_score = regularazter(weights1) + regularazter(weights2)
    return layer_2, r_score

def train():
    files_path = (TFRecord_file_path + TFRecord_data_name + "*")
    files_list = tf.train.match_filenames_once(files_path)
    manage_files = tf.train.string_input_producer(files_list, shuffle=False)

    reader = tf.TFRecordReader()
    _, series_example = reader.read(manage_files)
    features = tf.parse_single_example(
        series_example,
        features={
                'pixels': tf.FixedLenFeature([], tf.int64),
                # 他会自动将获取到的信息分配给三种类型(bytes(string), int, float)
                'label': tf.FixedLenFeature([], tf.int64),
                'image_raw': tf.FixedLenFeature([], tf.string)}
    )

    decode_images = tf.decode_raw(features['image_raw'], tf.uint8)
    retype_images = tf.cast(decode_images, tf.float32)
    retype_labels = tf.cast(features['label'], tf.int64)
    # retype_pixels = tf.cast(features['pixels'], tf.int32)

    images = tf.reshape(retype_images, [784])
    labels = retype_labels
    images_hight = 299
    images_width = 299
    #images = preprocess_image(image=images, high=images_hight, width=images_width, bbox=None)

    batch_size = 100
    min_after_dequeue = 10000
    capacity = min_after_dequeue + batch_size * 3 # capacity mus bigger min_after_dequeue
    trains, labels = tf.train.shuffle_batch(
        [images, labels], batch_size=batch_size,
        capacity=capacity, min_after_dequeue=min_after_dequeue
    )
    regularazter = tf.contrib.layers.l2_regularizer(regularaztion_rate)
    y, r_score = inference(trains, regularazter)
    # global_steps = tf.Variable(0, trainable=False) 此变量通常伴生与learning_rate的修缮函数
    # 交叉熵
    cross_entory = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=labels
    )
    cross_entory_mean = tf.reduce_mean(cross_entory)
    # 损失函数
    losses = cross_entory_mean + r_score
    model_steps = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_base).minimize(
        loss=losses
    )
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(train_steps):
            if i % 1000 == 0:
                print("this steps is %d this loss score is %g " % (i, losses.eval()))
            sess.run(model_steps)
        coord.request_stop()
        coord.join(threads)
train()

"""


# 数据集 - tf.data
'''
#基本使用方法
x = [1, 2, 3, 4, 5]
# 从数组中创建数据集
datalist = tf.data.Dataset.from_tensor_slices(x)
# 类似于初始化吧
iterator = datalist.make_one_shot_iterator()
y = iterator.get_next()
z = y + 1
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    for i in range(len(x)):
        print(sess.run(z))
input_file = ["DL/normaldata/data.tfrecords-00000-*", "DL/normaldata/data.tfrecords-00001-of-00002"]
datatext = tf.data.TextLineDataset(input_file)
interator = datatext.make_one_shot_iterator()
x = interator.get_next()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(len(input_file)):
        print(sess.run(x))
'''

def parser(example):
    features = tf.parse_single_example(
        example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'chanel': tf.FixedLenFeature([], tf.int64)
        }
    )
    decoded_images = tf.decode_raw(features['image_raw'], tf.uint8)
    decoded_images = tf.reshape(decoded_images, shape=[features['height'], features['width'], features['chanel']])
    labels = tf.cast(features['label'], tf.int32)
    return decoded_images, labels

'''
# 需要注意这里的数据文件一定是完整形式
# 该方法是直接将文件写死
input_file = ["../py与贝叶斯/DL/TFRecord/output.tfrecords"] 
# 我现在使用两种方法获取TFR数据


# data.TFR可以代替train.match_filenames_once与train.string_input_producer的组合
# 且使用时只需要map就可应付各种解析情况
data = tf.data.TFRecordDataset(input_file)
# 使用map函数来表示对数据集中每一条数据进行调用解析对应的方法
dataset = data.map(parser)

# 注意使用one_shot_iterator方法时所有参数都已确定无需初始化
# 它的功能类似于TFRecordReader用来获取example
iterator = dataset.make_one_shot_iterator()
# 注意此时的迭代器一次返回的是两个值，这和前面的map作用对应
images, labels = iterator.get_next()
with tf.Session() as sess:
    # print(dataset) 查看后可得知他是将整个features包括到正如书上所说的一个数据块内
    for i in range(10):
        image, label = sess.run([images, labels])
        print(label)

files_list = tf.train.match_filenames_once("../py与贝叶斯/DL/TFRecord/outpu*")
# 这里的数据文件不需要完整形式即可
manager_file = tf.train.string_input_producer(files_list, shuffle=False)
reader = tf.TFRecordReader()
z, example = reader.read(manager_file)
# 这里的z是文件的路径信息
features = tf.parse_single_example(
    example,
    features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
    }
)
with tf.Session() as sess:
    tf.local_variables_initializer().run()
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print(features)
    for i in range(3):
        print(sess.run([features['label']]))
    coord.request_stop()
    coord.join(threads)


# 使用之前经常使用的占位符函数
input_file = tf.placeholder(dtype=tf.string)
data = tf.data.TFRecordDataset(input_file)
datatf = data.map(parser)
iterator = tf.data.make_initializable_iterator(dataset=datatf)
images, labels = iterator.get_next()
with tf.Session() as sess:
    tf.local_variables_initializer().run()
    tf.global_variables_initializer().run()
    sess.run(iterator.initializer,
             feed_dict={input_file: "../py与贝叶斯/DL/TFRecord/output.tfrecords"})
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(10):
        print("i:", i)
        image, label = sess.run([images, labels])
        print(label)
    coord.request_stop()
    coord.join(threads)
'''

# 使用data进行完整的数据预处理操作

input_file = tf.placeholder(dtype=tf.string)
layer_images_size = 250  # 神经网络层图片大小
batch_size = 100  # batch
shuffle_buffer = 10000  # 随机缓冲区大小
initial_data = tf.data.TFRecordDataset(input_file)
initial_data = initial_data.map(parser)
basic_data = initial_data.map(
    lambda image, label:
    (preprocess_image(image=image, height=layer_images_size, width=layer_images_size, bbox=None), label)
)
process_data = basic_data.shuffle(shuffle_buffer).batch(batch_size)
epoch_nums = 10
process_data = process_data.repeat(epoch_nums)
iterator = process_data.make_initializable_iterator()
train_images, train_labels = iterator.get_next()


learing_rate_base = 0.01
input_node = 784
layer_node = 500
output_node = 10
regularizer_rate = 0.0001
fit_steps = 5001

def get_weights(shape, r_rate):
    weights = tf.get_variable(
        name='weights', shape=shape,
        initializer=tf.truncated_normal_initializer(stddev=0.1)
    )
    if r_rate != None:
        tf.add_to_collections(names='losses', value=r_rate(weights))
    return weights

def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):
        weights1 = get_weights(shape=[input_node, layer_node], r_rate=regularizer)
        biases1 = tf.get_variable(
            name='biases', shape=[layer_node],
            initializer=tf.constant_initializer(0.0)
        )
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
    with tf.variable_scope('layer2'):
        weights2 = get_weights(shape=[layer_node, output_node], r_rate=regularizer)
        biases2 = tf.get_variable(
            name='biases', shape=[output_node,],
            initializer=tf.constant_initializer(0.0)
        )
        results_layer = tf.matmul(layer1, weights2) + biases2
    return results_layer, regularizer(weights1) + regularizer(weights2)

regularizer = tf.contrib.layers.l2_regularizer(regularizer_rate)
predict_label, regularizer_values = inference(input_tensor=train_images, regularizer=regularizer)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=train_labels, logits=predict_label
)
cross_entropy_mean = tf.reduce_mean(cross_entropy)
losses = cross_entropy_mean + regularizer_values

train_model = tf.train.GradientDescentOptimizer(learning_rate=learing_rate_base).minimize(
    loss=losses
)

input_file_test = tf.placeholder(dtype=tf.string)
basic_data_test = tf.data.TFRecordDataset(input_file_test)
initial_data_test = basic_data_test.map(parser)
iterator_test = tf.data.make_initializable_iterator()
test_images, test_labels = iterator_test.get_next()
predict_test_labels = inference(test_images, regularizer=None)
predict_score = tf.argmax(predict_test_labels, axis=-1, output_type=tf.int32)


with tf.Session() as sess:
    tf.local_variables_initializer().run()
    tf.global_variables_initializer().run()
    sess.run(iterator.initializer,
             feed_dict = {input_file: "../py与贝叶斯/DL/TFRecord/fdatas/tf_fdatas.tfrecords"})
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    while True:
        try:
            sess.run(train_model)
        except:
            break
    coord.request_stop()
    coord.join()

    test_label_values = []
    predict_test_label_values = []
    sess.run(iterator_test.initializer,
             feed_dict={input_file_test: "../py与贝叶斯/DL/TFRecord/fdatas/tf_fdatas_test.tfrecords"})
    while True:
        try:
            test_label_value, predict_test_label_value = sess.run([predict_test_labels, test_labels])
            test_label_values.extend(test_label_value)
            predict_test_label_values.extend(predict_test_label_value)
        except:
            break
result = [float(y == y_) for (y, y_) in zip(test_label_values, predict_test_label_values)]
result = sum(result) / len(result)
print("Accuracy is %g" % (result))


# 构建一些彩色图片的tfdatas用来测试以上代码
"""
input_file = "../py与贝叶斯/DL/fdatas/"
tf_file_path = "../py与贝叶斯/DL/TFRecord/fdatas/"
tf_name = "tf_fdatas_test.tfrecords"
# tf_name = "tf_fdatas.tfrecords" # 训练用数据
'''
if not os.path.exists(tf_file_path):
    os.makedirs(tf_file_path)

writer = tf.python_io.TFRecordWriter(tf_file_path + tf_name)
with tf.Session() as sess:
    for kind, files in enumerate(os.listdir(input_file)):
        files_nums = len(os.listdir(input_file+files))
        nums = 0
        if files_nums > 500:
            nums = files_nums // 4
        #     nums = files_nums // 4
        # for file_name in os.listdir(input_file+files)[:nums]: 这一块是训练需数据
        for file_name in os.listdir(input_file + files)[nums:(nums+20)]:
            basic_image = tf.read_file(input_file + files + '/' + file_name)
            process_image = tf.image.decode_jpeg(basic_image)
            image = sess.run(process_image)
            width = image.shape[0]
            height = image.shape[1]
            chanel = image.shape[-1]
            image = image.tostring()
            features = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image_raw': string_feature(image),
                        'label': int64_feature(kind),
                        'height': int64_feature(height),
                        'width': int64_feature(width),
                        'chanel': int64_feature(chanel)
                    }
                )
            )
            writer.write(features.SerializeToString())
    writer.close()
# 测试以上数据是否读写完毕
'''
test_file = [tf_file_path+tf_name]
reader = tf.data.TFRecordDataset(test_file)
datas = reader.map(parser)
iterator = datas.make_one_shot_iterator()
image, label = iterator.get_next()
with tf.Session() as sess:
    tf.local_variables_initializer().run()
    tf.global_variables_initializer().run()
    while True:
        try:
            f1, f2 = sess.run([image, label])
            print(f1.shape)
        except:
            break
"""
