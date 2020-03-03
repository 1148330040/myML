#coding:gbk

# DNN、CNN、RNN(DNN, CNN被统称为前向传播网络)三种主流神经网络的发展与区别
'''
DNN:简单来说就是使用更多的层DENSE来训练更多的参数从而实现更优的效果
但是我们知道权重的变化是需要一个参数(学习率)来配合前向传播或者梯度下降算法来改变的
但是随着我们层数的增多(又是全连接层DENSE)会导致参数的巨幅上升因此随着训练的推进
最后会发现训练信号几乎为0(比如说信号为1 衰减梯度(学习率)为0.25, 每次变化都需要乘以0.25
这导致在后期几乎接受不到有效的信号)-梯度消失问题
因此为了解决DNN由于层数的变化而导致的训练权重的增多引进了CNN(卷积层, 非常适合处理图像(输入元素很多)问题)网络
我们知道DNN层层之间是全连接层因此导致了训练权重的增多,因此CNN在层与层之间添加了
卷积核, 卷积核的作用简单来说就是(减少训练权重)充当中介, 每一层在通过卷积核后还保留了原有的位置
CNN模型限制参数的个数并挖掘了局部结构的这个特点
DNN层还有一个问题就是无法对时间序列上的问题解决(典型例子就是语音每一段话都需要结合上一段话的语境才能做出判断),
而前向传播网络都只是处理完当下的数据并传递到一层后就并不参与到接下来的训练过程,显然并不能满足语音等相关问题
因此RNN出现, RNN的隐藏层是互联的它的输出可以作用到下一个隐藏的同时还作用到自身
因此该神经网络的输出是全部隐藏层共同作用的结果, 可以说RNN的深度取决于时间上的长度,那么假如'时间'过长
是不是也会出现梯度消失这个问题？此时引入了LSTM(长短时间记忆单元)结构配合RNN网络
长短时记忆单元LSTM，通过门的开关实现时间上记忆功能，并防止梯度消失
LSTM 由遗忘门: 管理输出信息(0: 表示拒绝通过也就是遗忘 1: 表示允许通过)
输入门: 由两个前向传播神经网络(sigmoid: 全连接层决定那一部分输入信息需要被更新
       tanh: 输出一个向量Ct) 两个全连接层合并成一个新神经元Ct‘
状态控制: 将Ct' 更新成新的Ct
输出门: 通过更新后的Ct和两个全连接层sigmoid和tanh层生成输出结果
'''
from __future__ import absolute_import, division, print_function, unicode_literals
'''
二分类任务常用sigmoid作为损失函数非0即1
多分类任务则需要使用softmax作为损失函数以获取每个类别的概率
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
# 优化器, 学习率, 损失函数三者间的关系
# 学习率是优化器的优化单位, 损失函数是优化器的参考
# 因此为了达到最优学习效果, 我们既需要保证学习率不要太大, 以防止跳过最优点
# 例如 最优点0.00015  学习率: 0.00005 此时我们优化的结果是0.00018
# 这时下一轮训练的结果就是0.00013明显跳过了最优点0.00015
# 还要保证学习率不要太小, 以防止训练时间过长
# 理论上学习率越小最终获取的结果越好, 但是花费的时间会太长了
import tensorflow_datasets as tfds
import tensorflow_hub as tfub
'''
fit():
训练数据可以 完整地 放入到内存（RAM）里
数据已经不需要再进行任何处理了
fit_generator():
内存不足以一次性加载整个训练数据的时候
需要一些数据预处理（例如旋转和平移图片、增加噪音、扩大数据集等操作）
在生成batch的时候需要更多的处理

通常fit_generator()函数配合
tf.keras.preprocessing.image.ImageDataGenerator.flow_from_directory
这里的flow_from_directory()可以视作数据处理函数放入fit_generator()中
作为第一个参数, 当然也可以自定义数据处理函数(ps: 不要使用return或者exit)
'''
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None) # 显示所有列
deep_path = 'C:\\Users\\tzl17\\Desktop\\Python&ML\\DeepLearning\\'
from sklearn.model_selection import train_test_split
import random
import re
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # mpimg 用于读取图片连接
import traceback
import seaborn as sns

import json
import pathlib
# 面向对象的文件系统路径, 该模块提供了一些使用语义表达来
# 表示文件系统路径的类，这些类适合多种操作系统
# 路径类被划分为纯路径（该路径提供了不带I/O的纯粹计算操作）
# 以及具体路径（从纯路径中继承而来，但提供了I/O操作）
import shutil
# 高级的 文件、文件夹、压缩包 处理模块
import tempfile
# 生产临时文件和目录
import functools

'''
函数前单下划线
_XXX
此类函数只有类和子类能够访问调用，无法通过Import引入

函数前双下划线
__XXX
此类函数只有类对象本身能够访问

函数前后双下划线
__XXX__
此类函数为系统定义函数名，命名函数时尽量避免此类命名方式
'''

# mnist图像识别问题
'''
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
# np.newaxis, tf.newaxis 都是增添一个新的空维度
X_train = X_train[..., tf.newaxis]
X_test = X_test[..., tf.newaxis]
train_datas = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)).shuffle(10000).batch(32)
test_datas = tf.data.Dataset.from_tensor_slices(
    (X_test, y_test)).batch(32)

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu') # 卷积层
        self.flatten = tf.keras.layers.Flatten() # 将数据展开, 从(28, 28)的形状 -> 28*28，即Flatten:扁平化
        self.dens1 = tf.keras.layers.Dense(128, activation='relu') # 全连接层 即交叉处理（特征工程）层
        self.dens2 = tf.keras.layers.Dense(10, activation='softmax') 
        # 'softmax'的作用是将分类值转变为回归值，因为数字包含0-9有10中因此最后一次的神经元数是10

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.dens1(x)
        return self.dens2(x)
model = MyModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy() # 损失函数
optimizer = tf.keras.optimizers.Adam() # 优化器 适用于数据或者参数较大的模型

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

def train_step(images, labels):
    with tf.GradientTape() as tape: # 梯度计算，可以查看每一阶段的相应值
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)
def test_step(images, labels):
    predictions = model(images)
    loss = loss_object(labels, predictions)
    test_loss(loss)
    test_accuracy(labels, predictions)


for step in range(5):
    for images, labels in train_datas:
        train_step(images, labels)
    for images, labels in test_datas:
        test_step(images, labels)

    template = 'Step {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(step + 1,train_loss.result(),train_accuracy.result() * 100,
                          test_loss.result(), test_accuracy.result() * 100)
          )
'''

print(tf.__version__)

# keras 基本图像识别问题
'''
datas = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = datas.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(X_train[i])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(class_names[y_train[i]])
    plt.grid(False)


X_train = X_train[..., tf.newaxis]
X_test = X_test[..., tf.newaxis]


def merge_datas(X, y):
    return tf.data.Dataset.from_tensor_slices(
        (X, y)).shuffle(10000 if X.shape[0] > 10000 else X.shape[0] // 5).batch(32)
train_ds = merge_datas(X_train, y_train)
test_ds = merge_datas(X_test, y_test)

class my_model(tf.keras.Model):
    def __init__(self):
        super(my_model, self).__init__()
        # self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28)) # 将数据展开
        self.dens1 = tf.keras.layers.Dense(128, activation='relu')
        self.dens2 = tf.keras.layers.Dense(10, activation='softmax')
    def call(self, datas):
        # datas = self.conv1(datas)
        datas = self.flatten(datas)
        datas = self.dens1(datas)
        return self.dens2(datas)
model = my_model()

loss_funcation = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


def train_step(images, labels):
    with tf.GradientTape() as gt:
        predication = model(images)
        loss = loss_funcation(labels, predication)
    graditens = gt.gradient(loss, model.trainable_variables) # 记录当前轮的值
    optimizer.apply_gradients(zip(graditens, model.trainable_variables)) # 优化器根据当前轮数的值调整模型相关值

    train_loss(loss)
    train_accuracy(labels, predication)

def test_step(images, labels):
    t_predications = model(images)
    t_loss = loss_funcation(labels, t_predications)

    test_loss(t_loss)
    test_accuracy(labels, t_loss)

for epoch in range(5):
    for images, labels in train_ds:
        train_step(images, labels)
    for images, labels in test_ds:
        test_step(images, labels)

    print(epoch, train_loss, train_accuracy, test_loss, test_accuracy)


'''

# IMDB影评文本分类(使用tf.hub ps: 已训练好的模型库)
"""
import tensorflow_hub as hub
import tensorflow_datasets as tfds

train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])

(train_datas, validation_datas), test_datas = tfds.load(
    # tfds.load() 构造生成器、下载数据和 创建输入管道(pipeline)，返回“tf.data.Dataset”。
    # tfds.builder 按名称获取 tfds.core.DatasetBuilder
    # tfds.core.DatasetBuilder 主要包含三种作用
    # 1: 记录 2: 下载 3: 建立pipeline
    name='imdb_reviews',
    split=(train_validation_split, tfds.Split.TEST), # 返回要拆分的数据集，None则返回dict结构
    as_supervised=True, # 为True则返回类似于(train, label)的二元结构False返回dict结构
    download=True # 默认True 下载/准备数据及集
)
train_example_batch, train_label_batch = next(iter(train_datas.batch(10)))

# 创建一个keras层
embding = 'https://hub.tensorflow.google.cn/google/tf2-preview/gnews-swivel-20dim/1'
keras_layers = hub.KerasLayer(embding, input_shape=[],
                              dtype=tf.string, trainable=True)
model = tf.keras.Sequential()
model.add(keras_layers)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_datas.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_datas.batch(512),
                    verbose=1)
results = model.evaluate(test_datas.batch(512), verbose=2)
for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))

'''
# 尝试更细致的划分模型的训练步骤
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.keras1 = hub.KerasLayer(embding, input_shape=[],
                                     dtype=tf.string, trainable=True)
        self.dens1 = tf.keras.layers.Dense(16, activation='relu')
        self.dens2 = tf.keras.layers.Dense(1, activation='sigmoid')
        # 为什么明明标签只有两种0,1但是输出层的节点数确是1 而不是2呢？
        # 从train_label_example标签中可以看出，他是一个长度为20的列表
        # 因而可以得出该模型的预测方式并不是直接根据句子输出一个结果，而是根据句子整体划分出20个结果来看
        # 因而输出的标签可以认为是一个个的浮点数所以，输出层的节点是1， 激活函数是sigmoid
    def call(self, datas):
        datas = self.keras1(datas)
        datas = self.dens1(datas)
        return self.dens2(datas)
model = MyModel()

# 构建优化器和损失函数
ooptimizer = tf.keras.optimizers.Adam()
loss_funcation = tf.keras.losses.SparseCategoricalCrossentropy()
accuracy = tf.keras.metrics.Accuracy()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
def train_step(datas, labels):
    with tf.GradientTape() as gt:
        prediction = model(datas)
        loss = loss_funcation(labels, prediction)
    gradients = gt.gradient(loss, model.trainable_variables)
    ooptimizer.apply_gradients(zip(gradients, model.trainabel_variables))
    # accuracy.apply_gradients(labels, prediction)
    train_loss(loss)
    train_accuracy(labels, prediction)

def test_step(datas, labels):
    t_prediction = model(datas)
    t_loss = loss_funcation(labels, t_prediction)
    test_loss(t_loss)
    test_accuracy(labels, t_prediction)

for step in range(5):
    for datas, labels in next(iter(train_datas.shuffle(10000).batch(512))):
        train_step(datas, labels)
    for datas, labels in next(iter(test_datas.shuffle(10000).batch(512))):
        test_step(datas, labels)

    template = 'Step {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(step + 1,train_loss.result(),train_accuracy.result() * 100,
                          test_loss.result(), test_accuracy.result() * 100)
          )
'''
"""

# 对预处理文本进行分类
"""
imdb = tf.keras.datasets.imdb

(train_datas, train_labels), (test_datas, test_labels) = imdb.load_data(num_words = 10000)

print("train_datas len: {}, test_datas len: {}".format(len(train_datas), len(test_datas)))

# 获取数字对应的值
# 仅供参考，真正意义上的输入是不可能直接使用字符串的都需要转化为数字
'''
word_index = imdb.get_word_index()
word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3
new_word_index = dict([(value, key) for (key, value) in word_index.items()])

def change_datas_word(data):
    data = [word_index.get(num, '?') for num in data]
    # 字典的get函数用于获取某个值 如果该值不存在则用 ？ 代替
    return ' '.join(data)
# train_datas = [change_datas_word(text) for text in train_datas]
'''

# 输入神经网络的值必须为形状一致的张量
# 使用填充的方法填充0使其长度一致
def change_datas_tensor(data):
    return tf.keras.preprocessing.sequence.pad_sequences(
        data,
        value=0,
        # 默认值为0.0 这里的word_index['<PAD>']其实也是0
        padding='post',
        maxlen=256
    )

train_datas = change_datas_tensor(train_datas)
test_datas = change_datas_tensor(test_datas)
# 输入形状是用于电影评论的词汇数目（10,000 词）
# 获取imdb由强调过num_words=10000
vocab_size = 10000
# 构建模型
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 16))
model.add(tf.keras.layers.GlobalAvgPool1D())
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

x_val = train_datas[:10000]
x_train = train_datas[10000:]
y_val = train_labels[:10000]
y_train = train_labels[10000:]

history = model.fit(
    x_train,
    y_train,
    epochs=40,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=1
)
results = model.evaluate(test_datas,  test_labels, verbose=2)

print(results)

'''
# 尝试使用更细化的方式实现，仍然失败
class my_model(tf.keras.Model):
    def __init__(self):
        super(my_model, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, 16)
        # 嵌入层 用于获取一个向量维度为16
        self.gap = tf.keras.layers.GlobalAvgPool1D()
        # 将通过对序列维度求平均值来为每个样本返回一个定长输出向量。这允许模型以尽可能最简单的方式处理变长输入。
        self.dens1 = tf.keras.layers.Dense(16, activation='relu')
        self.dens2 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, datas):
        datas = self.embedding(datas)
        datas = self.gap(datas)
        datas = self.dens1(datas)
        return self.dens2(datas)

optimizer = tf.keras.optimizers.Adam()
loss_funcation = tf.keras.losses.SparseCategoricalCrossentropy()

def datas_loss_accuracy(names):
    return tf.keras.metrics.Mean(name=names+'_loss')\
        , tf.keras.metrics.SparseCategoricalAccuracy(name=names+'_accuracy')
train_loss, train_accuracy = datas_loss_accuracy('train')
test_loss, test_accuracy = datas_loss_accuracy('test')

x_val = train_datas[:10000]
x_train = train_datas[10000:]
y_val = train_labels[:10000]
y_train = train_labels[10000:]

model = my_model()

def train_step(datas, labels):
    with tf.GradientTape() as gt:
        predictions = model(datas)
        loss = loss_funcation(labels, predictions)
    gradient = gt.gradient(predictions, model.trainable_variables)
    optimizer.apply_gradients(zip(gradient, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

def test_step(datas, labels):
    t_predictions = model(datas)
    t_loss = loss_funcation(labels, t_predictions)
    test_loss(t_loss)
    test_accuracy(labels, t_predictions)

train_step(x_train, y_train)
print(train_loss.result(),train_accuracy.result() * 100)
'''
"""

# 回归问题
"""
# 数据集: Auto MPG 数据集
dataset_path = tf.keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
# 英里/加仑,汽缸,取代,马力,重量,加速度,车型年,起源
datas = pd.read_csv(dataset_path, names=column_names,
                    na_values = "?", comment='\t',
                    sep=" ", # 指定分隔符，否则会使用逗号作为分隔符
                    skipinitialspace=True)

print(datas.isna().sum())
print(datas.head())
print(datas['Origin'].value_counts())
datas.dropna(inplace=True)

# "Origin" 列实际上代表分类，而不仅仅是一个数字。所以把它转换为独热码 （one-hot）:

datas['USA'] = datas['Origin'].apply(lambda x: 1.0 if x==1 else 0)
datas['Europe'] = datas['Origin'].apply(lambda x: 1.0 if x==2 else 0)
datas['Japan'] = datas['Origin'].apply(lambda x: 1.0 if x==3 else 0)
origin = datas.pop('Origin')

train_datas = datas.sample(frac=0.8, random_state=123)
test_datas = datas.drop(train_datas.index)

# sns.pairplot(datas[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind='kde')
# plt.show()

train_labels = train_datas.pop('MPG')
test_labels = test_datas.pop('MPG')

train_states = train_datas.describe()
train_states = train_states.transpose()

def norm(x):
    return (x - train_states['mean']) / train_states['std']
norm_train_datas = norm(train_datas)
norm_test_datas = norm(test_datas)

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=[len(train_datas.columns)]),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.01)

    model.compile(
        loss='mse',
        optimizer=optimizer,
        metrics=['mae', 'mse'] # 使用均方误差 mse , 均绝对值误差 mae
    )

    return model
model=build_model()
print(model.summary())

class output_step(tf.keras.callbacks.Callback):
    def epocs(self, epoch, logs):
        if epoch%100 == 0:
            print()

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(
    norm_train_datas, train_labels,
    epochs=1000, validation_split=0.2, verbose=0,
    callbacks=[early_stop, output_step()]
)

print(history)
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
# print(hist.tail())

results = model.evaluate(norm_test_datas,  test_labels, verbose=2)

test_predict_result = model.predict(norm_test_datas).flatten()

plt.scatter(test_labels, test_predict_result)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()
"""

# 过拟合的处理方法
"""
logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)
gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz')

FEATURES = 28
ds = tf.data.experimental.CsvDataset(gz,[float(),]*(FEATURES+1), compression_type="GZIP")

def pack_row(*row):
    label = row[0]
    feature = tf.stack(row[1:], 1)
    return feature, label
packed_ds = ds.batch(10000).map(pack_row).unbatch()
# unbatch() 数据集的元素拆分为连续元素的序列
n_validation = int(1e3)
n_train = int(1e4)
buffer_size = int(1e4)
batch_size = 500
steps_per_epoch= n_train//batch_size
validation_ds = packed_ds.take(n_validation).cache()
train_ds = packed_ds.skip(n_validation).take(n_train).cache()
# .skip() 顾名思义-跳过，即跳过validation_ds数据使其与train_ds分开
# .skip() 函数只能在TensorFlow内使用
# .cache() 方法来确保加载器不需要在每个时期重新读取文件中的数据

learning_rate_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    initial_learning_rate=0.001,
    decay_steps=steps_per_epoch * 1000,
    # steps_per_epocj=1, 因此这里的意思是以1000为周期每次变化率为1/3
    decay_rate=1,
    staircase=False
)

def get_optimizer():
    return tf.keras.optimizers.Adam(learning_rate_schedule) # Adam作为优化器
# step = np.linspace(0,100000)
# lr = learning_rate_schedule(step)
# plt.figure(figsize = (8,6))
# plt.plot(step/steps_per_epoch, lr)
# plt.ylim([0,max(plt.ylim())])
# plt.xlabel('Epoch')
# _ = plt.ylabel('Learning Rate')
# plt.show()
# 由此图可以看到以1000为节点均会发生变化

def get_callback(name):
    return [
        tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy'),
        tf.keras.callbacks.TensorBoard(logdir/name) # 获取训练日志
    ]

def compile_fit(model, name, optimizer=None, max_epochs=10000):
    if optimizer==None:
        optimizer = get_optimizer()
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 'binary_crossentropy']
    )
    model.summary()
    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=max_epochs,
        validation_data=validation_ds,
        callbacks=get_callback(name),
        verbose=0
    )
    return history

tiny_model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_history = {}

model_history['tiny'] = compile_fit(tiny_model, 'sizes/tiny')

small_model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
    tf.keras.layers.Dense(16, activation='elu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_history['small'] = compile_fit(small_model, 'sizes/small')

medium_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='elu', input_shape=(FEATURES,)),
    tf.keras.layers.Dense(64, activation='elu'),
    tf.keras.layers.Dense(64, activation='elu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_history['medium'] = compile_fit(small_model, 'sizes/medium')

large_model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='elu', input_shape=(FEATURES,)),
    tf.keras.layers.Dense(512, activation='elu'),
    tf.keras.layers.Dense(512, activation='elu'),
    tf.keras.layers.Dense(512, activation='elu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model_history['large'] = compile_fit(small_model, 'sizes/large')

# 正则化模型
L_model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(FEATURES, ),
                          regularizer=tf.keras.regularizers.l2),
    tf.keras.layers.Dense(16, activation='relu',
                          regularizer=tf.keras.regularizers.l1)
])
# 添加dropout, dropout通常用于大型模型和数据中
# dropout的作用是使输入数据中的某些元素变为0，其它没变0的元素变为原来的1/keep_prob
droup_out_model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(FEATURES, )),
    tf.keras.layers.Dropout(keep_prob=0.5),
    tf.keras.layers.Dense(512, activation='elu'),
    tf.keras.layers.Dropout(keep_prob=0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
"""

# 保存和加载模型
"""
(train_datas, train_labels), (test_datas, test_labels) = tf.keras.datasets.mnist.load_data()
train_datas = train_datas[:1000].reshape(-1, 28 * 28) / 255.0
test_datas = test_datas[:1000].reshape(-1, 28 * 28) / 255.0
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

def get_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(784, )),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
model = get_model()
print(model.summary())

checkpoint_path = 'training_1/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callbacks = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# model.fit(
#     train_datas, train_labels,
#     epochs=10,
#     validation_data=(test_datas, test_labels),
#     callbacks=[cp_callbacks]
# )

test_model = get_model()
loss, acc = test_model.evaluate(test_datas, test_labels, verbose=2)
print(loss, acc)

test_model.load_weights(checkpoint_path)
loss, acc = test_model.evaluate(test_datas, test_labels, verbose=2)
print(loss, acc)

new_checkpoint_path = 'training_2/cp-{epoch:04d}.ckpt'
new_checkpoint_dir = os.path.dirname(new_checkpoint_path)
new_cp_callbacks = tf.keras.callbacks.ModelCheckpoint(
    filepath=new_checkpoint_path,
    verbose=1,
    save_weights_only=True,
    period=5 # 每五个epoch保存一次权重节点
)

new_model = get_model()
new_model.save_weights(new_checkpoint_path.format(epoch=0))
new_model.fit(
    train_datas, train_labels,
    epochs=50,
    validation_data=(test_datas, test_labels),
    callbacks=[cp_callbacks],
    verbose=1
)

latest = tf.train.latest_checkpoint(new_checkpoint_dir)
print(latest)
new_model.save('my_model.h5')

new_model.save_weights('./checkpoints/my_checkpoint_dir')
# 不使用tf.keras.callbacks.ModelCheckpoint，直接手动保存
new_2_model = get_model()
loss, acc = new_2_model.evaluate(test_datas, test_labels, verbose=2)
print(loss, acc)
new_2_model.load_weights('./checkpoints/my_checkpoint_dir')
loss, acc = new_2_model.evaluate(test_datas, test_labels, verbose=2)
print(loss, acc)
new_3_model = tf.keras.models.load_model('my_model.h5')
loss, acc = new_3_model.evaluate(test_datas, test_labels, verbose=2)
print(loss, acc)
"""


# CSV 数据
"""
Train_data_url = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
Test_data_url = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

Train_data_path = tf.keras.utils.get_file('train.csv', Train_data_url,)
Test_data_path = tf.keras.utils.get_file('test.csv', Test_data_url)
np.set_printoptions(precision=3, suppress=True)

# 使用pandas先看看数据的样子
trains = pd.read_csv(Train_data_path,
                     skipinitialspace=True)
print(trains.head())

# 如果检查数据发现第一行不是数据的列名，则可以手动添加
# 在pandas 内可以直接使用names=column_names， 在TensorFlow中也可以使用tf.datas
CSV_COLUMNS = ['survived', 'sex', 'age', 'n_siblings_spouses', 'parch', 'fare', 'class', 'deck', 'embark_town', 'alone']
# train = tf.data.experimental.make_csv_dataset(
#     columns_names=CSV_COLUMNS
# )
# 如果想获取指定的列的话，在pandas可以使用usecols,在TensorFlow中可以使用
# columns_my_need=['survived', 'sex', 'age']
# train = tf.data.experimental.make_csv_dataset(
#     select_columns=columns_my_need
# )
# 提供数据组的待预测标签，则需要自己指定
label_cols = 'survived'
labels = [0, 1]
def get_datas(file_path):
    datas = tf.data.experimental.make_csv_dataset(
        file_path, # 文件的地址
        batch_size=12,  # 为了示例更容易展示，手动设置较小的值
        label_name=label_cols, # 需要预测的标签，注意是字符串label_cols = 'survived'
        na_value="?", # 和pandas内的read_csv一致,用于填充Nan
        num_epochs=1,
        ignore_errors=True
    )
    return datas
train_datas = get_datas(Train_data_path)
test_datas = get_datas(Test_data_path)
examples, labels = next(iter(train_datas)) # 第一个批次
print("EXAMPLES: \n", examples, "\n")
print("LABELS: \n", labels)

# 分类数据
# 部分数据是分类的情况，可以使用使用 tf.feature_column API 创建一个
# tf.feature_column.indicator_column 集合
# 每个 tf.feature_column.indicator_column 对应一个分类的列

# 是后续构建模型时处理输入数据的一部分
CATEGORIES = {
    'sex': ['male', 'female'],
    'class' : ['First', 'Second', 'Third'],
    'deck' : ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'embark_town' : ['Cherbourg', 'Southhampton', 'Queenstown'],
    'alone' : ['y', 'n']
}
category_columns = []
for feature, vocab in CATEGORIES.items():
    cate_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature, vocabulary_list=vocab
    )
    category_columns.append(tf.feature_column.indicator_column(
        cate_col
    ))
print(category_columns)
# 连续数据
# 连续数据则需要为其构建标准化函数
def process_continue_datas(mean, data):
    data = tf.cast(data, tf.float32) / mean # tf.cast 将数据转换为
    return tf.reshape(data, [-1, 1]) # 将传入的每一列值转化形状

MEANS = {
    # 这里面建议配合Dataframe直接填写datas['age'].mean()
    'age' : 29.631308,
    'n_siblings_spouses' : 0.545455,
    'parch' : 0.379585,
    'fare' : 34.385399
}
numerical_columns = []
# tf.feature_columns.numeric_column API 会使用 normalizer_fn 参数
# 在传参的时候使用 functools.partial，functools.partial 由使用每个列的均值进行标准化的函数构成
for feature in MEANS.keys():
    num_col = tf.feature_column.numeric_column(
        feature, normalizer_fn=functools.partial(process_continue_datas, MEANS[feature])
    )
    # functools.partial的作用相似于map函数和lambda函数
    numerical_columns.append(num_col)
print(numerical_columns)

# 分类的数据列和连续的数据列处理，开始建立预处理层
processing_layers = tf.keras.layers.DenseFeatures(
    category_columns+numerical_columns
)
print(processing_layers)
def get_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(monitor='binary_crossentropy', patience=200),
        tf.keras.callbacks.TensorBoard('test_solv_csvdatas')
    ]


def model_step():
    model = tf.keras.Sequential([
        processing_layers,
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        loss='binary_crossentropy',
        optimize='adam',
        metrics=['accuracy']
    )

    return model
model = model_step()
history = model.fit(
    train_datas.shuffle(500),
    epochs=20
)
print(pd.DataFrame(history.history))

test_loss, test_accuracy = model.evaluate(test_datas)
print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))

predictions = model.predict(test_datas)
# 显示部分结果

for prediction, survived in zip(predictions[:10], list(test_datas)[0][1][:10]):
    print("Predicted survival: {:.2%}".format(prediction[0]),
          " | Actual outcome: ",
          ("SURVIVED" if bool(survived) else "DIED"))
          # bool(survived) 相当于bool(1)-True 或者 bool(0)-False 因为 list(test_datas)[0][1]非1即0
# print("this is test_datas\n", test_datas)
# print("this is list(test_datas)\n", list(test_datas))
# print("this is list(test_datas)[0][1]\n", list(test_datas)[0][1])
"""

# Numpy 数据 .npz数据
"""
# 主要环节包括提取数据，将数据和标签结合，打乱(针对训练数据)和批次化数据

DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
minist_data_path = tf.keras.utils.get_file('mnist.npz', DATA_URL)

with np.load(minist_data_path) as mnist:
    train = mnist['x_train']
    train_label = mnist['y_train']
    test = mnist['x_test']
    test_label = mnist['y_test']

# 将数据和其标签放到一个文件内
train_data = tf.data.Dataset.from_tensor_slices((train, train_label))
test_data = tf.data.Dataset.from_tensor_slices((test, test_label))

# 将数据打乱和批次化(这是针对训练数据使用)
shuffle_size = 100
batch_size = 64
train_data = train_data.shuffle(shuffle_size).batch(batch_size)
test_data = test_data.batch(batch_size)

def model_step():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)), # 将数据铺平
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.RMSprop(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    history = model.fit(
        train_data,
        epochs=10,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='binary_crossentropy', patience=20)]
    )
    return history
history = model_step()
print(pd.DataFrame(history.history))
"""

# DataFrame 数据
"""
DATA_URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
heart_path = tf.keras.utils.get_file('heart.csv', DATA_URL)

heart_datas = pd.read_csv(heart_path, na_values='?')
print(heart_datas.head())
# 查看数据后将thal列转化为数值(离散值)
heart_datas['thal'] = pd.Categorical(heart_datas['thal'])
heart_datas['thal'] = heart_datas.thal.cat.codes
# 这个函数承接上面的pd.Categorical 是一种类似于LabelEncode的功能
# 大体上都是将类别信息转化为了数字信息，不过 Categorical 是对自己编码
# 而LabelEncoder 是通过通过一个样本 制成标准 然后 对其他样本编码
# 因而pd.Categorical相对的更加灵活

# 将数据与标签结合，然后进行打乱(针对训练数据)和批次化 因为数据量小，就不进行批次化处理了
train_datas = heart_datas.sample(frac=0.8, random_state=123)
test_datas = heart_datas.drop(train_datas.index)
# 因为处理Dataframe因此不能使用skip
train_labels = train_datas.pop('target')
test_labels = test_datas.pop('target')
# 注意合并数据时要加.values
trains = tf.data.Dataset.from_tensor_slices((train_datas.values, train_labels.values))
tests = tf.data.Dataset.from_tensor_slices((test_datas.values, test_labels.values))
trains = trains.shuffle(len(train_datas)).batch(1)
print(trains)
tests = tests.shuffle(len(test_datas))
print(tests)
# BatchDataset shapes: ((None, 13), (None,)), types: (tf.float64, tf.int64)> 加了.batch()
# <ShuffleDataset shapes: ((13,), ()), types: (tf.float64, tf.int64)> 没加.batch()


def model_step():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model
model = model_step()
history = model.fit(
    trains,
    epochs=5,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='binary_crossentropy', patience=5)]
)
print("this is model1\n", model.summary())
print(pd.DataFrame(history.history))


# 直接使用数据的特征作为模型的层也可以产生类似的效果
target = heart_datas.pop('target')
inputs = {key : tf.keras.layers.Input(shape=(), name=key) for key in heart_datas.keys()}
print("this inputs\n", inputs)
x = tf.stack(list(inputs.values()), axis=1)
print("this x1 \n", x)
x = tf.keras.layers.Dense(10, activation='relu')(x)
print("this x2 \n", x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
print("this output \n", outputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    loss='binary_crossentropy',
    opytimizer='adam',
    metrics=['accuracy']
)
dict_slices = tf.data.Dataset.from_tensor_slices((heart_datas.to_dict('list'), target.values)).batch(16)
# .to_dict('list') 将columns变为keys， values 变为list
# {'age':[1,2,3,4...], 'sex':[1,2,3,4...]..}这种形式 
history = model.fit(dict_slices, epochs=5)
print(pd.DataFrame(history.history))
print("this is model2\n", model.summary())
"""

# 图片 数据
"""
AUTOTUNE = tf.data.experimental.AUTOTUNE
# 根据可用的CPU动态设置并行调用的数量
import pathlib
data_root_orig = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                         fname='flower_photos', untar=True)
data_root = pathlib.Path(data_root_orig)

all_image_paths = list(data_root.glob('*/*')) # 类似于正则表达式 将所有的path link 获取
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)
print(all_image_paths[:10])

attributions = (data_root/"LICENSE.txt").open(encoding='utf-8').readlines()[4:]
attributions = [line.split(' CC-BY') for line in attributions]
# 将图片的名称、类型和地址切割出来
attributions = dict(attributions)
def caption_image(image_path):
    image_rel = pathlib.Path(image_path).relative_to(data_root)
    # relative_to 返回pathlib.Path(image_path)相对于data_root的路径,长路径变为短路径
    image_rel = str(image_rel).replace("\\", "/")
    return "Image (CC BY 2.0) " + ' - '.join(attributions[str(image_rel)].split(' - ')[:-1])
'''
import IPython.display as display
#  display.display(display.Image(image_path))
#  也可以显示图片但是没有下面的方法简便
for n in range(3):
    image_path = random.choice(all_image_path)
    lena = mpimg.imread(image_path)
    plt.imshow(lena)
    plt.show()
    print(caption_image(image_path))
'''

labels = sorted([item.name for item in data_root.glob('*/') if item.is_dir()])
# item 如果是..\..\..\roses那么
# item.name就是roses， 将花的种类排序
labels_to_index = {label:value for value, label in enumerate(labels)}

all_image_labels = [labels_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]
print(all_image_labels)

# pathlib.Path(path).parent.name 获取文件的父文件 如下文件序列
# a\b\c\d\tang.txt 则可以获取到d 而这里的d文件名就是花的类型名

def process_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image = image / 255.0
    return image
'''
image_path = all_image_paths[0]
image_label = all_image_labels[0]
image = process_image(image_path)
plt.imshow(image)
plt.grid(False)
plt.xlabel(caption_image(image_path))
plt.title(labels[image_label])
plt.show()
# 没有问题
'''
image_datas_paths = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_datas = image_datas_paths.map(process_image)
'''
plt.figure(figsize=(8,8))
for n, image in enumerate(image_datas_labels.take(4)):
  plt.subplot(2,2,n+1)
  plt.imshow(image)
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.xlabel(caption_image(all_image_paths[n]))
plt.show()
# 没有问题
'''

# 既然我们把所有的images打包到了一起，那么也可以将image和labels也放到一起
image_datas_labels = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
images_datas_all = tf.data.Dataset.zip((image_datas, image_datas_labels))
# for value in image_datas_labels.take(10):
#     print(labels[value.numpy()])
# for value in images_datas_all.take(10):
#     print(value)
#     # 可以观察到图片数据和numpy-标签数据已经结合到了一起

ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
# 元组被解压缩到映射函数的位置参数中
def load_and_preprocess_from_path_label(path, label):
  return process_image(path), label

images_datas_all = ds.map(load_and_preprocess_from_path_label)

batch_size = 32
# ds = images_datas_all.shuffle(buffer_size = len(all_image_paths))
# ds = ds.repeat()
# 数据集重复了指定次数
# 一般情况下repeat()在batch操作输出完毕后再执行相当于把整个数据集复制n次
# 为了配合输出次数，一般默认repeat()空，复制两次
# 考虑到repeat()的功能是重复数据，因此当使用shuffle时要注意先后顺序
# 然而repeat()会导致在数据集使用结束之前，被打乱的数据集不会报告数据集的结尾
# Dataset（数据集）由 .repeat 重新启动，导致需要再次等待随机缓冲区被填满。
# 因此使用tensorflow 内置的包含shuffle和repeat

ds = images_datas_all.cache(filename='./cache.tf-data')
# 加这一行代码可以确保内存不足时有一个缓存文件
# 缓存文件快速重启数据集而无需重建缓存的优点
ds = ds.apply(
    tf.data.experimental.shuffle_and_repeat(buffer_size=len(all_image_paths))
)
ds = ds.batch(batch_size) # 最后一次输入可能少于batch_size
ds = ds.prefetch(buffer_size=AUTOTUNE)
# 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch

model_v2 = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
# 使用该层，需要设置传入数据的格式符合要求
# 因此需要标准化，将图片内点的取值范围从[0, 1], 转为[-1, 1]
model_v2.trainable=False
print(help(tf.keras.applications.mobilenet_v2.preprocess_input))
'''
该函数使用“Inception”预处理，将
'''
def change_image(image, label):
    return 2*image-1, label
keras_datas = ds.map(change_image)
image_batch, image_label = next(iter(keras_datas))
model = tf.keras.Sequential([
    model_v2,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(labels_to_index), activation='softmax')
])

logit_batch = model(image_batch).numpy()
print("min logit:", logit_batch.min())
print("max logit:", logit_batch.max())
print()
print("Shape:", logit_batch.shape)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())
print(len(model.trainable_variables))

# 真实情况下的steps_per_epoch是这样子计算的
steps_per_epoch=tf.math.ceil(len(all_image_paths)/batch_size).numpy()
# ceil函数 向上取整
# 为了便于展示，设置steps_per_epoch为3

# history = model.fit(
#     images_datas_all,
#     epochs=1, steps_per_epoch=3
# )
# print(pd.DataFrame(history.history))

import time
default_time_steps = steps_per_epoch*2+1
def timeit(datas, steps=default_time_steps):
    time_start = time.time()
    it = iter(datas.take(steps+1)) # .take() 获取一部分数据
    next(it)

    data_start_time = time.time()
    for i, (images, labels) in enumerate(it):
        if i % 10 == 0:
            pass
        data_end_time = time.time()
    duration = data_end_time-data_start_time
    print("{} batches: {} s".format(steps, duration))
    print("{:0.5f} Images/s".format(batch_size * steps / duration))
    print("Total time: {}s".format(data_end_time - time_start))
ds = images_datas_all.cache()
ds = ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=len(all_image_paths)))
ds = ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
print("this is 1:", ds)
# timeit(datas=ds)

# 以上是对图片数据进行直接的处理，也可以选择将数据转化为TFRecord
# 这样子可以一次性读入多个实例，并且提升性能

record_image_paths = tf.data.Dataset.from_tensor_slices(all_image_paths).map(tf.io.read_file)
# 获取连接
record_image_file = tf.data.experimental.TFRecordWriter(
    'image.tfrec'
)
record_image_file.write(record_image_paths)
# 写入tfrec文件

image_ds = tf.data.TFRecordDataset('image.tfrec').map(process_image)
# 读取并处理文件
ds = tf.data.Dataset.zip((image_ds, image_datas_labels))
print(len(all_image_paths))
ds = ds.apply(
    tf.data.experimental.shuffle_and_repeat(buffer_size=len(all_image_paths))
)
ds = ds.batch(batch_size).prefetch(AUTOTUNE)
print(ds)
# timeit(datas=ds)


path_datas = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_datas = path_datas.map(process_image)
ds = image_datas.map(tf.io.serialize_tensor)
# 将tensor数据转化为字符串集
# <MapDataset shapes: (), types: tf.string>
print(ds)
tfrec = tf.data.experimental.TFRecordWriter('imageds.tfrec')
tfrec.write(ds)
# 数据缓存
ds = tf.data.TFRecordDataset('imageds.tfrec')
# 提取数据并且反序列化
def parse(datas):
    datas = tf.io.parse_tensor(datas, out_type=tf.float32)
    datas = tf.reshape(datas, [192, 192, 3])
    return datas
ds = ds.map(parse, num_parallel_calls=AUTOTUNE)
ds = tf.data.Dataset.zip((ds, image_datas_labels))
ds = ds.apply(
    tf.data.experimental.shuffle_and_repeat(buffer_size=len(all_image_paths))
)
ds = ds.batch(batch_size).prefetch(AUTOTUNE)
timeit(datas=ds)
"""

# 文本 数据
"""
# TextLineDataset 通常被用来以文本文件构建数据集（原文件中的一行为一个样本)
# 这适用于大多数的基于行的文本数据（例如，诗歌或错误日志)
DIRECTORY_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
FILE_NAMES = ['cowper.txt', 'derby.txt', 'butler.txt']

data_file_path = tf.keras.utils.get_file(origin=DIRECTORY_URL+FILE_NAMES[0],
                                         fname=FILE_NAMES[0])
parent_file_path = os.path.dirname(data_file_path)

def mark_file(data, label):
    return data, tf.cast(label, tf.int64)

label_datas = []
for i, file_name in enumerate(FILE_NAMES):
    lines_datas = tf.data.TextLineDataset(os.path.join(parent_file_path, file_name))
    # print(lines_datas)
    # <TextLineDatasetV2 shapes: (), types: tf.string>
    # print(os.path.join(parent_file_path, file_name))
    # C:/Users/tzl17/.keras/datasets/butler.txt
    label_data = lines_datas.map(lambda x: mark_file(x, i))
    # print(label_data)
    # <MapDataset shapes: ((), ()), types: (tf.string, tf.int64)>
    label_datas.append(label_data)

all_label_datas = label_datas[0]
for label_data in label_datas[1:]:
    all_label_datas = all_label_datas.concatenate(label_data)

Buffer_size = 50000
Batch_size = 64
Take_size = 5000
all_label_datas = all_label_datas.shuffle(
    Buffer_size,
    reshuffle_each_iteration=False
    # 为true 则表示每次迭代数据集时都应进行伪随机重排（默认为True）
)
# for label_data in all_label_datas.take(5):
#     print(label_data)

# 构建单词-数字的映射表
tokenzer = tfds.features.text.Tokenizer()
# Tokenizer.tokenize函数可以将句子转化为切割后的单词列表
# 然后配合tfds.features.text.TokenTextEncoder 可以构建出相应的单词-数字映射表
vocabulary_set = set()
for text_tensor, _ in all_label_datas:
    tokens_text = tokenzer.tokenize(text_tensor.numpy())
    # 将tensor 直接变为切割后的列表
    vocabulary_set.update(tokens_text)
    # 检查可知道 列表内不包含重复值
    # 既然是列表也可以使用set进行去除

vocab_size = len(vocabulary_set)
# tfds.features.text.TokenTextEncoder 将单词列表变为数字
encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)
example_text = next(iter(all_label_datas))[0].numpy()
example_nums = encoder.encode(example_text)
# b'As by the rising north-wind driven ashore'
# [154, 92, 150, 45, 182, 21, 403, 88]
print(all_label_datas)
# <ShuffleDataset shapes: ((), ()), types: (tf.string, tf.int64)>
def encode_text(text_tensor, label):
    text = encoder.encode(text_tensor.numpy())
    return text, label
def encoder_map_fn(text, label):
    encoder_text, label = tf.py_function(
        func=encode_text,
        inp=[text, label], # Tensor对象列表
        Tout=(tf.int64, tf.int64)
        # 张量流数据类型的列表或元组 或者如果只有一个张量流数据类型
        # 则指示一个func返回值 如果没有返回值（即返回值为None）则为空列表
    )
    # tf.py_function 将python函数包装到TensorFlow op中，并执行该函数
    encoder_text.set_shape([None])
    label.set_shape([])
    return encoder_text, label
all_encoder_datas = all_label_datas.map(encoder_map_fn)
print(all_encoder_datas)
# <MapDataset shapes: ((None,), ()), types: (tf.int64, tf.int64)>

# 每一行的文本长度不一致，因此需要为其填充，确保长度一致
# 这里使用 tf.data.Dataset.padded_batch
# 还可以使用tf.keras.preproocessing.sequence.pad_sequences
train_datas = all_encoder_datas.skip(Take_size).shuffle(Buffer_size)
train_datas = train_datas.padded_batch(batch_size=Batch_size,
                                       padded_shapes=([None],[]))
# padded_shapes 需要确保和输入的tensor一致，在前面我们设置encode_text 为[None], label 为[]
# 因此在这里也需要设置padded_shapes为([None],[])

test_datas = all_encoder_datas.take(Take_size)
test_datas = test_datas.padded_batch(batch_size=Batch_size,
                                     padded_shapes=([None],[]))

example_test_text, example_test_label = next(iter(test_datas))
print(example_test_text[0], example_test_label[0])

# 由于我们引入了一个新的 token 来编码（填充零），因此词汇表大小增加了一个
vocab_size += 1
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 64))

# 下一层是 LSTM 层它允许模型利用上下文中理解单词含义
# LSTM 上的双向包装器有助于模型理解当前数据点与其之前和之后的数据点的关系
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
# tf.keras.layers.Bidirectional Rnn层循环神经网络层
# RNN 擅长对序列数据进行建模处理
# 添加两层全连接层
for units in [64, 64]:
  model.add(tf.keras.layers.Dense(units, activation='relu'))

model.add(tf.keras.layers.Dense(3, activation='softmax'))
# 三个种类，所以参数为3
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_datas, epochs=3, validation_data=test_datas)
eval_loss, eval_acc = model.evaluate(test_datas)

print('Eval loss: {}, Eval accuracy: {}'.format(eval_loss, eval_acc))
"""

# Unicode 字符串 数据
"""
test = tf.constant(u'thank your ?')
test = tf.constant([u"You're", u"welcome!"]).shape
test_utf8 = tf.constant(u'深度学习')
test_utf16 = tf.constant(u'深度学习'.encode("UTF-16-BE"))
test_chars = tf.constant([ord(ur) for ur in u'深度学习'])
# test = tf.constant(ord(u'深度学习')) ord() 只针对长度为1的字符
# tf.strings.unicode_decode 将编码的字符串标量转换为代码点的向量(数字之类的)
# tf.strings.unicode_encode 将代码点的向量(数字之类的)转换为编码的字符串标量
# tf.strings.unicode_transcode 将编码的字符串标量转换为其他编码
test_utf8_decode = tf.strings.unicode_decode(test_utf8,
                                             input_encoding='UTF-8')
print(test_utf8, "test_utf8_decode: ", test_utf8_decode)
test_char_encode = tf.strings.unicode_encode(test_chars,
                                             output_encoding='UTF-8')
print(test_chars, "test_char_encode: ", test_char_encode)
test_utf8_unicode_transcode = tf.strings.unicode_transcode(test_utf8,
                                                           input_encoding='UTF8',
                                                           output_encoding='UTF-16-BE')
print("test_utf8_unicode_transcode: ",test_utf8_unicode_transcode)

batch_utf8 = [u'h?llo',  u'What is the weather tomorrow',  u'G??dnight', u'?']
# 当需要使用unicode_decode时最好先统一类型
batch_utf8 = [s.encode('UTF-8') for s in batch_utf8]
batch_chars_ragged = tf.strings.unicode_decode(
    batch_utf8, input_encoding='UTF-8'
)
# 转化为数字类型的向量时，可以直接使用to_tensor 填充相同长度, to_sparse 将数据合并到一起(不会消除重复值)
# 将其复原为Raggedtensor时直接使用tf.RaggedTensor.from_ + tensor/sparse
# 填充为相同长度
batch_chars_padded = batch_chars_ragged.to_tensor(default_value=-1)
print("batch_chars_ragged: ", batch_chars_ragged)
print("batch_chars_padded: ", batch_chars_padded.numpy())
# 将数据转化为密集型to_sparse() ps: 不会消除重复值,自动忽略空值
# s = tf.constant([[1, 2, 3], [4], [], [5, 6]])
# print(s.to_sparse())
# ... values=tf.Tensor([1 2 3 4 5 6], shape=(6,), dtype=int32) ...
batch_chars_sparse = batch_chars_ragged.to_sparse()
# print("batch_chars_sparse:  ", batch_chars_sparse)
batch_nums_strings = tf.strings.unicode_encode([[99, 97, 116], [100, 111, 103], [ 99, 111, 119]],
                                               output_encoding='UTF-8')
print("batch_nums_strings: ", batch_nums_strings)
batch_nums_strings_ragged = tf.strings.unicode_encode(
    batch_chars_ragged, output_encoding='UTF-8'
)
print("batch_nums_strings_ragged: ", batch_nums_strings_ragged)
batch_nums_strings_padded = tf.strings.unicode_encode(
    batch_chars_padded, output_encoding='UTF-8'
)
print("batch_nums_strings_padded: ", batch_nums_strings_padded)
# 可以看到填充的-1 转换为了格式为xxx的字符
# 经测试负数均为xxx该格式的字符
batch_ragged_chars_padded = tf.strings.unicode_encode(
    tf.RaggedTensor.from_tensor(batch_chars_padded, padding=-1),
    output_encoding='UTF-8'
)
print("batch_ragged_chars_padded: ", batch_ragged_chars_padded)
batch_ragged_chars_sparse = tf.strings.unicode_encode(
    tf.RaggedTensor.from_sparse(batch_chars_sparse),
    output_encoding='UTF-8'
)
print("batch_ragged_chars_sparse: ", batch_ragged_chars_sparse)

# 计算字符长度/数目 默认n个空格为n个字符
thanks = u'thanks  ?'.encode('UTF-8')
num_bytes = tf.strings.length(thanks).numpy()
num_chars = tf.strings.length(thanks, unit='UTF8_CHAR').numpy()
print('{} bytes; {} UTF-8 characters'.format(num_bytes, num_chars))
thanks_substr = tf.strings.substr(thanks, pos=2, len=2).numpy()
# 从pos开始长度为len的字符子串
print("thanks_substr: ", thanks_substr)
thanks_reggad = tf.strings.unicode_decode(thanks, input_encoding='UTF-8')
print("thanks_reggad: ", thanks_reggad)
# thanks_reggad_numbytes = tf.strings.length(thanks_reggad).numpy()
# thanks_reggad_numchars = tf.strings.length(thanks_reggad, unit='UTF8_CHAR').numpy()
# thanks_reggad_substr = tf.strings.substr(thanks_reggad, pos=1, len=3).numpy()
# print(thanks_reggad_substr)
# 只有字符串可以使用lenth, substr()等函数

# 将数据细分
sentence_texts = [u'Hello, tensorflow.', u'世界こんにちは']
sentence_texts_decode = tf.strings.unicode_decode(sentence_texts, 'UTF-8')
sentence_texts_script = tf.strings.unicode_script(sentence_texts_decode)
# unicode_script() 作用是帮助确定操作语言的所属
'''
uscript = tf.strings.unicode_script([33464, 1041])  # ['芸', 'Б']
print(uscript.numpy())  # [17, 8] == [USCRIPT_HAN 汉文, USCRIPT_CYRILLIC 西里尔文]
'''
print("1", sentence_texts_decode)
print("2", sentence_texts_script)
sentence_texts_stars_word = tf.concat(
    [tf.fill([sentence_texts_script.nrows(), 1], True),
     # sentence_texts_script.nrows() 相当于获取它的行数
     # 这一行相当于tf.fill([2, 1], True) 填充为True
     tf.not_equal(sentence_texts_script[:, 1:], sentence_texts_script[:, :-1])],
     # 同类型数据对比(注意是类型), 相同则为False, 不同则为True
    axis=1
)
print("sentence_texts_stars_word", sentence_texts_stars_word.values)
word_starts = tf.squeeze(tf.where(sentence_texts_stars_word.values), axis=1)
# tf.where(sentence_texts_stars_word.values) 找到值为True的位置
# tf.squeeze()从张量形状中移除大小为1的维度
# 例如tf.where(sentence_texts_stars_word.values)形状为(6,1) 格式为[[],[],[],[],[],[]]
# 使用tf.squeeze 后则变为形状(6,) 格式为[x,x,x,x,x,x]
sentence_texts_char_codepoint = tf.RaggedTensor.from_row_starts(
    values=sentence_texts_decode.values,
    # 需要拆分的值
    row_starts=word_starts
    # 拆分的位置, 或者说拆分的依据
)
# 相当于对一组没有标点符号的单词加了逗号将每个小句子划分出来
print(sentence_texts_char_codepoint)
sentence_texts_num_words = tf.reduce_sum(
    # 计算一个张量的各个维度上元素的总和
    tf.cast(sentence_texts_stars_word, tf.int64),
    # True = 1  Flase = 0
    axis=1
)
# tf.Tensor([4 2], shape=(2,), dtype=int64)
sentence_texts_word_char_codepoint = tf.RaggedTensor.from_row_lengths(
    values=sentence_texts_char_codepoint,
    row_lengths=sentence_texts_num_words
)
# 按照每个段落拥有的个数，在sentence_texts_char_codepoint基础上
# 将同属于一个段落的句子合并到一块
# 相当于加了句号，将每个段落划分出来
print(sentence_texts_word_char_codepoint)
sentence_texts_words = tf.strings.unicode_encode(sentence_texts_word_char_codepoint, 'UTF-8').to_list()
print(sentence_texts_words)
"""

# TFRecord和tf.Example 数据
"""

# TFRecord格式是一种用于存储二进制记录序列的简单格式
# tf.Example类似于字典类型{'string': value} 严格来说value指的是tf.train.Feature
# 因此tf.Example 主要针对tf.train.Feature做文章
# tf.train.Feature 只要接受string/byte, int之类, flota之类, bool这几类型数据
# tf.train.Feature:

def _bytes_feature(value):
    # if isinstance(value, (int, str, bool)):
    #     print("不是标量")
    #     pass
    # else:
    #     print("初始值及其类型: ", value)
    #     value = tf.io.serialize_tensor(value)
    #     print("转化成的标量字符串: ", value)
    #     # print("标量字符串转换的原始值: ", tf.io.parse_tensor(value, out_type=tf.int32))
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
print(tf.version)
'''
# 该三个函数均为标量输入,处理非标量的方法是使用tf.io.serialize_tensor
# 将其转化为二进制字符串,字符串是张量流中的标量,之后若要返回
# 则使用tf.parse_tensor 转换二进制字符串到张量
# 检测输入是否是标量可以使用numpy的

# print(_bytes_feature(tf.constant([12,3,4], dtype=tf.int32)))
print(_bytes_feature(u'test_bytes'.encode('utf-8')))

print(_float_feature(np.exp(1)))

# print(_int64_feature(True))
# print(_int64_feature(1))
# feature = _float_feature(np.exp(1))
# print(feature)
# print(type(np.exp(1)))
# feature_str = feature.SerializeToString()
# 序列化为二进制字符串,注意无法使用tf.io.parse_tensor转化

n_observations = int(1e4)

feature0 = np.random.choice([False, True], n_observations) # bool
feature1 = np.random.randint(0, 5, n_observations) # int
strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
feature2 = strings[feature1] # strings
feature3 = np.random.randn(n_observations) # flota
def serialize_example(f0, f1, f2, f3):
    features = {
        'feature0': _int64_feature(f0),
        'feature1': _int64_feature(f1),
        'feature2': _bytes_feature(f2),
        'feature3': _float_feature(f3)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=features))
    return example_proto.SerializeToString()
    # 序列化为二进制字符串

example_observation = []
serialized_examples = serialize_example(False, 4, b'goat', 0.9876)
# 此处的解码函数为tf.train.Example.FromString
print(tf.train.Example.FromString(serialized_examples))

tf_data_features = tf.data.Dataset.from_tensor_slices((feature0, feature1,
                                                    feature2, feature3))
for f0, f1, f2, f3 in tf_data_features.take(1):
    print(f0.numpy(), f1.numpy(), f2.numpy(), f3.numpy())
def tf_serialize_example(f0,f1,f2,f3):
    tf_string = tf.py_function(
      serialize_example,
      (f0,f1,f2,f3),  # pass these args to the above function.
      tf.string)      # the return type is `tf.string`.
    return tf.reshape(tf_string, ())

serialized_features_dataset = tf_data_features.map(tf_serialize_example)
print(serialized_features_dataset)

def generator():
    for feature in tf_data_features:
        yield serialize_example(*feature)
serialized_features_dataset = tf.data.Dataset.from_generator(
    generator, output_types=tf.string, output_shapes=()
)
print(serialized_features_dataset)
filenames = 'test.tfrecord'
writer = tf.data.experimental.TFRecordWriter(filenames)
writer.write(serialized_features_dataset)

filenames = [filenames]
raw_dataset = tf.data.TFRecordDataset(filenames)

for raw_record in raw_dataset.take(10):
    print(repr(raw_record))
# 解析数据

# 描述features的类型和形状-基本信息
feature_description = {
    'feature0' : tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature1' : tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature2' : tf.io.FixedLenFeature([], tf.string, default_value=''),
    'feature3' : tf.io.FixedLenFeature([], tf.float32, default_value=0.0)
}
def _parse_features(example_proto):
    return tf.io.parse_single_example(example_proto, features=feature_description)
    # tf.io.parse_single_example 用于解析单个Example,注意是单个Example
    # tf.io.parse_example 用于解析批量Example
parse_raw_dataset = raw_dataset.map(_parse_features)
# parse_raw_dataset = tf.io.parse_example(raw_dataset.take(10),
#                                         features=feature_description)
for parsed_record in parse_raw_dataset.take(10):
    print(repr(parsed_record))
    # repr() 函数将对象转化为供解释器读取的形式
'''

# python中的Tfrecord
# tf.io 模块包含用于读取和写入TFRcord的纯Python函数
'''
py_filenames = "py_test.tfrecord"
with tf.io.TFRecordWriter(py_filenames) as write:
    for i in range(n_observations):
        example = serialize_example(feature0[i], feature1[i],
                                    feature2[i], feature3[i])
        write.write(example)

py_filenames = [py_filenames]
py_raw_datas = tf.data.TFRecordDataset(py_filenames)
for i in py_raw_datas.take(3):
    print("初始1", i)
    print("初始", repr(i))
    example = tf.train.Example()
    example.ParseFromString(i.numpy()) # 对于序列化的数据可以使用该函数直接解码
    print("变化", example)
'''

# 处理图片数据将其变为TFRecord

cat_on_snow  = tf.keras.utils.get_file('320px-Felis_catus-cat_on_snow.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg')
williamsburg_bridge = tf.keras.utils.get_file('194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg')

cat_image = mpimg.imread(cat_on_snow)
williamsburg_image = mpimg.imread(williamsburg_bridge)
plt.subplots(1, 2)
plt.subplot(1, 2, 1)
plt.imshow(cat_image)
plt.subplot(1, 2, 2)
plt.imshow(williamsburg_image)

image_labels = {
    cat_on_snow : 0,
    williamsburg_bridge : 1
}
def image_features(labels, image_path):
    image_string = open(image_path, 'rb').read()
    label = labels[image_path]
    image_shapes = tf.image.decode_jpeg(image_string).shape
    # height, width, depth 对应 image_shapes
    features = {
        'height' : _int64_feature(image_shapes[0]),
        'width' : _int64_feature(image_shapes[1]),
        'depth' : _int64_feature(image_shapes[2]),
        'label' : _int64_feature(label),
        'image_raw' : _bytes_feature(image_string)
    }
    return tf.train.Example(features = tf.train.Features(feature=features))

image_names = {
    'cat_on_snow.tfrecord' : cat_on_snow,
    'will_brage.tfrecord': williamsburg_bridge
}
# for name, pathe in image_names.items():
#     with tf.io.TFRecordWriter(name) as write:
#         features = image_features(image_labels, pathe)
#         write.write(features.SerializeToString())
#         # .SerializeToString方法将所有原始消息序列化为二进制字符串 即序列化为标量

image_features_description = {
    'height' : tf.io.FixedLenFeature([], tf.int64),
    'width' : tf.io.FixedLenFeature([], tf.int64),
    'depth' : tf.io.FixedLenFeature([], tf.int64),
    'label' : tf.io.FixedLenFeature([], tf.int64),
    'image_raw' : tf.io.FixedLenFeature([], tf.string),
}
def parse_image(data):
    return tf.io.parse_single_example(data, image_features_description)

plt.subplots()
image_datas = []
for names in image_names.keys():
    image_data = tf.data.TFRecordDataset(names)
    parse_image_data = image_data.map(parse_image)
    image_datas.append(parse_image_data)
plt.subplots(1, 2)
for i in range(len(image_datas)):
    for image_features in image_datas[i]:
        image_raw = image_features['image_raw'].numpy()
        image = tf.image.decode_jpeg(image_raw, channels=3)
    plt.subplot(1, 2, i+1)
    plt.imshow(image)
plt.show()
"""



# Estimators 完整模型的高级表示 Keras进阶
# Estimators的特征大致如下
# 特征的处理 : 数字特征, category特征, 对特征的标准化, 特征的组合等 api: tf.feature_columns.
# 输入的处理(管道建设) : 设置输入函数 对数据进行合并tf.data.Dataset.from_tensor_slices
# 数据的shuffle, repeat, batch
# 模型的可解释性 : experimental_predict_with_explanations可得出DFC值也可以得出预测值
# DFC的一个不错的特性是，贡献的总和+偏差等于给定示例的预测 这两者均可以从以上API的出

# estimators 的简单演示
"""
csv_columns_names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
species = ['Setosa', 'Versicolor', 'Virginica']

train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")
train_data = pd.read_csv(train_path, names=csv_columns_names, header=0)
test_data = pd.read_csv(test_path, names=csv_columns_names, header=0)

train_label = train_data.pop('Species')
test_label = test_data.pop('Species')

def input_fn(features, label, training=True, batch_size=256):
    datas = tf.data.Dataset.from_tensor_slices((
        dict(features), label
    ))
    if training:
        datas = datas.shuffle(1000).repeat()
    return datas.batch(batch_size)

# 构建输入特征,需要将分类和数字特征划分开来并放入一个列表内
my_feature_columns = []
for feature in train_data.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=feature))
    # 数字特征的话使用numeric_column, 分类特征的话使用category_xxxx
    
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[30, 10], # 隐藏层节点数说明隐藏层有两个且节点数分别为30 10
    n_classes=3 # 几个分类就是几
)

fit_result = classifier.train(
    input_fn=lambda : input_fn(train_data, train_label, training=True),
    steps=5000
)
print(fit_result)
eval_result = classifier.evaluate(
    input_fn=lambda : input_fn(test_data, test_label, training=False)
)
print('Test set accuracy: {accuracy:0.3f}'.format(**eval_result))
print(eval_result)

expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
}
def input_predict(features, batch_size=256):
    return tf.data.Dataset.from_tensor_slices((dict(features))).batch(batch_size)
predict_result = classifier.predict(
    input_fn=lambda : input_predict(predict_x)
)
for pred_dict, expec in zip(predict_result, expected):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(
        species[class_id], 100 * probability, expec))
    print(pred_dict)
"""

# estimators 线性模型
"""
from sklearn.metrics import roc_curve
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc

train_path = tf.keras.utils.get_file('train.csv', 'https://storage.googleapis.com/tf-datasets/titanic/train.csv')
eval_path = tf.keras.utils.get_file('eval.csv', 'https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

train = pd.read_csv(train_path)
eval = pd.read_csv(eval_path)

y_train = train.pop('survived')
y_eval = eval.pop('survived')

age = train['age'].value_counts()
sex = train['sex'].value_counts()
calss_rank = train['class'].value_counts()
sex_surive = pd.concat([train, y_train], axis=1).groupby('sex').survived.mean()
# .sum()可以获悉存活的男性和女性各有多人 .mean() 则可以获悉存货的男性女性的比例(相对于他们本身)
# sns.barplot(x=sex_surive.index, y=sex_surive.values)
category_columns = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                    'embark_town', 'alone']
numeric_columns = ['age', 'fare']
feature_columns = []
for col in category_columns:
    vocabulary = train[col].unique()
    feature_columns.append(
        tf.feature_column.categorical_column_with_vocabulary_list(col, vocabulary))
for col in numeric_columns:
    feature_columns.append(tf.feature_column.numeric_column(col, dtype=tf.float32))
# print(feature_columns)

# 将数据转换为tf.data.Dataset
def make_input_fn(datas, labels, num_epochs=10, shuffle=True, batch_size=32):
    def input_funcations():
        ds = tf.data.Dataset.from_tensor_slices((dict(datas), labels))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_funcations
# age_column = feature_columns[7]
# gender_column = feature_columns[0]
# for feature_batc, label in train_input_fn().take(1): # 注意这里有个小括号
#     print(feature_batc.keys(), label.numpy())
#     print(tf.keras.layers.DenseFeatures([age_column])(feature_batc).numpy())
#     # 可以使用tf.keras.layers.DenseFeatures图层检查特定要素列的结果
#     # DenseFeatures 仅接受密集张量(numeric) 要检查分类列，您需要先将其转换为指标列
#     # 使用[tf.feature_column.indicator_column(category_feature)
#     print(tf.keras.layers.DenseFeatures([tf.feature_column.indicator_column(gender_column)])(feature_batc).numpy())
train_input_fn = make_input_fn(train, y_train)
eval_input_fn = make_input_fn(eval, y_eval, num_epochs=1, shuffle=False)

liner_model = tf.estimator.LinearClassifier(feature_columns=feature_columns)
liner_model.train(train_input_fn)
result = liner_model.evaluate(eval_input_fn)
# 构建派生特征列
age_and_gender = tf.feature_column.crossed_column(['age', 'sex'],
                                                  hash_bucket_size=100)
derive_feature_columns = [age_and_gender]
liner_model_s = tf.estimator.LinearClassifier(feature_columns=feature_columns+derive_feature_columns)
liner_model_s.train(train_input_fn)
result = liner_model_s.evaluate(eval_input_fn)

predict = list(liner_model_s.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in predict])
print(roc_curve(y_eval, probs))
fpr, tpr, _ = roc_curve(y_eval, probs)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.xlim(0,)
plt.ylim(0,)
plt.show()
"""

# 提升树 Boosted Trees estimators
"""
train_path = tf.keras.utils.get_file('train.csv', 'https://storage.googleapis.com/tf-datasets/titanic/train.csv')
eval_path = tf.keras.utils.get_file('eval.csv', 'https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

train = pd.read_csv(train_path)
eval = pd.read_csv(eval_path)

y_train = train.pop('survived')
y_eval = eval.pop('survived')
# plt.subplots(2, 2)
# plt.subplot(2, 2, 1)
# train['age'].hist(bins=20)
# plt.subplot(2, 2, 2)
# train['sex'].value_counts().plot(kind='barh')
# plt.subplot(2, 2, 3)
# train['class'].value_counts().plot(kind='barh')
# plt.subplot(2, 2, 4)
# pd.concat([train, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survived')

category_features = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                     'embark_town', 'alone']
numeric_features = ['age', 'fare']
# 将ceategory列转换为指标列
feature_columns = []
def one_hot_features(feature, vocab):
    return tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(feature, vocab)
    )
for feature in category_features:
    vocab = train[feature].unique()
    feature_columns.append(one_hot_features(feature, vocab))
for feature in numeric_features:
    feature_columns.append(tf.feature_column.numeric_column(feature, dtype=tf.float32))

# example_feature = dict(train.head(1))
# print(example_feature)
# print(tf.keras.layers.DenseFeatures(feature_columns)(example_feature).numpy())

# 创建输入流

nume_sizes = len(y_train) // 4
def make_input_fn(datas, labels, shuffle=True, n_epochs=None):
    def input_fn():
        ds = tf.data.Dataset.from_tensor_slices((dict(datas), labels))
        if shuffle:
            ds = ds.shuffle(nume_sizes)
        # 对于训练, 可以按需多次循环数据集n_epochs=None
        ds = ds.batch(nume_sizes).repeat(n_epochs)
        return ds
    return input_fn
train_input_fn = make_input_fn(train, y_train)
eval_input_fn = make_input_fn(eval, y_eval, shuffle=False, n_epochs=1)


print(" 线性模型创建 ")
age_and_gender = tf.feature_column.crossed_column(['sex', 'age'], hash_bucket_size=20)
dervie_feature = [age_and_gender]
liner_model = tf.estimator.LinearClassifier(feature_columns+dervie_feature)
liner_model.train(train_input_fn, max_steps=100)
liner_result = liner_model.evaluate(eval_input_fn)
print(liner_result)

n_batch = 1 # 意味着使用全部数据
boost_tree_model = tf.estimator.BoostedTreesClassifier(feature_columns,
                                                       n_batches_per_layer=n_batch)
print(" 提升树模型 ")
boost_tree_model.train(train_input_fn, max_steps=20) # 建立n棵树
tree_result = boost_tree_model.evaluate(eval_input_fn)
print(tree_result)

from sklearn.metrics import roc_curve
def roc_auc_plot(predict, label):
    fpr, tpr, _ = roc_curve(label, predict)
    plt.plot(fpr, tpr)
    plt.title('ROC curve')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.xlim(0, )
    plt.ylim(0, )
plt.subplots(1, 2)
plt.subplot(1, 2, 1)
liner_predict = liner_model.predict(eval_input_fn)
probs = pd.Series([pred['probabilities'][1] for pred in liner_predict])
roc_auc_plot(probs, y_eval)
tree_predict = boost_tree_model.predict(eval_input_fn)
plt.subplot(1, 2, 2)
probs = pd.Series([pred['probabilities'][1] for pred in tree_predict])
roc_auc_plot(probs, y_eval)
plt.show()
"""

# 提升树 模型的理解,解释方法 estimators
"""
# 获取数据, train数据和eval数据依然使用之前提升树内的数据
# 训练模型

params = {
    'n_trees': 50,
    'max_depth': 3,
    'n_batches_per_layer': 1,
    'center_bias': True
}
# tree_model = tf.estimator.BoostedTreesClassifier(feature_columns, **params)
# tree_model.train(train_input_fn, max_steps=100)
# tree_result = tree_model.evaluate(eval_input_fn)
# print(pd.Series(tree_result).to_frame())
# print("tree result ", tree_result)

in_many_params = dict(params)
in_many_params['n_batches_per_layer'] = 1

def new_make_input_fn(X, y):
    y = np.expand_dims(y, axis=1)
    # expand_dims 扩展数组的形状, axis=1 表示在位置1处添加虚假数据
    # 比如说一个数组的形状是(1, 2, 3) 则axis=1, 2, 3时它的形状就变为了
    # (1, 1, 2, 3), (1, 2, 1, 3), (1, 2, 3, 1)
    def input_fn():
        return dict(X), y
    return input_fn
new_train_input_fn = new_make_input_fn(train, y_train)
# print("train ", train)
# print("dict train ", dict(train))
# print("y train ", y_train)
# print("expand_dims y train ", np.expand_dims(y_train, axis=1))

new_tree_model = tf.estimator.BoostedTreesClassifier(
    feature_columns,
    train_in_memory=True,
    **in_many_params
)

new_tree_model.train(new_train_input_fn, max_steps=10)
new_result = new_tree_model.evaluate(eval_input_fn)

sns_colors = sns.color_palette('colorblind')
pred_dicts = list(new_tree_model.experimental_predict_with_explanations(eval_input_fn))
# 根据示例计算模型的可解释性输出
# 以及可以获取预测值

labels = y_eval.values
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])
# 这个是我们正常情况下需要的预测值 probs = dfc.sum(axis=1) + pred_dicts[0]['bias]
df_dfc = pd.DataFrame([pred['dfc'] for pred in pred_dicts])
# DFC : directional feature contributions  这个是我们获取的用于可解释性的输出
# 简单来说就是描述特征对模型预测功能的重要性
# print("pred_dicts", pred_dicts)
# print(df_dfc.describe().T)
# print("df_dfc", df_dfc)
# 关于DFC 还有一个特性是 # Sum of DFCs + bias == probabality
bias = pred_dicts[0]['bias']
sum_of_dfc = pd.DataFrame([pred['dfc'] for pred in pred_dicts]).sum(axis=1)
probabality = sum_of_dfc + bias
# print("probabality", probabality)
# print("probs", probs)
# 可以发现probs 和 probabality 是相同的

def _get_color(value):
    green, red = sns.color_palette()[2:4] # 调色盘, 内含有多种颜色
    if value >= 0 : return green
    return red

def _add_features_values(feature_values, ax):
    x_coord = ax.get_xlim()[0] # 返回当前视图x的上下限
    offset = 0.15
    for y_coord, (feature_name, feature_value) in enumerate(feature_values.items()):
        # 根据特征的xcoord(x坐标位置)和ycoord(y坐标位置)标记这个特征的名称, 值
        t = plt.text(x_coord, y_coord - offset, '{}'.format(feature_value), size=12)
        t.set_bbox(dict(facecolor='white', alpha=0.5))
        # bbox设置目的内容的透明度相关参数
        # facecolor调节 box 前景色，edgecolor 设置边框，alpha设置透明度
    from matplotlib.font_manager import FontProperties
    font = FontProperties()
    font.set_weight('bold')
    t = plt.text(x_coord, y_coord + 1 - offset, 'feature\nvalue',
                 fontproperties=font, size=12)

def _plot_example(example):
    TOP_N = 8  # View top 8 features.
    sorted_ix = example.abs().sort_values()[-TOP_N:].index  # Sort by magnitude.
    example = example[sorted_ix]
    colors = example.map(_get_color).tolist()
    print(colors)
    ax = example.to_frame().plot(kind='barh',
                                 color=[colors],
                                 legend=None,
                                 alpha=0.75,
                                 figsize=(10, 6))
    ax.grid(False, axis='y')
    ax.set_yticklabels(ax.get_yticklabels(), size=14)
    # Add feature values.
    _add_features_values(eval.iloc[ID][sorted_ix], ax)
    return ax
ID = 182
example = df_dfc.iloc[ID]  # Choose ith example from evaluation set.
TOP_N = 8  # View top 8 features.
sorted_ix = example.abs().sort_values()[-TOP_N:].index
# ax = _plot_example(example)
# ax.set_title('Feature contributions for example {}\n pred: {:1.2f}; label: {}'.format(ID, probs[ID], labels[ID]))
# ax.set_xlabel('Contribution to predicted probability', size=14)
# plt.show()
# 正值则说明该特征对模型的预测功能有促进作用，负值则相反

# 绘制小提琴图
def dist_violin_plot(df_dfc, ID):
    # Initialize plot.
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Create example dataframe.
    TOP_N = 8  # View top 8 features.
    example = df_dfc.iloc[ID]
    ix = example.abs().sort_values()[-TOP_N:].index
    example = example[ix]
    example_df = example.to_frame(name='dfc')

    # Add contributions of entire distribution.
    parts = ax.violinplot([df_dfc[w] for w in ix],
                          vert=False,
                          showextrema=False,
                          widths=0.7,
                          positions=np.arange(len(ix)))
    face_color = sns_colors[0]
    alpha = 0.15
    for pc in parts['bodies']:
        pc.set_facecolor(face_color)
        pc.set_alpha(alpha)

    # Add feature values.
    _add_features_values(eval.iloc[ID][sorted_ix], ax)

    # Add local contributions.
    ax.scatter(example,
               np.arange(example.shape[0]),
               color=sns.color_palette()[2],
               s=100,
               marker="s",
               label='contributions for example')

    # Legend
    # Proxy plot, to show violinplot dist on legend.
    ax.plot([0, 0], [1, 1], label='eval set contributions\ndistributions',
            color=face_color, alpha=alpha, linewidth=10)
    legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large',
                       frameon=True)
    legend.get_frame().set_facecolor('white')

    # Format plot.
    ax.set_yticks(np.arange(example.shape[0]))
    ax.set_yticklabels(example.index)
    ax.grid(False, axis='y')
    ax.set_xlabel('Contribution to predicted probability', size=14)

# dist_violin_plot(df_dfc, ID)
# plt.title('Feature contributions for example {}\n pred: {:1.2f}; label: {}'.format(ID, probs[ID], labels[ID]))
# plt.show()

# 全局功能的重要性 从整体上观察
'''
基于特征的重要性 tf.estimator.BoostedTreesClassifier.experimental_feature_importances
排列特征(置换特征)重要性
DFC tf.estimator.BoostedTreesClassifier.experimental_predict_with_explanations 可以获取DFC也可以获取predict
基于增益的特征重要性可衡量在拆分特定特征时的损耗变化
而排列特征(置换特征)的重要性是通过对评估集上的模型性能进行评估来计算的
方法是将每个特征一一改组并将模型性能的变化归因于改组特征。
一般而言，排列特征(置换特征)重要性要优先于基于增益的特征重要性
尽管在潜在的预测变量的测量规模或类别数量以及特征相关时两种方法都不可靠
'''
plt.subplots(2, 2)

# 基于特征的重要性观察
plt.subplot(2, 2, 1)
importances = new_tree_model.experimental_feature_importances(normalize=True)
df_imp = pd.Series(importances)
N = 8
ax = (df_imp.iloc[0:N][::1].plot( # [::-1] 将数组倒序排列
    kind='barh',
    color=sns_colors[0],
    title='Gain feature importances'
))
ax.grid(False, axis='y')

# 使用DFC的绝对值的平均值观察
plt.subplot(2, 2, 2)
dfc_abs_mean = df_dfc.abs().mean()
N = 8
dfc_sort_index = dfc_abs_mean.abs().sort_values()[-N:].index
ax = dfc_abs_mean[sorted_ix].plot(
    kind='barh',
    color=sns_colors[1],
    title='Mean |directional feature contributions|'
)

# 查看DFC随着指定特征的变化而变化
# 意思就是DFC随特征值的改变而改变
plt.subplot(2, 2, 3)
feature = 'fare'
feature = pd.Series(df_dfc[feature].values, # 从这行代码可以看出, DFC值针对每个特征都有对应的值
                    index=eval[feature].values).sort_index()
print("feature ", feature)
axs = sns.regplot(feature.index.values, feature.values, lowess=True)
axs.set_ylabel('contribution')
axs.set_xlabel('fare')
axs.set_xlim(0, 100)

# 排列特征(置换特征)的重要性
# 这里置换的含义可以理解为对某列特征进行洗牌的意思
plt.subplot(2, 2, 4)
def permutation_importances(tree_model, x_eval, y_eval, metric, features):
    base_line = metric(tree_model, x_eval, y_eval)
    # 获取到针对于x_eval和y_eval 的预测值的accuracy
    print("accuracy :", base_line)
    imp = []
    for col in features:
        # 里的代码的内容是 循环每一列特征,将该特征洗牌
        # 然后放入到eval数据并获取其accuracy
        # 然后用原始eval获取的acuuracy 减去 某列特征洗牌后的eval的accuracy
        # 同时在最后需要确保x_eval不发生变化
        save = x_eval[col].copy()
        x_eval[col] = np.random.permutation(x_eval[col])
        # np.random.permutation 功能上是对数组进行洗牌,类似于shuffle
        # 但是它不直接作用在原数组而是返回一个新的数组
        m = metric(tree_model, x_eval, y_eval)
        x_eval[col] = save
        # 整个过程中 x_eval 最后未发生变化
        print("col: ", base_line - m)
        imp.append(base_line - m)
    return np.array(imp)
def accuracy_metric(tree_model, X, y):
    eval_input_fn = make_input_fn(X, y,
                                  shuffle=False,
                                  n_epochs=1)
    return tree_model.evaluate(input_fn=eval_input_fn)['accuracy']
features = category_features + numeric_features
importances = permutation_importances(new_tree_model, eval, y_eval, accuracy_metric, features)
print("importances", importances)
df_imp = pd.Series(importances, index=features)
sorted_index = df_imp.abs().sort_values().index
axss = df_imp[sorted_ix][-5:].plot(
    kind='barh',
    color=sns_colors[2]
)
axss.grid(False, axis='y')
axss.set_title('Permutation feature importance')
plt.show()"""

# 从keras模型到estimators模型 keras_&_estimators
"""
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam'
)
print(model.summary())
def input_fn():
    split = tfds.Split.TRAIN
    ds = tfds.load('iris', split=split, as_supervised=True)
    ds = ds.map(lambda feature, label: ({'dense_input': feature}, label))
    # dense_input 是必要的名称而不能仅仅看做一个常量字符串
    ds = ds.batch(32).repeat()
    return ds

# for features_batch, labels_batch in input_fn().take(1):
#       print(features_batch)
#       print(labels_batch)

# 使用keras API 创建 estimator模型

model_dir = "/tmp/tfkeras_example/"
keras_estimator = tf.keras.estimator.model_to_estimator(
    keras_model=model, model_dir=model_dir)
keras_estimator.train(input_fn=input_fn, steps=25)
eval_result = keras_estimator.evaluate(input_fn=input_fn, steps=10)
print('Eval result: {}'.format(eval_result))
"""

# TensorFlow 高级功能

# 张量和操作以及GPU
"""
x = tf.random.uniform([3, 3])

print("Is there a GPU available: "),
print(tf.config.experimental.list_physical_devices("GPU"))

print("Is the Tensor on GPU #0:  "),
print(x.device.endswith('GPU:0'))

import time

def time_matmul(x):
  start = time.time()
  for loop in range(10):
    tf.matmul(x, x)

  result = time.time()-start

  print("10 loops: {:0.2f}ms".format(1000*result))

# Force execution on CPU
print("On CPU:")
with tf.device("CPU:0"):
  x = tf.random.uniform([1000, 1000])
  assert x.device.endswith("CPU:0")
  time_matmul(x)

# Force execution on GPU #0 if available
if tf.config.experimental.list_physical_devices("GPU"):
  print("On GPU:")
  with tf.device("GPU:0"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
    x = tf.random.uniform([1000, 1000])
    assert x.device.endswith("GPU:0")
    time_matmul(x)"""

# 自定义层
"""
# 层 最常见的就是Dense、Conv2D、LSTM、BatchNormalization、Dropout等
'''
exampl_model = tf.keras.Sequential([
    tf.keras.layers.Dense(), # 全连接层 用于综合特征
    tf.keras.layers.BatchNormalization(),
    # 批量标准化(归一化)处理 一般情况下放在Dense
    # 对每一批输入数据进行标准化(归一化)处理
    # 某种程度上标准化和归一化相似都是将数据所放在（0, 1）范围内
    tf.keras.layers.Conv2D(),
    # 卷积层 卷积运算的目的是提取输入的不同特征通过反向传播算法实现
    # 第一层卷积层可能只能提取一些低级的特征如边缘、线条和角等层级
    # 更多层的网路能从低级特征中迭代提取更复杂的特征
    tf.keras.layers.LSTM(),
    # 长短期记忆网络 是一种时间循环神经网络 是为了解决一般的RNN（循环神经网络）
    # LSTM适合于处理和预测时间序列中间隔和延迟非常长的重要事件
    # 作为非线性模型 可作为复杂的非线性单元用于构造更大型深度神经网络
    tf.keras.layers.Dropout()
    # 防止过拟合,一定程度上可以看做是l1正则化 即 忽略部分特征
])
'''

# 创建自定义层

class Mydenselayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(Mydenselayer, self).__init__()
        self.num_outputs = num_outputs
        # num_outputs 10

    def build(self, input_shape):
        # input_shape (10, 5)
        self.kernel = self.add_variable(
            'kernel',
            shape=[int(input_shape[-1]),
                   self.num_outputs]
        )
    def call(self, input):
        # input tf.zeros([10, 5])
        return tf.matmul(input, self.kernel)
layer = Mydenselayer(10)

_ = layer(tf.zeros([10, 5]))

# 组合层
class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name='')
        filter1, filter2, filter3 = filters
        self.conv2a = tf.keras.layers.Conv2D(filter1, (1, 1))
        self.bn1 = tf.keras.layers.BatchNormalization()
        # 标准化层
        self.conv2b = tf.keras.layers.Conv2D(filter2, kernel_size, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filter3, (1, 1))
        self.bn3 = tf.keras.layers.BatchNormalization()
    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn1(x, training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2(x, training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn3(x, training)
        x = x + input_tensor
        return tf.nn.relu(x)
block = ResnetIdentityBlock(1, [1, 2, 3])
_ = block(tf.zeros([1, 2, 3, 3]))

print(block.summary())
print(block.layers)

seq_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(1, (1, 1), input_shape=(None, None, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(2, 1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(3, (1, 1)),
    tf.keras.layers.BatchNormalization()
])
seq_model(tf.zeros([1, 2, 3, 3]))
print("seq_model summary ", seq_model.summary())
print("seq_model layers", seq_model.layers)

"""

# 自动微分 gradient tape
"""
x = tf.ones((2, 2))

with tf.GradientTape(persistent=True) as t:
    # persistent=True可以确保GradientTape长久的控制资源
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)
dz_dx = t.gradient(z, x)
# 调用 gradient()后GradientTape控制资源将会立即释放
for i in [0, 1]:
  for j in [0, 1]:
    assert dz_dx[i][j].numpy() == 8.0
    # 断言 dz_dx内的值是否均为8.0否则会出错
print(dz_dx)
dz_dy = t.gradient(z, y)
# 此时会失败的原因是在上面dz_dx已经将GradientTape的资源释放完毕
print(dz_dy)
assert dz_dy.numpy() == 8.0
# 若需要GradientTape保持持久以便实验可以为其加入persistent=True

def f(x, y):
    out_put = 10
    for i in range(y):
        if i > 1 and i < 5:
            out_put = tf.multiply(out_put, x)
    return out_put
def grade(x, y):
    with tf.GradientTape() as t:
        t.watch(x)
        out = f(x, y)
    return t.gradient(out, x)
x = tf.convert_to_tensor(2.0)
print(x)
assert grade(x, 6).numpy() == 120.0
assert grade(x, 5).numpy() == 120.0
assert grade(x, 4).numpy() == 40 # 类型尽量确保一致, 虽然不算错
"""

# 自定义训练
"""
'''
# TensorFlow变量
v = tf.Variable(3)
# TensorFlow的变量不可以直接赋予值
# 但是赋予之后该变量将会变成一个普通的变量而不是tf.Variable类型
# 因此TensorFlow的变量属于不可变无状态对象,同时也是内建可变状态操作
v = 4
# 此时相当于把v这个原本的TensorFlow变量 变成了一个值等于4的普通变量
# 使用v.assign()这个属于TensorFlow的状态改变函数时提示int 类型不可以使用assign函数
try:
    v.assign(4)
    print(v)
except:
    print("此时v已经不是TensorFlow变量")
# v.assign 是属于TensorFlow的变量改变函数
'''

# 自定义一个线性模型 f(x) = w*x + b

class LinerModel(object):
    def __init__(self):
        self.W = tf.Variable(5.0)
        self.B = tf.Variable(0.0)
    def __call__(self, x):
        return self.W * x + self.B
liner_m = LinerModel()
assert liner_m(3).numpy() == 15

def loss(predict, label):
    return tf.reduce_mean(tf.square(predict - label))
    # l2损失函数
TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000

inputs = tf.random.normal(shape=[NUM_EXAMPLES])
noise = tf.random.normal(shape=[NUM_EXAMPLES])
outputs = TRUE_W * inputs + TRUE_b
# plt.scatter(inputs, outputs, color='red')
# plt.scatter(inputs, liner_m(inputs), color='skyblue')
# plt.show()

# print(loss(liner_m(inputs), outputs))
def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        loss_value = loss(model(inputs), outputs)
    dW, dB = t.gradient(loss_value, [model.W, model.B]) # 可以直接写成model.trainable_variables
    # 释放GradentTape资源并按照提供的函数获取值
    model.W.assign_sub(learning_rate * dW)
    model.B.assign_sub(learning_rate * dB)
    # assign_sub() 用于更新value
list_W, list_B = [], []
epochs = range(10)
for epoch in epochs:
    list_W.append(liner_m.W.numpy())
    list_B.append(liner_m.B.numpy())
    loss_value = loss(liner_m(inputs), outputs)
    train(liner_m, inputs, outputs, learning_rate=0.1)
    print("epoch : {:d}, model.W : {:.5f}, model.B : {:.5f}, loss value: {:.5f}".format(
        epoch, list_W[-1], list_B[-1], loss_value
    ))
plt.plot(epochs, list_W, 'r',
         epochs, list_B, 'b')
plt.plot([TRUE_W] * len(epochs), 'r--',
         [TRUE_b] * len(epochs), 'b--')
plt.legend(['W', 'b', 'true W', 'true_b'])
# plt.show()
"""

# 自定义训练和演示
"""
print("动态图机制是否存在: ", tf.executing_eagerly())
train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
train_path = tf.keras.utils.get_file(os.path.basename(train_dataset_url),
                                     train_dataset_url)
# print(os.path.basename(train_dataset_url))
# 获取path的结尾也就是数据文件的名称

feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
# trains = pd.read_csv(train_path,  names=feature_names, header=0)
# print(trains.head())

batch_size = 32
train_datas = tf.data.experimental.make_csv_dataset(
    # 只要是csv同类型的数据均可以使用比如说xls, xlsx等excel格式的数据
    train_path,
    batch_size,
    column_names=feature_names,
    label_name=feature_names[-1],
    num_epochs=1
)
# 当前的train_datas:
# [........]'sepal_length', [..........]'sepal_width', [.........]'petal_length'..
def stack_features(feature, label):
    features = tf.stack(list(feature.values()), axis=1)
    return features, label
train_datas = train_datas.map(stack_features)
# 对比之前的train_datas可以看到现在的train_datas的特征数据更加整齐
# 现在的:[[5.  2.  3.5 1. ]
#        [6.9 3.2 5.2 2. ]
#        [5. 3.3 2.7 2.3]
#        [4.9 4.2 5.6 3.3]
#        [3.6 5.1 1.7 4. ]]

# 模型

# tf.keras.Sequential 模型是层的线性堆叠
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),
    # 激活函数可决定层中每个节点的输出形式
    # 在这里就是确保层以非线性的形式输出,否则就只是简单的一个层
    # 激活函数有很多种, 但隐藏层通常使用 ReLU
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(3)
])

example_datas, example_label = next(iter(train_datas))
# example_predict = model(example_datas)
# print(example_predict)
# # 获得是一些对数, 使用softmax将其转变为概率
# print(tf.nn.softmax(example_predict))
# # 使用argmax()可获得对应的索引,在此处的索引也就是三种鸾尾花的类型
# print(tf.argmax(example_predict, axis=1))

# 定义损失函数

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# 会接受模型的类别概率、预测结果和预期标签, 然后返回样本的平均损失
# from_logits : 返回值是否为对数张量也就是之前模型默认的预测结果

def loss_fc(model, datas, label):
    y_p = model(datas)
    return loss_object(y_true=label, y_pred=y_p)
# print(loss_fc(model, example_datas, example_label))

# 使用梯度函数以优化模型
def grad(model, datas, label):
    with tf.GradientTape() as t:
        loss_value = loss_fc(model, datas, label)
    return loss_value, t.gradient(loss_value, model.trainable_variables)

# 优化器
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

loss_value, grads = grad(model, example_datas, example_label)
# 此时的grads 是模型自带的 随机梯度下降 函数调整后的参数
print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                          loss_value.numpy()))
optimizer.apply_gradients(zip(grads, model.trainable_variables))
# 此时是经过优化器调整后的参数
print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(),
                                          loss_fc(model, example_datas, example_label).numpy()))

train_loss_results = []
train_accuracy_results = []
epochs = 201
# 需要明确的一点事结果的优良并非是随着训练次数的增加而更优
for epoch in range(epochs):
    loss_avg = tf.keras.metrics.Mean()
    loss_acuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    # model(datas) 返回的是一个形状为(X, 3)的列表 而label是一个numpy数组
    # 因此使用 SparseCategoricalAccuracy() 获取它的准确率
    for datas, label in train_datas:
        loss_value, grads = grad(model=model, datas=datas, label=label)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        loss_avg(loss_value)
        loss_acuracy(label, model(datas))

    train_loss_results.append(loss_avg.result())
    train_accuracy_results.append(loss_acuracy.result())
    if epoch % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    loss_avg.result(),
                                                                    loss_acuracy.result()))
'''
# plt.subplots(1, 2)
# plt.subplot(1, 2, 1)
# plt.plot(train_loss_results)
# plt.subplot(1, 2, 2)
# plt.plot(train_accuracy_results)
# plt.show()
'''

test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"
test_path = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                     origin=test_url)
test_datas_pd = pd.read_csv(test_path, names=feature_names, header=0)
test_datas = tf.data.experimental.make_csv_dataset(
    test_path,
    batch_size,
    column_names=feature_names,
    label_name='species',
    num_epochs=1,
    shuffle=False
)
test_datas = test_datas.map(stack_features)
test_predict = model.predict(test_datas)
print(loss_acuracy(test_datas_pd['species'], test_predict))
"""

# 使用tf.funcation()以提高性能
# 看不懂
"""
# 定义一个辅助函数来演示可能遇到的错误类型
def assert_raises(error_class):
  try:
    yield
  except error_class as e:
    print('Caught expected exception \n  {}:'.format(error_class))
    traceback.print_exc(limit=2)
  except Exception as e:
    raise e
  else:
    raise Exception('Expected {} to be raised but no error was raised!'.format(
        error_class))
'''
@tf.function()
# 将接下来的第一个函数变为TensorFlow类型
# 注意: 假如往一个普通函数内部传入TensorFlow的张量, 那么返回的结果也是TensorFlow
def add(a, b):
    return a+b
print(add(2, 2))
# 此时就可以更好的介入Gradient 梯度计算中

v = tf.Variable(1.0)
with tf.GradientTape() as t:
    r = add(v, 2)
print(r)
t.gradient(r, v)

def adds(a, b):
    return a+b
print(adds(2, 2))

# 可以在函数内部使用函数
@tf.function()
def dense_layer(x, w, b):
    return add(tf.matmul(x, w), b)
print(dense_layer(tf.ones([3, 2]), tf.ones([2, 2]), tf.ones([2])))


@tf.function
def f(x):
  print("Traced with", x)
  tf.print("Executed with", x)
f(tf.Variable(1))
f(1)
f(1)
# tf.print()会保留原有的值只要tf.numpy() = variable
# 那么无论是张量tensor还是python变量都可以得以重现
'''
@tf.function
def f(y):
    a = tf.Variable(1.0)
    a.assign_add(y)
    return a

# 正常情况下代码会按照预期的顺序执行, 但是有可能报出 "函数试图在非首次调用时创建变量"
# 放在图的上下文中就很容易出现问题, 因此需要其他的方式来实现功能
v = tf.Variable(3.0)
@tf.function
def fs(z):
    return v.assign_add(z)

# 或者我们能证明变量仅在第一次执行函数时创建
class P: pass
variable = P()
variable.v = None
@tf.function
def fd(z):
    if variable.v is None: # 注意是is不是==
        variable.v = tf.Variable(1.0)
    return variable.v.assign_add(z)
print(fd(1.0))
"""

# 分布式训练
# 在 TensorFlow 中分布式训练包括同步训练(其中训练步骤跨工作器和副本同步)、异步训练(训练步骤未严格同步)
# MultiWorkerMirroredStrategy 是同步多工作器训练的推荐策略

# keras的分布式训练
"""
# tf.distribute.Strategy API 提供了一个抽象的 API, 用于跨多个处理单元（processing units）分布式训练
# 它的目的是允许用户使用现有模型和训练代码,只需要很少的修改,就可以启用分布式训练

# tfds.disable_progress_bar()
# 禁用进度条

datas, info = tfds.load('mnist', with_info=True, # 获取元数据(描述数据的数据)
                        as_supervised=True)
mnist_trains, mnist_tests = datas['train'], datas['test']
print(mnist_tests)

strategy = tf.distribute.MirroredStrategy() # 用于分配策略并且提供一个上下文管理器
# print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

num_train_example = info.splits['train'].num_examples
num_test_example = info.splits['test'].num_examples

Buffer_size = 10000
Batch_size_per_replica = 64
Batch_size = Batch_size_per_replica * strategy.num_replicas_in_sync
print("Batch size: ", Batch_size)
# 像素点标准化
def scale(image, label):
    image = tf.cast(image, tf.float32)
    image = image/255
    return image, label

train_datas = mnist_trains.map(scale).cache().shuffle(Buffer_size).batch(Batch_size)
eval_datas = mnist_tests.map(scale).batch(Batch_size)

with strategy.scope(): # 在strategy的上下文管理器内建立模型
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(), # 将数据展开
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )

# 定义回调 callback
# TensorBoard: 此回调（callbacks）为 TensorBoard 写入日志，允许您可视化图形。
# Model Checkpoint: 此回调（callbacks）在每个 epoch 后保存模型。
# Learning Rate Scheduler: 使用此回调（callbacks），您可以安排学习率在每个 epoch/batch 之后更改

# 定义检查点
checkpornt_dir_path = 'C:\\Users\\tzl17\\Desktop\\Python&ML\\DeepLearning\\training_checkpoints'
# 将检查点阶段化
checkpornt_path_name = os.path.join(checkpornt_dir_path, 'ckpt_{epoch}')
# print(checkpornt_path_name)
# 定义衰减速率
def decay_learning_rate(epoch):
    if epoch < 3:
        return 1e-3
    elif epoch >= 3 and epoch < 7:
        return 1e-4
    else:
        return 1e-5

# 设置在每个\epoch\rs结束时打印当前速率
class PrintLrate(tf.keras.callbacks.Callback):
    def epoch_end(self, epoch, logs=None):
        print("epoch is {:d}, lerning rate is {:.5f}".format(
            epoch+1, model.optimizer.lr.numpy()
        ))

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='C:\\Users\\tzl17\\Desktop\\Python&ML\\DeepLearning\\logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpornt_dir_path,
                                       save_weights_only=True),
    tf.keras.callbacks.LearningRateScheduler(decay_learning_rate),
    PrintLrate()
]
# model.fit(train_datas, epochs=3, callbacks=callbacks)
# result = model.evaluate(eval_datas)
# print("1", result)

# 调用最后的检查点模型
model.load_weights(tf.train.latest_checkpoint(checkpornt_dir_path))
# reslut = model.evaluate(eval_datas)
# print(reslut)

# 保存模型
save_model_path = "save_model/"
# tf.keras.experimental.export_saved_model(model, save_model_path)
model_from_save_model_path = tf.keras.experimental.load_from_saved_model(save_model_path)
# 取得model后还需要重新设置compile
model_from_save_model_path.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)
result_loss, result_accuracy = model_from_save_model_path.evaluate(eval_datas)
print("result 2: ", result_loss, result_accuracy)

with strategy.scope():
  replicated_model = tf.keras.experimental.load_from_saved_model(save_model_path)
  replicated_model.compile(loss='sparse_categorical_crossentropy',
                           optimizer=tf.keras.optimizers.Adam(),
                           metrics=['accuracy'])

  eval_loss, eval_acc = replicated_model.evaluate(eval_datas)
  print ('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))"""

# 自定义分布式训练
"""
mnist_datas = tf.keras.datasets.fashion_mnist
(trains_mnist, trains_label), (tests_mnist, test_label) = mnist_datas.load_data()
trains_mnist = trains_mnist[:5000]
trains_label = trains_label[:5000]
tests_mnist = tests_mnist[:1000]
test_label = test_label[:1000]
# 添加维度 (28, 28) -> (28, 28, 1)
trains_mnist = trains_mnist[..., None]
tests_mnist = tests_mnist[..., None]

trains_mnist = trains_mnist / np.float32(255)
tests_mnist = tests_mnist / np.float32(255)

# 创建策略和图像分布函数

strategy = tf.distribute.MirroredStrategy()
# 策略的运作方式:
# 所有变量和模型图都复制在副本上。
# 输入都均匀分布在副本中。
# 每个副本在收到输入后计算输入的损失和梯度。
# 通过求和，每一个副本上的梯度都能同步。
# 同步后，每个副本上的复制的变量都可以同样更新。

print("num of strategy :", strategy.num_replicas_in_sync)

Buffer_size = len(trains_mnist)
Batch_size_per_replica = 100 # 这里的batch_size不能乱设置要确保数据总量能够整除它
Global_batch_size = Batch_size_per_replica * strategy.num_replicas_in_sync

# 数据合并
trains_mnist_datas = tf.data.Dataset.from_tensor_slices((trains_mnist, trains_label))\
    .shuffle(Buffer_size).batch(Global_batch_size)
tests_mnist_datas = tf.data.Dataset.from_tensor_slices((tests_mnist, test_label)).batch(Global_batch_size)

trains_strategy_mnist = strategy.experimental_distribute_dataset(trains_mnist_datas)
tests_strategy_mnist = strategy.experimental_distribute_dataset(tests_mnist_datas)

def creat_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(), # 2D空间数据的最大交叉操作
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

checkpoint_dir = './checkpoint_path'
checkpoint_dir_path = os.path.join(checkpoint_dir, 'ckpt')

# 损失函数
# 通常，在一台只有一个 GPU / CPU 的机器上, 损失需要除去输入批量中的示例数
# 在分布式训练中由于n个CPU / GPU的情况需要将BATCH_SIZE 除以n以分发给n个单量处理
# 此时损失就需要去除以总的BATCH_SIZE而不是和之前一样除以输入批量中的实例数(BATCH_SIZE / n)
# 需要这样做是因为在每个副本上计算梯度之后, 它们通过 summing 来使得在自身在各个副本之间同步
# 举个例子4个GPU BATCH_SZIE=64 分发后每个GPU 16个数据,假设此时损失为16那么每个GPU的损失值就是1
# 进行了summing后总的损失就变为了1*4=4 这样明显是不对的, 因此每个GPU除以64 就是 (1/4)*4
# 最后在求和可以得出损失值为1

with strategy.scope():
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction = tf.keras.losses.Reduction.NONE
    )
    def comput_loss(predict, label):
        per_loss_value = loss_object(label, predict)
        # 每个训练单量的损失值
        return tf.nn.compute_average_loss(per_loss_value, global_batch_size=Global_batch_size)
        # 次函数用于配合自定义分布式训练和分布策略中使用
        # 返回标量损失值

# 衡量指标(metric)用于跟踪损失和获取准确值
with strategy.scope():
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy'
    )
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='test_accuracy'
    )
# 模型和优化器
with strategy.scope():
    model = creat_model()
    optimizer = tf.keras.optimizers.Adam()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)


with strategy.scope():
    def train_step(inputs):
        datas, labels = inputs
        with tf.GradientTape() as t:
            predictions = model(datas, training=True)
            loss_value = loss_object(labels, predictions)
        print("model variables train: ", model.trainable_variables)
        gradients = t.gradient(loss_value, model.trainable_variables)
        # 释放Gradient的资源的同时调整model的参数
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_accuracy.update_state(labels, predictions)
        return loss_value
    def test_step(inputs):
        datas, labels = inputs
        t_predictions = model(datas, training=False)
        t_loss = loss_object(labels, t_predictions)

        test_loss.update_state(t_loss)
        test_accuracy.update_state(labels, t_predictions)

Epochs = 10
# 训练过程 第一种
'''
with strategy.scope():
    @tf.function
    def distribute_train(datas_inputs):
        per_loss_value = strategy.experimental_run_v2(fn=train_step, args=(datas_inputs,))
        return strategy.reduce(reduce_op=tf.distribute.ReduceOp.SUM, value=per_loss_value, axis=None)
        # strategy.experimental_run_v2 使用给定的参数args在fn上运行
        # strategy.reduce() reduce_op通常获取副本的计算方式tf.distribute.ReduceOp.xxx
        # value 则是传递给副本让其计算的值, 通常用来配合experimental_run_v2来返回一个张量值
        # tf.distribute.ReduceOp.SUM根据指示添加所有值 这里的意思就是将per_loss累加
    def distribute_test(datas_inputs):
        return strategy.experimental_run_v2(test_step, args=(datas_inputs,))

    for epoch in range(Epochs):
        total_loss = 0.0
        batch_num = 0
        # 训练数据循环
        for x in trains_strategy_mnist:
            total_loss += distribute_train(x)
            batch_num += 1
        train_loss = total_loss / batch_num
        # 测试数据循环
        for x in tests_strategy_mnist:
            distribute_test(x)
        if epoch % 2 == 0:
            checkpoint.save(checkpoint_dir_path)
        template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
                    "Test Accuracy: {}")
        print(template.format(epoch + 1, train_loss,
                              train_accuracy.result().mean() * 100, test_loss.result(),
                              test_accuracy.result().mean() * 100))
        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()
'''

# 调用最后一次检查点参数
# 需要重新设置compile
'''
model = creat_model()
eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='eval_accuracy'
)
eval_optimizer = tf.keras.optimizers.Adam()
eval_datas = tf.data.Dataset.from_tensor_slices((tests_mnist, test_label)).batch(100)
@tf.function
def eval_step(datas, labels):
    eval_predict = model(datas)
    return eval_accuracy(labels, eval_predict)
eval_checkpoint = tf.train.Checkpoint(optimizer=eval_optimizer, model=model)
eval_checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

for datas, labels in eval_datas:
    eval_step(datas, labels)
print ('Accuracy after restoring the saved model without strategy: {}'.format(
    eval_accuracy.result()*100))

# model_path = 'saved_model_S/'
# tf.keras.experimental.export_saved_model(model, model_path)
# model = tf.keras.experimental.load_from_saved_model(model_path)
'''

# 当处理一部分不需要遍历的数据集时(可以看做是对一个数据进行多次训练)
'''
with strategy.scope():
    @tf.function
    def distribute_train(datas_inputs):
        per_loss_value = strategy.experimental_run_v2(fn=train_step, args=(datas_inputs,))
        return strategy.reduce(reduce_op=tf.distribute.ReduceOp.SUM,
                               value=per_loss_value, axis=None)
    for _ in range(Epochs):
        total_loss = 0.0
        num_batches = 0
        train_iter = iter(trains_strategy_mnist)
        for _ in range(10):
            total_loss += distribute_train(next(train_iter))
            num_batches += 1
            print("total_loss: ", total_loss.numpy().mean())
            average_train_loss = total_loss / num_batches

            template = ("Epoch {}, Loss: {}, Accuracy: {}")
            print (template.format(10, average_train_loss.numpy().mean(), train_accuracy.result()*100))
            train_accuracy.reset_states()
'''

# 在tf.funcation()中迭代训练 第二种
# 个人感觉和第一种训练过程一样只是把内容更多的包含在了函数中
"""

# keras进行多工作器训练
"""
tfds.disable_progress_bar()

datas, info = tfds.load(name='mnist', with_info=True, as_supervised=True)

def scale(image, label):
    image = tf.cast(image, tf.float32)
    image = image / 255
    return image, label

Batch_size = 64
Buffer_size = 60000
mnist_datas = datas['train'].map(scale).cache().shuffle(Buffer_size)

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(), # 二维空间数据最大交叉
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=['accuracy']
    )
    return model
mnist_datas_0 = mnist_datas.batch(Batch_size)
first_single_model = create_model()
# first_single_model.fit(mnist_datas_0, epochs=3)

# 多工作器配置
# TF_CONFIG 环境变量来训练多台机器, 每台机器可能具有不同的角色
# TF_CONFIG 有两个组件: cluster 和 task
# cluster 提供有关训练集群的信息(集群的工作器和参数服务器) task 提供当前任务的信息
# 保存检查点和为 TensorBoard 编写摘要文件的工作期被称为主工作器: index 0
# 其他的工作器也许需要配置相同环境变量特指相同的cluster字典 type和index取决于它本身的角色
# 要基于全局数据批量大小来调整学习速率

# TF_CONFIG的模样大致如下
# os.environ['TF_CONFIG'] = json.dumps({
#     'cluster': {
#         'worker': ["localhost:12345", "localhost:23456"]
#     },
#     'task': {'type': 'worker', 'index': 0}
# })

# MultiWorkerMirroredStrategy 是同步多工作器训练的推荐策略
# MultiWorkerMirroredStrategy 在所有工作器的每台设备上创建模型层中所有变量的副本
# 它使用 CollectiveOps, 一个用于集体通信的 TensorFlow 操作, 来聚合梯度并使变量保持同步

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
# 创建strategy之前要先设置TF_CONFIG
# strategy = tf.distribute.MirroredStrategy() 这是之前的分布式训练的分配策略API
# print("nume of devices", strategy.num_replicas_in_syc) 之前查看GPU数目的代码段

Num_workers = 2
Global_batch_size = Batch_size * Num_workers
mnist_datas_1 = mnist_datas.batch(Global_batch_size)
# 由于没有设置 TF_CONFIG 因此这实际上是单机训练 尽管可以运行
with strategy.scope():
    model_1 = create_model()
# model_1.fit(mnist_datas_1, epochs=3)

# 数据集分片和batch大小
# tf.distirbute.stragtegy 可以进行自动 数据分片
# 如果不愿意自动分片可以使用tf.data.experimental.DistributeOptions
# 以下代码实现上述的取消自动分片
options = tf.data.Options()
options.experimental_distribute.auto_shard = False
mnist_datas_1_no_auto_shard = mnist_datas_1.with_options(options)
# no_auto_shard 取消自动分片


# 容错能力(添加检查点, 以便于工作器出错导致整体失败后可以回调)
# 因此可将检查点机制放在callbacks中
checkpoint_path = 'C:\\Users\\tzl17\\Desktop\\Python&ML\\DeepLearning\\tmp\\keras-ckpt'
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path)
    # tf.keras.callbacks.TensorBoard(),
    # tf.keras.callbacks.EarlyStopping()
]
# 同时callbacks 可以放入earlystop, tensorbord等便于检查与训练的api

mnist_datas_2 = mnist_datas.batch(Batch_size)
with strategy.scope():
    model_2 = create_model()
model_2.fit(mnist_datas_2, epochs=3, callbacks=callbacks)
"""

# tf.estimator进行多工作器训练
"""
tfds.disable_progress_bar()

Buffer_size = 10000
Batch_size = 64

def input_fn(input_context=None):
    datas, info = tfds.load(name='mnist', as_supervised=True, with_info=True)

    mnist_datas = (datas['train'] if tf.estimator.ModeKeys.TRAIN else datas['test'])

    # tf.estimator.ModeKeys是Estimator模型模式的标准名称
    # TRAIN：训练/拟合(fit)模式
    # EVAL：测试/评估(evaluate)模式
    # PREDICT：谓词/推断(predict)模式
    def scale(images, label):
        images = tf.cast(images, tf.float32)
        images = images / 255
        return images, label
    print("mnist_datas: ", mnist_datas)
    print("mnist_datas shard ", mnist_datas.shard(input_context.num_input_pipelines,
                                        input_context.input_pipeline_id))
    if input_context:
        mnist_datas = mnist_datas.shard(input_context.num_input_pipelines,
                                        input_context.input_pipeline_id)
        # shard 根据GPU的数量、编号进行数据的分片
    return mnist_datas.map(scale).shuffle(Buffer_size).batch(Batch_size)

# 正常来说此时需要先配置TF_COFING 然后配置strategy
'''
os.environ['TF_CONFIG'] = json.dump({
    'cluster': {
        'worker': ['localhost: ' , 'localhost: ']
    },
    'task': {
        'type': 'worker', 'index': tf.int
    }
})
'''

Learning_rate = 1e-4
def model_fn(features, labels, mode):
    # 这里的features就是数据Images 
    print("model_fn :", mode)
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    logits = model(features, training=False)
    # 这里的logits就是预测结果
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'logits': logits}
        return tf.estimator.EstimatorSpec(labels=labels, predictions=predictions)
    # tf.estimator.EstimatorSpec根据mode也就是(tf.estimator.ModeKeys.TRAIN/EVAL/PREDICTIONS)
    # 用于确认这是训练还是测试评估还是预测

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(
        learning_rate=Learning_rate
    )
    # tf.compat.v1.train用于设置训练步骤
    # 定义训练步骤(优化步骤)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )(labels, logits)
    loss = tf.reduce_sum(loss) * (1.0 / Batch_size)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)
    return tf.estimator.EstimatorSpec(
        # 这个过程是estimator的训练过程
        # 因为有optimizer.minimize这个就是自动训练梯度下降
        mode=mode,
        loss=loss,
        train_op=optimizer.minimize(
            loss, tf.compat.v1.train.get_or_create_global_step()
        )
        # train_op用于确定训练步骤 GradientDescentOptimizer(learning_rate).minimize(loss)
        # 自动梯度下降需要给出Loss和Learning_rate
        # minimize()由compute_gradients() 和apply_gradients()组合在一起
        # compute_gradients(): 可以得到一个loss梯度变化的列表
        # apply_gradients(): 将loss梯度变化的值应用
    )

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
# tf.distribute.experimental: 分布式训练分配策略库
# tf.distribute: 分布式训练设备运行计算库(框架)

config = tf.estimator.RunConfig(train_distribute=strategy)
# 使用RunConfig将strategy应用
classifier = tf.estimator.Estimator(
    model_fn=model_fn, model_dir=deep_path +'tmp\\multiworker', config=config
)
result = tf.estimator.train_and_evaluate(
    classifier,
    train_spec=tf.estimator.TrainSpec(input_fn=input_fn), # 训练
    eval_spec=tf.estimator.EvalSpec(input_fn=input_fn) # 评估
)
# 预测的话就直接使用classifier.predict()
'''
当我去掉input_fn内的参数mode并将
mnist_dataset = (datas['train'] if mode == tf.estimator.ModeKeys.TRAIN else
                datas['test'])
改为
mnist_dataset = (datas['train'] tf.estimator.ModeKeys.TRAIN else
                datas['test'])
时，发现依然可以使用因此可以得知当我们使用tf.estimator.train_and_evaluate他会自动
给对象函数(例如输入函数input, 模型函数)传入一个tf.estimator.ModeKeys.TRAIN\EVAL\PREDICT
以解决我们需要针对fit\evaluate\predict所额外做的工作
'''
print(result)
"""

# 保存和加载模型
# (保存和加载的方式之前多有使用本次主要是知道如何将中途断掉的模型继续放入训练过程继续训练)
"""
# 先简单的构建一部分数据和创建一个模型
strategy = tf.distribute.MirroredStrategy()

def get_data():
    datas, info = tfds.load('mnist', as_supervised=True, with_info=True)
    trains, tests = datas['train'], datas['test']
    def scale(images, labels):
        images = tf.cast(images, tf.float32)
        images = images / 255.0
        return images, labels
    Buffer_size = 10000
    Batch_size_per = 64
    Batch_size = Batch_size_per * strategy.num_replicas_in_sync

    trains_mnist = trains.map(scale).cache().shuffle(Buffer_size).batch(Batch_size)
    tests_mnist = tests.map(scale).batch(Batch_size)
    return trains_mnist, tests_mnist

def get_model():
    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10)
        ])

        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])
        return model
# model = get_model()
train_dataset, eval_dataset = get_data()
# model.fit(train_dataset, epochs=2)

model_path = deep_path + 'tmp_2\\keras_model'
# model.save(model_path)

# 获取模型的各种方法

# 第一类 直接使用keras
'''
# 第一种
model_2 = tf.keras.models.load_model(model_path)
model_2.fit(train_dataset, epochs=1)

# 第二种在分布式训练中需要在分配策略内调用它
strategy = tf.distribute.OneDeviceStrategy('/cpu:0')
with strategy.scope():
    model_3 = tf.keras.models.load_model(model_path)
    model_3.fit(train_dataset, epochs=1)
'''

# 第二类 使用较低leval的API (也意味着更广泛的应用范围)
model_path_2 = deep_path + 'tmp_3\\keras_model'
predict_dataset = eval_dataset.map(lambda image, label: image)
'''
# 第一种

# model_4 = get_model()
# model_4.fit(train_dataset, epochs=2)
# print("model_4 :", model_4)
# model_4 : <tensorflow.python.keras.engine.sequential.Sequential object at 0x0000021F66B95E08>
# 可以看到以Sequential结尾
# tf.saved_model.save(model_4, model_path_2)

model_4_1 = tf.saved_model.load(model_path_2)
Defalut_save_model = 'serving_default' # keras模型的保存键

inference_func = model_4_1.signatures[Defalut_save_model] # 相当于获取了keras模型
print("inference_func :", inference_func)
for batch in predict_dataset.take(1):
    print(inference_func(batch))
# 注意此时返回的是一个对象 加载的对象可能包含多个功能, 每个功能都与一个键关联

#print("model_4_1 :", model_4_1)
# model_4_1 : <tensorflow.python.saved_model.load.Loader._recreate_base_user_object.<locals>._UserObject object ..>
# 可以看到以userobject 结尾 即可以利用的对象

# 第二种 以分布式的方式在分布策略内处理对象
# 产生无法找到资源错误
strategy_2 = tf.distribute.MirroredStrategy()
with strategy_2.scope():
    Defalut_save_model = 'serving_default'  # keras模型的保存键
    predict_dataset = eval_dataset.map(lambda image, label: image)
    model_4_2 = tf.saved_model.load(model_path_2)
    # 获得对象 加载的对象可能包含多个功能, 每个功能都与一个键关联
    inference_func_2 = model_4_2.signatures[Defalut_save_model]
    for batch in predict_dataset.take(1):
        print(inference_func_2(batch))
        # 在之前 experimental_run_v2 使用给定的参数args在fn上运行
        # 在此时 可以知晓experimental_run_v2不仅是可以用函数对象和参数之间的传递
        # 也可以看作数据和模型之间的传递
'''

# 通过将加载的对象放入keras层上继续训练
# 可以确保即使训练途中因为意外导致训练失败, 可以在纠正错误后继续上次的训练
# 使用TensorFlow_hub
# 可以知道对象是可以放入模型的层内
def build_model(model_object):
    x = tf.keras.layers.Input(shape=(28, 28, 1), name='input_x')
    keras_layers = tfub.KerasLayer(model_object, trainable=True)(x)
    model = tf.keras.Model(x, keras_layers) # 注意这个x
    return model

model_object = tf.saved_model.load(model_path)
# 这里证明了无论是keras保存的模型还是save_model保存的模型 过程和结果是一致的
# 都可以被两者获取, 因此两者是可以混合使用的
new_model = build_model(model_object)
new_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
new_model.fit(train_dataset, epochs=2)

# 在strategy.scope()下还是无法使用save_model
"""

# 图像
"""
# 训练卷积神经网络

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# 标签含义
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
print(train_labels[:10])
# for i in range(20):
#     plt.subplot(4, 5, i+1)
#     plt.imshow(train_images[i])
#     plt.xlabel(class_names[train_labels[i][0]])
# plt.show()
def get_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                                     # filters: 卷积层中过滤器数量 也是处理后图像的通道数
                                     # 比如一个图像初始(x, x, x, 3)意思是该图像的通道数是3(RGB) 经过卷积后我们要求它的图像通道数是32
                                     # 相当于我们用初始的三种颜色R, G, B混合处理获得了32种颜色也就是是说我们的特征从初始的三个特征变成了32个特征
                                     # kernel_size: 卷积层的高度和宽度通常是两个数字
                                     # 一个数字则意味着高度和宽度一致 kernel_size=(3, 3)和kernel_size=3一致
                                     activation='relu',
                                     input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(10))
    return model


# train_datas = tf.data.Dataset.from_tensor_slices((train_images[:5000], train_labels[:5000])).shuffle(5000).batch(1000)
# test_datas = tf.data.Dataset.from_tensor_slices((test_images[:500], test_labels[:500]))
# 对数据进行这样的处理只是为了方便对数据统一进行清洗不能够直接带入到模型内(假如模型的第一层是CONV2D)
# 因为CONV2D作为第一层需要input_shape要求是一个列表 而进行过from_tensor_slics的数据的形状是一个tupe元组((x, x, x), (x,))
# 可以进行这样训练(注意要有BATCH):
# for x, y in train_datas:
#     history = model.fit(x, y, epochs=3)
#     print(pd.DataFrame(history.history))
model = get_model()
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)
# history = model.fit(train_images[:10000], train_labels[:10000], epochs=3,
#                     validation_data=(test_images[:2000], test_labels[:2000]))


"""

# 图像分类
"""
URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

path = tf.keras.utils.get_file('cats_and_dogs_filtered.zip', origin=URL, extract=True)
# extract 尝试将文件提取文存档文件
path = os.path.join(os.path.dirname(path), 'cats_and_dogs_filtered')
# 组成了数据文件存放的路径
# os.path.dirname(path)获取path的头文件前缀 os.path.basename(path)获取头文件名称

train_datas_path = os.path.join(path, 'train')
validation_datas_path = os.path.join(path, 'validation')

train_dogs_images_path = os.path.join(train_datas_path, 'dogs')
train_cats_images_path = os.path.join(train_datas_path, 'cats')

validations_dogs_images_path = os.path.join(validation_datas_path, 'dogs')
validations_cats_images_path = os.path.join(validation_datas_path, 'cats')

train_dogs_nums = len(os.listdir(train_dogs_images_path))
train_cats_nums = len(os.listdir(train_cats_images_path))
validation_dogs_nums = len(os.listdir(validations_dogs_images_path))
validation_cats_nums = len(os.listdir(validations_cats_images_path))
train_images_nums = train_dogs_nums + train_cats_nums
validation_images_nums = validation_dogs_nums + validation_cats_nums
print("train datas, dog images nums: {}, cat images nums: {}".format(train_dogs_nums,
      train_cats_nums))
print("validation datas, dog images nums: {}, cat images nums: {}".format(validation_dogs_nums,
      validation_cats_nums))

Batch_size = 128
epochs = 15
Images_height = 150
Images_width = 150

# 数据处理
# 1: 需要从磁盘读取所有数据 2: 需要将数据从(0, 255) -> (0, 1)
# 3: 解码图像内容将其输入到网络内 4: 转换为浮点张量

# 使用ImageDataGenerator可以方便的读取图片数据并将这些图像转换成张量生成器
train_process_images = tf.keras.preprocessing.image.ImageDataGenerator(1 / 255.0)
validation_process_images = tf.keras.preprocessing.image.ImageDataGenerator(1 / 255.0)

train_images = train_process_images.flow_from_directory(batch_size=Batch_size,
                                                        directory=train_datas_path,
                                                        shuffle=True,
                                                        target_size=(Images_height, Images_width),
                                                        class_mode='binary')
# flow_from_directory 可以读取、缩放、调整数据
validation_images = validation_process_images.flow_from_directory(batch_size=Batch_size,
                                                                  directory=validation_datas_path,
                                                                  target_size=(Images_height, Images_width),
                                                                  class_mode='binary')
'''
train_dogs_images = os.listdir(train_dogs_images_path)
for i in range(10):
    imagepath = train_dogs_images_path + "\\" + train_dogs_images[i]
    images = tf.io.read_file(imagepath)
    images = tf.image.decode_jpeg(images)
    images = tf.image.resize_with_crop_or_pad(images, 350, 350)
    plt.subplot(2, 5, i+1)
    plt.imshow(images)
plt.show()
'''

def get_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(Images_height, Images_width, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        # from_logits True可以确保数值(将激活函数应用在了损失函数中不需要手动添加)更稳定
        # 默认情况下数值为[0, 1] 现在则可以有更多的可能与激活函数的功能相似
        metrics=['accuracy']
    )
    return model
model_1 = get_model()
history_1 = model_1.fit_generator(
    train_images,
    steps_per_epoch=train_images_nums // Batch_size,
    epochs=epochs,
    validation_data=validation_images,
    validation_steps=validation_images_nums // Batch_size
)

def plot_result(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss=history.history['loss']
    val_loss=history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

plot_result(history_1)

# 从图中可以看到训练数据的结果明显要比validation数据的结果好很多
# 明显过拟合 此时需要更多的数据, 或者减小权重/丢失部分参数以降低噪音的影响(正则化 l1, l2)
# 或者提升模型的精度

# 使用tf.keras.processing.image.ImageDataGenerator类内的一部分参数
# horizontal_flip: 水平直线旋转 vertical_flip: 垂直随机旋转
# rescale: 缩放 rotation_range=int(): 随机旋转的度数范围
# zoom_range=float() or list[up, down]: 缩放的范围
# 演示horizontal_flip

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# raise_images_test = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.0,
#                                                                     horizontal_flip=True)
# train_images_2 = raise_images_test.flow_from_directory(
#     batch_size=Batch_size,
#     directory=train_datas_path,
#     shuffle=True,
#     target_size=(Images_height, Images_width)
# )
# example_images = [train_images_2[0][0][0] for i in range(5)]
# 重复的处理第一张图片5次可以看到细微的差别

raise_images = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=45, # 随机旋转度数范围
    width_shift_range=.15, # 宽度偏移
    height_shift_range=.15, # 高度偏移
    horizontal_flip=True, # 水平旋转
    zoom_range=0.5 # 缩放比例
)

train_images_new = raise_images.flow_from_directory(
    batch_size=Batch_size,
    directory=train_datas_path,
    target_size=(Images_height, Images_width),
    shuffle=True,
    class_mode='binary'
)
example_images = [train_images_new[0][0][0] for i in range(5)]

model_2 = get_model()
history_2 = model_2.fit_generator(
    train_images_new,
    steps_per_epoch=train_images_nums // Batch_size,
    epochs=epochs,
    validation_data=validation_images,
    validation_steps=validation_images_nums // Batch_size
)
plot_result(history_2)

# 通过处理权重来避免过拟合
def get_new_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu',
                               input_shape=(Images_height, Images_width, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),  # 设置为小数时它将随机丢弃小数个百分比的参数 设置为0时将会在训练过程将部分参数变为0(相当于降低影响)
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['accuracy']
    )
    return model
new_model = get_new_model()
new_history = new_model.fit_generator(
    train_images,
    steps_per_epoch=train_images_nums // Batch_size,
    epochs=epochs,
    validation_data=validation_images,
    validation_steps=validation_images_nums // Batch_size
)
plot_result(new_history)
"""

# 图像分类 使用tf.hub -迁移学习
"""
classifier_url ="https://hub.tensorflow.google.cn/google/tf2-preview/mobilenet_v2/classification/2" #@param {type:"string"}

Image_shape = (224, 224)
classifier_model = tf.keras.Sequential([
    tfub.KerasLayer(classifier_url, input_shape=Image_shape+(3, ))
    # 注意调整层的输入形状
])

example_image = tf.keras.utils.get_file('image.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
import PIL.Image as Image
example_image = Image.open(example_image).resize(Image_shape)
example_image = np.array(example_image) / 255.0
example_image = example_image[np.newaxis, ...]
# 添加新的一列 (224, 224, 3) -> (1, 224, 224, 3)

result = classifier_model.predict(example_image)
print(result.shape)
# 预测结果的长度为1001 说明原本模型对应的输出样本种类是1001种

# print(np.argmax(result[0]))
# 通过使用np.argmax() 获取对应的标签位置, 可以得知该样本的预测结果是什么种类

# 先获取预设的标签文件
classifier_labels = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
classifier_labels = np.array(open(classifier_labels).read().splitlines())

# 获取预测结果的种类
print(classifier_labels[np.argmax(result[0])])
# military uniform: 军装

# 更全面的测试
datas_path = tf.keras.utils.get_file('flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                     untar=True)
process_images = tf.keras.preprocessing.image.ImageDataGenerator(1/255.0)
image_datas = process_images.flow_from_directory(
    directory=datas_path,
    target_size=Image_shape
)
print("图片数量: ", image_datas.samples, "Batch: ", image_datas.batch_size)
for images, labels in image_datas:
    print(images.shape)
    print(labels.shape)
    break
    # 可以得知 Batch_size = 32, 样本种类为5
labels_names = ['daisy', 'dandelion', 'roise', 'sunflowers', 'tulips']
images_batch, labels_batch = next(image_datas)
# 取一batch数据
results = classifier_model.predict(images_batch)
results_predict_labels = np.argmax(results, axis=-1)
results_predict_labels = [classifier_labels[i] for i in results_predict_labels]
'''
for i in range(30):
    plt.subplot(5, 6, i+1)
    plt.imshow(images_batch[i])
    plt.title(results_predict_labels[i])
    plt.axis('off')
_ = plt.suptitle("ImageNet predictions")
plt.show()
'''

# hub 还有不包含顶层分类的模型
feature_extractor_url = "https://hub.tensorflow.google.cn/google/tf2-preview/mobilenet_v2/feature_vector/2" #@param {type:"string"}
feature_extractor_layer = tfub.KerasLayer(feature_extractor_url,
                                          input_shape=(224, 224, 3))

# print(feature_extractor_layer(images_batch).shape)
# (32, 1280) 可以看到默认输出结果为1280个类别

feature_extractor_layer.trainable=False
# 相当于把原有模型的输出DENSE层取消

# 现在根据实际情况为其安装最后一层输出层
classifier_model_new = tf.keras.Sequential([
    feature_extractor_layer,
    tf.keras.layers.Dense(image_datas.num_classes)
])

# new_results = classifier_model_new.predict(images_batch)
# print(new_results.shape)
# 可以看到最终结果从(32, 1280) -> (32, 5)
# 重新训练以使其的输出类型为现有数据的类型

# 编译
classifier_model_new.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['acc']
)
print(classifier_model_new.summary())
# 收集培训进度
class CollectbatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []
    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])
        self.model.reset_metrics()

steps_epochs = np.ceil(image_datas.samples / image_datas.batch_size)
print(steps_epochs)
# 向上取整: np.ceil  向下取整: np.floor
batch_stats_callback = CollectbatchStats()
history = classifier_model_new.fit_generator(image_datas, epochs=2,
                                             steps_per_epoch=steps_epochs,
                                             callbacks = [batch_stats_callback])
model_dir = deep_path + 'Tfhub_tmp\\tfhub_model'
classifier_model_new.save(model_dir)
classifier_model_new = tf.keras.models.load_model(model_dir)
# 通过image_datas 获取lable名称
image_labels = sorted(image_datas.class_indices.items(), key=lambda pair: pair[1])
print(image_labels)
image_labels = np.array([key.title() for key, value in image_labels])
print(dir(image_datas))

result = classifier_model_new.predict(images_batch)
result_label_id = np.argmax(result, axis=-1)
result_label_names = image_labels[result_label_id]
# 预测结果

labels_batch_id = np.argmax(labels_batch, axis=-1)
labels_batch_names = image_labels[labels_batch_id]
# 原始结果
print("原始结果: ", labels_batch_names,
      "预测结果: ", result_label_names)
"""

# 图像分类 使用ConvNet -迁移学习
"""
URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

path = tf.keras.utils.get_file('cats_and_dogs_filtered.zip', origin=URL, extract=True)
# extract 尝试将文件提取文存档文件
path = os.path.join(os.path.dirname(path), 'cats_and_dogs_filtered')
# 组成了数据文件存放的路径
# os.path.dirname(path)获取path的头文件前缀 os.path.basename(path)获取头文件名称

train_datas_path = os.path.join(path, 'train')
validation_datas_path = os.path.join(path, 'validation')

process_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(1 / 255.0 - 1)

Image_height = 160
Image_width = 160
Batch_size = 32
train_images = process_image_generator.flow_from_directory(
    directory=train_datas_path,
    target_size=(Image_height, Image_width),
    batch_size=Batch_size,
    shuffle=True,
    class_mode='binary'
)
validation_images = process_image_generator.flow_from_directory(
    directory=validation_datas_path,
    target_size=(Image_height, Image_width),
    batch_size=Batch_size,
    class_mode='binary'
)

train_example, train_example_labels = next(train_images)
Image_shape = (Image_height, Image_width, 3)

base_model_layers = tf.keras.applications.MobileNetV2(input_shape=Image_shape,
                                                      include_top=False,
                                                      weights='imagenet')

train_example_results_1 = base_model_layers(train_example)
# 该层将模型从(160, 160, 3) -> (5, 5, 1280)
print("1", train_example_results_1.shape)

# 特征处理

# 在重新训练和编译模型之前 冻结模型的基础可以
# 防止重新训练期间模型的权重发生变化
base_model_layers.trainable = False
# print(base_model_layers.summary())
# 添加分类头
global_average_layers = tf.keras.layers.GlobalAveragePooling2D()
# 使特征平铺
train_example_results_2 = global_average_layers(train_example_results_1)
print("2", train_example_results_2.shape)

# 添加输出层
predition_layers = tf.keras.layers.Dense(1)
train_example_results_3 = predition_layers(train_example_results_2)
print("3", train_example_results_3.shape)

# 将三个层组合成一个模型
model = tf.keras.Sequential([
    base_model_layers,
    global_average_layers,
    predition_layers
])
# 编译模型
base_learning_rate = 0.0001
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
    metrics=['accuracy']
)
print(model.summary())
print("冻结base_model后的模型待训练参数", len(model.trainable_variables))

validation_example, validation_example_labels = next(validation_images)
# print(train_example)
initial_epochs = 10
validation_steps=20
# history = model.fit_generator(
#     train_images,
#     epochs=initial_epochs,
#     validation_data=validation_images
# )
# print(history)

# 在此过程中并未更新M_V2模型的参数 为了使模型更贴合训练数据本身
# 需要调整预训练模型(M_V2模型)的参数
# 在大多数卷积网络中(CNN)前几层的参数具有通用型, 层数越靠近高层
# 参数的专业性(针对数据本身的特征)也就越强 因此微调就是调整高层参数使其
# 对现有数据的特性更符合

# 首先解封基础模型的参数
base_model_layers.trainable = True
print("解封base_model后的待训练参数: ", len(base_model_layers.layers))

# 可以选择性的冻结部分参数
frozen_variables_nums = 100
# 因为之前提到过的初始层数参数具有通用性, 因此冻结前一部分层数只重新训练顶层的参数

for layers_variable in base_model_layers.layers[:frozen_variables_nums]:
    layers_variable.trainable = False

new_model = tf.keras.Sequential([
    base_model_layers,
    global_average_layers,
    predition_layers
])

new_model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
    metrics=['accuracy']
)
print("冻结部分base_model参数后的待训练参数: ", len(new_model.trainable_variables))
print(new_model.summary())

# new_history = model.fit_generator(
#     train_images,
#     epochs=initial_epochs,
#     initial_epoch=history.epoch[-1],
#     validation_data=validation_images
# )
"""

# 图像分割 (一个物体在一张图像中的位置、这个物体的形状、以及哪个像素属于哪个物体)
"""
from IPython.display import clear_output
tfds.disable_progress_bar()

dataset, info = tfds.load('oxford_iiit_pet:3.0.0', with_info=True)

def normalize_datas(image, image_mask):
    image = tf.cast(image, tf.float32) / 255.0
    image_mask = image_mask - 1
    return image, image_mask

def load_image_train(datas):
    image = tf.image.resize(datas['image'], (128, 128))
    image_mask = tf.image.resize(datas['segmentation_mask'], (128, 128))

    if tf.random.uniform(()) > 0.5:
        # 随机抽取部分图片水平翻转
        image = tf.image.flip_left_right(image)
        image_mask = tf.image.flip_left_right(image_mask)

    image, image_mask = normalize_datas(image, image_mask)
    return image, image_mask

def load_image_test(datas):
    image = tf.image.resize(datas['image'], (128, 128))
    image_mask = tf.image.resize(datas['segmentation_mask'], (128, 128))

    image, image_mask = normalize_datas(image, image_mask)
    return image, image_mask

Train_length = info.splits['train'].num_examples
Batch_size = 64
Buffer_size = 1000
Steps_per_epochs = np.ceil(Train_length // Batch_size)

train_datas = dataset['train'].map(load_image_train,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
# num_parallel_calls: 并行个数 tf.data.experimental.AUTOTUNE自行选择最优并行个数
test_datas = dataset['test'].map(load_image_test)

train_images = train_datas.cache().shuffle(Buffer_size).batch(Batch_size).repeat()
train_images = train_images.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
# prefetch函数 有一个后台线程和一个内部缓存区在数据被请求前就从dataset 中预加载一些数据处理
# 数据产生过程和数据消耗过程发生重合就可以使用它来提高性能
test_images = test_datas.batch(Batch_size)


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()
'''
for image, image_mask in train_datas.take(1):
    display([image, image_mask])
'''

# 定义模型 迁移模型: U-NET

# U-Net 由一个编码器(下采样器(downsampler))和一个解码器(上采样器(upsampler))组成
# 编码器在模型的训练过程中不会被训练


output_channels = 3
# 输出信道为3 是因为对每个元素分配类别(像素属于主要物品,像素属于主要物品的轮廓, 像素属于其他)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False
)

layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]
# model.get_layer() 获取层对象

# 创建特征提取模型层
# 它将输入图像特征从(128, 128)->(4, 4)方便网络进行拟合和训练
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable=False
# 冻结NET模型参数可以提高模型
print(down_stack.summary())

from tensorflow_examples.models.pix2pix import pix2pix

# 定义解码器
# 简单来说就是把缩小的图形放大
# 在layer_names中我们吧图像变小方便网络训练现在得出结果后我们要按layer_names的顺序将图形放大
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels):
    last_model = tf.keras.layers.Conv2DTranspose(
        # 转置的卷积层-反卷积 参数与Conv2D的参数含义一样
        filters=output_channels,
        kernel_size=3, strides=2,
        padding='same', activation='softmax'
    )

    inputs = tf.keras.layers.Input(
        shape=[128, 128, 3]
    )
    x = inputs
    # 降频取样 将模型缩小
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])
    # reversed() 翻转处理
    # 'abcd' -> reversed('abcd'): 'dcba'

    # 升频 将处理后的图像还原
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    x = last_model(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

model = unet_model(output_channels)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

def predict_mask_channel(pred_mask):
    pred_mask = np.argmax(pred_mask, axis=-1)
    # pred_mask 是一个包含每个元素点预测类别的tensor 因为channels=3所以说先获取
    # 该元素点属于哪个类别将其单独提取
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predict(datas, num):
    for image, mask in datas.take(num):
        pred_mask = model.predict(image)
        display([image[0], mask[0], predict_mask_channel(pred_mask)])

# 定义callbacks 观察预测结果的准确性
class Callbacks(tf.keras.callbacks.Callback):
    def on_epoch_results(self, epoch, logs=None):
        clear_output(wait=True)
        show_predict(datas=None, nums=None)
        print('\nSample Prediction after epoch {}\n'.format(epoch + 1))

EPOCHS = 20
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

model_history = model.fit(train_datas, epochs=EPOCHS,
                          steps_per_epoch=Steps_per_epochs,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_datas,
                          callbacks=[Callbacks()])
"""

# 文本 单词嵌入
"""
# 处理一个句子有两种常用的方法
# 第一种: 我们将创建一个零矢量，其长度等于词汇量，然后在与该单词对应的索引中放置一个
# 缺点: 是输入到模型内时数据时稀疏的(包含大量的0)
# 第二种: 为每个单词提供一个数字代号
# 缺点:
# 1 数字代号是任意设定的因此数字代号之前没有相关性
# 2 模型解释性差, 特征之间的组合也因为数字之间没有相关性从而显得没有意义

# 单词嵌入 高效密集的表示方法

# 嵌入层使该方法更加的容易实现
# 嵌入层可以理解为一个查找表，它从整数索引(代表特定单词)映射到密集向量(其嵌入)
# 比如说输入一个整数然后他会按照映射表的映射值返还
# 嵌入的维数(或宽度)是您可以进行实验的参数

embedding_layer = tf.keras.layers.Embedding(input_dim=1000,
                                            # 词汇表的大小, 这个要看你提供的词汇表的具体大小不能乱填
                                            output_dim=5
                                            # 密集向量的长度, 在本例中嵌入密集向量的值是一个长度为5的列表
                                            )
# 创建嵌入层时嵌入层的权重会随机初始化 之后将随着模型的训练过程(反向传播方法)而发生改变
test_result = embedding_layer(tf.constant([1, 2, 3]))
# print(test_result)
# [[X, X, X, X, X], [Y, Y, Y, ..], [Z, Z, ...]]

test_result = embedding_layer(tf.constant([[0, 0, 0], [3, 4, 5]]))
# print(test_result)
# print(test_result.shape)
# [[[A, A, A, A, A], [A, ....], [A, ....]], 
# [[B, ....], [C, ....], [D, ....]]]


# 简单的实例-电影评论情感分析
(train_data, test_data), info = tfds.load(
    # imdb电影评论数
    'imdb_reviews/subwords8k',
    split=(tfds.Split.TRAIN, tfds.Split.TEST),
    as_supervised=True, with_info=True
)

encoder =info.features['text'].encoder
print(encoder.subwords[:10])
print(dir(encoder))
# 查看encoder(对象object)的属性时 可以看到里面包含['load_from_file', 'save_to_file']
# 因此可以确定encoder对象是一个继承自tfds.features.text.SubwordTextEncoder的编码器
for i, j in train_data.take(2):
    print(encoder.decode(i))
# 将train_datas里面数字代表的单词重新转成单词

# '_' 代表空格

train_batches = train_data.shuffle(1000).padded_batch(batch_size=32,
                                                      padded_shapes=((None,), ()),
                                                      )

test_batches = test_data.shuffle(1000).padded_batch(batch_size=32,
                                                    padded_shapes=((None,), ()))
# .padded_batch() 将电影评论的长度变成一致 注意padded_shapes的值要和输入数据的长度一致

train_batch_example, train_batch_example_labels = next(iter(train_batches))
# iter() 返回迭代器本身
# print(train_batch_example.numpy().shape)
# print(train_batch_example_labels.numpy())

# 创建模型
embedding_dim = 16
print(encoder.vocab_size)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, embedding_dim), # 将输入数据转化为嵌入向量
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)
history = model.fit(
    train_batches,
    epochs=10,
    validation_data=test_batches, validation_steps=10
)
print(pd.DataFrame(history.history))

history_dict = history.history

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss=history_dict['loss']
val_loss=history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12,9))
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(12,9))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim((0.5,1))
plt.show()
# 检索嵌入层

check_embedding_layer = model.layers[0]
# print(model.layers)
weights = check_embedding_layer.get_weights()
# print(weights, weights[0])
"""

# 文本 RNN(循环神经网络)进行文本分类
"""
# 循环神经网络和普通网络的显著区别
# 普通网络层与层之间是独立 即该隐藏层的输出是下一隐藏层层输入 例如DNN、CNN(又被统称为前向传播网络)
# 循环神经网络则在隐藏层之间增加了互联即最终的输出与每一层之间的输出都是相关 而前向传播网络则是
# 该层的输出出现就不在参与接下来层的处理因此这是与RNN最大的区别
datas, info = tfds.load(
    'imdb_reviews/subwords8k',
    with_info=True,
    as_supervised=True
)
train_datas, test_datas = datas['train'], datas['test']
'''
(train_data, test_data), info = tfds.load(
    'imdb_reviews/subwords8k', 
    split = (tfds.Split.TRAIN, tfds.Split.TEST), 
    with_info=True, as_supervised=True)
'''

encoder = info.features['text'].encoder
# encoder.encode() 是将单词转为编码 .decoder()是将编码转为单词

test_string = 'hellow word TensorFlow'
encoder_string = encoder.encode(test_string)
# print(encoder_string)
# print(encoder.decode(encoder_string))

encoder_size = encoder.vocab_size
print("lenth of encoder: ", encoder_size)

# for i in encoder_string:
#     print(i, encoder.decode([i]))
print(train_datas)

Buffer_size = 1000
Batch_size = 64
train_datas = train_datas.shuffle(Buffer_size).padded_batch(
    batch_size=Batch_size, padded_shapes=((None,), ())
)
test_datas = test_datas.padded_batch(
    batch_size=Batch_size, padded_shapes=((None,), ())
)

# 创建模型
# 在训练过程中具有相似含义的单词将会使用同一个向量

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder_size, Batch_size),
    # 嵌入层每个单词存储一个向量 随着训练的过程部分含义相似单词会使用同一个向量
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    # tf.keras.layers.Bidirectional 双向RNN 所谓双向RNN就是不仅仅有前向传播过长
    # 同时还有后向传播过程在作用
    # LSTM 用于处理梯度消失现象(在训练过程中传播信号几近为零的现象)
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(1e-4),
    metrics=['accuracy']
)
history = model.fit(
    train_datas, epochs=10,
    validation_data=test_datas,
    validation_steps=10
)
print(pd.DataFrame(history.history))

print(model.evaluate(test_datas))

def padd_to_size(vec, size):
    zeros = [0] * (size - len(vec))
    vec.extend(zeros)
    # 相当于将数据用0填充为长度为size的列表
    return vec
def predict_test(test_text, pred):
    test_text = encoder.encode(test_text)

    if pred:
        test_text = padd_to_size(test_text, 64)
    test_text = tf.cast(test_text, tf.float32)
    predictions = model.predict(tf.expand_dims(test_text, 0)) # 在axis=0处添加一个维度
    # t2.shape = [2, 3, 5]
    # shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
    # shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]
    # shape(expand_dims(t2, 3)) ==> [2, 3, 5, 1]
    return predictions

# 堆叠更多的LSTM层

model_new = tf.keras.Sequential([
    tf.keras.layers.Embedding(),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM()),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM()),
    ...
])
"""

# 文本 RNN生成文本
"""
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

unique_words = sorted(set(text))
unique_words_numpy = np.array(unique_words)
# 转成numpy的原因是numpy可以直接把一个numpy数列 直接转换比如说x = [1, 2, 3, 4, 5]
# 直接放入到unique_words_numpy[x.numpy()]可以直接获取对应的结果
# 而列表则必须一个一个来
word_num_table = {word:index for index,word in enumerate(unique_words)}
# print(word_num_table)
text_as_nums = np.array([word_num_table[word] for word in text])
# print(repr(text[:13]), text_as_nums[:13])

# 要预测下个单词最有可能能是什么 首先需要确保序列长度一致

sequence_lenth = 100
per_epoch_text = len(text) // sequence_lenth

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_nums)
# 把文本向量流转化成字符索引流
# print(pd.DataFrame(char_dataset)) 他把所有的句子拆分每个行的下一行是它的label
# 比如说一个句子 "hello word"它会将其转成
#           0
# 0         h
# 1         e
# 2         l
# ..
# 5         /n
# ...
# 8         r
# 9         d

# for i in char_dataset.take(5):
#     print(unique_words[i.numpy()])

sequence_text = char_dataset.batch(sequence_lenth+1, drop_remainder=True)
# drop_remainder=True 是否删除剩余项

# for i in sequence_text.take(1):
#     print(repr(''.join(unique_words_numpy[i.numpy()])))

# 创建滞后序列
def lag_text_sequence(text_batch):
    text_train = text_batch[:-1]
    text_label = text_batch[1:]
    return text_train, text_label

train_datas = sequence_text.map(lag_text_sequence)

# for text_train_example, text_label_example in train_datas.take(1):
#     for i, (input_idx, target_idx) in enumerate(zip(text_train_example[:5], text_label_example[:5])):
#         print("Step {:4d}".format(i))
#         print("  input: {} ({:s})".format(input_idx, repr(unique_words_numpy[input_idx])))
#         print("  expected output: {} ({:s})".format(target_idx, repr(unique_words_numpy[target_idx])))

Batch_size = 64
Buffer_size = 1000
train_datas = train_datas.shuffle(Buffer_size).batch(Batch_size, drop_remainder=True)

print(train_datas)

# 创建模型
vocab_size = len(unique_words)
embedding_dim = 256
run_units = 1024
def build_model(vocab_size, embedding_dim, run_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape = [batch_size, None]),
        # 初始输入形状: (Batch_size,)-> (Batch_size, embedding_size(也就是embedding_dim))
        # -> 然后是经过GRU(Batch_size, run_units)->Dense层(Batch_size, vocab_size)
        tf.keras.layers.GRU(run_units,
                            return_sequences=True,
                            # False:则返回输出序列的最后一个值
                            # True:则返回完整序列
                            stateful=True,
                            # 将当前索引中的结束状态作下批索引中的初始化状态 非常适合滞后数据的处理
                            recurrent_initializer='glorot_uniform' # 权重矩阵的初始化
                            ),
        # 该RNN层配合滞后处理或者说应用于时间序列的数据简直完美
        # GRU 基于可用的硬件 选择最佳处理方案 常用于配合GPU使用
        tf.keras.layers.Dense(vocab_size)
    ])
    return model
model = build_model(vocab_size, embedding_dim, run_units, Batch_size)

train_batch, train_batch_label = next(iter(train_datas))
print(train_batch_label.shape)
result = model(train_batch)
# print(result.shape) (64, 100, 65) (Batch_size, sequence_lenth, vocab_size)
# 这里的Sequence指的是数据的长度即一条有100个单词(包含0)的句子
# 它是64个一组 每个句子100个单词 每个单词产生65(vocab_size)个预测值

# 现在看下测试结果
# for i in np.argmax(result, axis=-1):
#     print(unique_words_numpy[i])
# 显然没有经过训练的模型 仅仅只是能做到按照预期输出的形式处理数据
# 按照列的形式看, 即每行的第1, 2, 3, 4, 5...列组成一个句子
# 原因是from_tensor_slices 将数据变成的格式

# print(np.argmax(result[0], axis=-1))
sampled_indices = tf.random.categorical(logits=result[0], num_samples=1)
# print(sampled_indices)
# 每行随机抽取num_samples个数据的下标 绘制的形状是[logits.shape[0], num_samples]
# 他会偏向于抽取值大的那个数据的下标 比如说一列数据[100, 1, 2, 3]
# 那么很大的可能抽取n次的结果是[0, 0, 0, 0, 0, 0, 0...] 可能抽取200次有190次都是0
# 因为尽管100在这四个值的和中占去的比重很大但仍然不是百分之100的因此也有可能抽取到1, 2, 3
# 优点是相比较argmax 随机抽取占用的资源更少效率得到大大提高这点可以参照
# 梯度下降算法和随机梯度下降算法的效率 缺点是有存在误差的可能性(但是没有什么数据是完美的
# 就算是人打字也有可能出现错别字)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
# print(''.join(unique_words_numpy[sampled_indices]))

# 配置模型
def loss(labels, predictions):
    return tf.keras.losses.\
        sparse_categorical_crossentropy(labels, predictions,
                                        from_logits=True # 逻辑回归模型
                                        )
# print(loss(train_batch_label, result).numpy().mean()) 测试一下
model.compile(
    loss=loss,
    optimizer='adam'
)

checkpoint_dir = deep_path + 'training_checkpoint\\ckpt_{epoch}'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_dir,
    save_weights_only = True
)

Epochs = 10
history = model.fit(
    train_datas, epochs=Epochs, callbacks=[checkpoint_callback]
)
print(pd.DataFrame(history.history))
# 可以看到检查点已经保存

# 不加载模型 加载训练权重
# new_model = build_model(vocab_size, embedding_dim, run_units, Batch_size)
# new_model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
# new_model.build(tf.TensorShape([1, None]))

# 循环预测
# 因为要不断的预测接下来的单词是什么所以需要确保其一直循环
# 过程如下:
# 首先设置起始字符串，初始化 RNN 状态并设置要生成的字符个数。
# 用起始字符串和 RNN 状态，获取下一个字符的预测分布。
# 然后，模型预测产生预测字符。把这个预测字符当作模型的下一个输入。
# 模型返回的RNN状态被输送回模型-更新模型。现在，模型有更多上下文可以学习，
# 而非只有一个字符。在预测出下一个字符后，更改过的 RNN 状态被再次输送回模型。
# 通过不断从前面预测的字符获得更多上下文，进行学习

def loop_model_fit(model, start_string):
    # 要预测的字符数目
    predict_words_nums = 100
    # 起始字符
    start_string = start_string
    # 数字化字符串
    input_eval = [word_num_table[i] for i in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    # tf.expand_dims() 在axis出增加一维

    text_result = [] # 储存结果
    temperature = 1.0
    model.reset_states()
    # 每个训练周期初始化隐藏层状态
    for i in range(predict_words_nums):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0) # 删除增加的维度

        # 用分类分布预测模型返回的字符
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        # 这里的[-1, 0](或者说[1, 0]) 可以看做是[0, 0]的预期输出
        # 因为循环模型的下一步是将预期输出当做下一轮模型的输入的因此需要做这样的操作
        # 把预测字符和前面的隐藏状态一起传递给模型作为下一个输入
        input_eval = tf.expand_dims([predicted_id], 0)

        text_result.append(unique_words_numpy[predicted_id])

    return (start_string + ''.join(text_result))

# 自定义循环神经网络训练过程

new_model_2 = build_model(vocab_size, embedding_dim, run_units, Batch_size)
optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(model, inp, labels):
    with tf.GradientTape() as tp:
        predictions = model(inp)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                labels, predictions, from_logits=True
            )
        )
    grads = tp.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

Epochs = 10

for epoch in range(Epochs):
    start = time.time()

    hidden = new_model_2.reset_states()
    for batch_n, (input, target) in enumerate(train_datas):
        loss = train_step(new_model_2, input, target)

        if batch_n % 100 == 0:
            template = 'Epoch {} Batch {} Loss {}'
            print(template.format(epoch + 1, batch_n, loss))

    if (epoch + 1) == 5:
        new_model_2.save_weights(checkpoint_dir.format(epoch))

new_model_2.save_weights(checkpoint_dir.format(epoch))"""

# 机器翻译
import unicodedata
# 用于查找对应字符数据库所设置的函数
# print(unicodedata.lookup('LEFT CURLY BRACKET')) 返回 '{'

path_to_zip = tf.keras.utils.get_file(
    'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
    extract=True)
path_to_file = os.path.dirname(path_to_zip) + '/spa-eng/spa.txt'

# datas = open(path_to_file, encoding='UTF-8').read().strip()
# print(datas) # 数据的大致形式

def unicode_to_ascii(unicode_str):
    return ''.join(c for c in unicodedata.normalize('NFD', unicode_str)
                   # unicodedata.normalizet() 返回正常形式unicode的字符串 unistr (NFD是其中一个格式类)
                   if unicodedata.category(c) != 'Mn')
    # unicodedata.category(c) 以字符串形式返回分配给Unicode字符unichr的常规类别
def process_ascii(unicode_str):
    ascii_str = unicode_to_ascii(unicode_str.lower().strip())
    # 将字符串转成小写.lower()并去除头尾的空格或换行符.strip()

    ascii_str = re.sub(pattern=r"([?.!,?])", repl=r" \1", string=ascii_str)
    # 正则表达式中的替换 相比较于replace() 能将指定类型或者指定类型之外的字符串全部替换
    # 在字符前面加^表示除了指定类型之外的所有字符被替换
    # repl=r'\1' 表示被替换pattern中指定的字符本身
    # 在本例中将pattern表示为pattern的字符本身在前面加个空格
    # 'hello?'-> 'hello ?'
    ascii_str = re.sub(pattern=r"[' ']+", repl=" ", string=ascii_str)
    # 将过长的空格转为一个空格
    ascii_str = re.sub(pattern=r"[^a-zA-Z?.!,?]+", repl=" ", string=ascii_str)
    # 除了a-zA-Z和?.!,?这几种字符外其他的特殊字符全转为空格
    ascii_str = ascii_str.rstrip().strip()
    # rstrip()删除字符串尾的空格strip()删除头尾两部的空格和换行符
    # 因为存在部分句子尾部有空格和换行符两个字符紧挨着因此使用两个功能相近的函数确保准确处理

    # 人工给句子加上开始和结束标志
    ascii_str = "<start> " + ascii_str + " <end>"
    return ascii_str
def creat_datas(datas_path, num_example):
    datas = open(datas_path, encoding='UTF-8').read().strip().split('\n')
    datas = [[process_ascii(line) for line in lines.split('\t')] for lines in datas[:num_example]]
    return zip(*datas)

def max_lenth(tensor):
    return max(len(t) for t in tensor)

def tokenize(lines):
    lines_tokenize = tf.keras.preprocessing.text.Tokenizer(
        filters=''
    )
    lines_tokenize.fit_on_texts(lines)
    # 将句子转成向量矩阵
    tensor = lines_tokenize.texts_to_sequences(lines)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(
        tensor, padding='post'
    )
    return tensor, lines_tokenize

def load_datas(path, num_examples=None):
    english_lines, spanish_lines = creat_datas(path, num_examples)
    input_tensor, inp_lang_tokenizer = tokenize(spanish_lines)
    target_tensor, targ_lang_tokenizer = tokenize(english_lines)
    # 虽然返回的是[eng, spa] 但是我们要做的是翻译西班牙语因此 西班牙语是输入 英语是标签
    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

input_tensor, target_tensor, inp_language, targ_language = load_datas(path_to_file, 30000)
max_length_targ, max_length_inp = max_lenth(target_tensor), max_lenth(input_tensor)
input_tensor_train, input_tensor_val, \
target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

# 显示长度
print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))

def show_datas(language, tensor):
    # language里面放的是类似于字典类型的  数字:数字代表的单词
    for t in tensor:
        if t!=0:
            # 0 指的是填充长度的0没有实际意义
            print("number: {}, words deputy by number: {} ".format(
                t, language.index_word[t]
            ))

# show_datas(inp_language, input_tensor_train[0])
# show_datas(targ_language, target_tensor_train[0])

# tf.data() 数据处理
Batch_size = 64
Buffer_size = len(input_tensor_train)
steps_per_epoch = Buffer_size // Batch_size
Embedding_dim = 256 # 嵌入层列的长度 也就是单个tensor的长度
units = 1024 # 输出最后为units * 1的向量
vocab_inp_size = len(inp_language.word_index) + 1 # 输入词汇表的长度
vocab_target_size = len(targ_language.word_index) + 1 # 标签词汇表的长度

train_datas = tf.data.Dataset.from_tensor_slices(
    (input_tensor_train, target_tensor_train)).shuffle(Buffer_size).batch(
    Batch_size, drop_remainder=True) # 删除多余的部分
train_example, train_example_labels = next(iter(train_datas))


# 编码器
# 编码器模型为我们提供形状为 (批大小，最大长度，隐藏层大小) 的编码器输出
# 这在使用RNN生成文本那一节力可以看出最终模型输出
# (Batch_size, sequence_lenth, dense_size(这里指的是dense(dense_size)))

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.embedding_dim = embedding_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.GRU = tf.keras.layers.GRU(
            self.enc_units,
            return_sequences=True, # 是返回输出序列中的最后一个输出还是完整序列 默认False这里指的是完整序列
            return_state=True, # 除输出外 是否返回最后一个状态 默认False这里指的是返回
            recurrent_initializer='glorot_uniform'
        )
    def call(self, x, hidden):
        x = self.embedding(x)
        outputs, state = self.GRU(x, initial_state =hidden)
        return outputs, state
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.enc_units))

encoder = Encoder(vocab_inp_size, Embedding_dim, units, Batch_size)
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(train_example, sample_hidden)
print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))
print(sample_output)
