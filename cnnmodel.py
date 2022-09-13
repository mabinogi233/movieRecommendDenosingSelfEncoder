
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import Row
from pyspark.sql import SQLContext

import params

import os
import sys

import numpy as np

import keras
from keras import losses

from elephas.spark_model import SparkModel

from elephas.utils.rdd_utils import to_simple_rdd

import get_data

# Path for spark source folder
os.environ['SPARK_HOME'] = r"D:\spark-3.2.0-bin-hadoop2.7\spark-3.2.0-bin-hadoop2.7"
os.environ['HADOOP_HOME'] = r"D:\hadoop-2.7.7\hadoop-2.7.7"
os.environ['JAVA_HOME'] = r"D:\jdk8"
os.environ['PYSPARK_PYTHON'] = "D:\Anaconda\python.exe"
os.environ['PYSPARK_DRIVER_PYTHON'] = "D:\Anaconda\python.exe"

# Append pyspark to Python Path
sys.path.append(r"D:\spark-3.2.0-bin-hadoop2.7\spark-3.2.0-bin-hadoop2.7\bin")
sys.path.append(r"D:\hadoop-2.7.7\hadoop-2.7.7\bin")
sys.path.append(r"D:\spark-3.2.0-bin-hadoop2.7\spark-3.2.0-bin-hadoop2.7\python")
sys.path.append(r"D:\spark-3.2.0-bin-hadoop2.7\spark-3.2.0-bin-hadoop2.7\python\lib\py4j-0.10.9.2-src.zip")

#模型构建
def build_keras_model():
    #模型构建
    # 编码器
    inputs = keras.layers.Input(shape=(1,params.input_shape,1))
    inputs_dp = keras.layers.Dropout(0.2)(inputs)
    #2次2D卷积
    ConvD_1 = keras.layers.Conv2D(params.latent_dim,(1,32),activation='relu',padding = 'same')(inputs_dp)
    MaxPoolD_1 = keras.layers.MaxPool2D((1,4),padding = 'same')(ConvD_1)
    ConvD_2 = keras.layers.Conv2D(params.latent_dim//2, (1,16),activation='relu',padding = 'same')(MaxPoolD_1)
    MaxPoolD_2 = keras.layers.MaxPool2D((1,4), padding= 'same')(ConvD_2)

    # 解码器
    #2次卷积2次整形
    ConvD_3 = keras.layers.Conv2D(params.latent_dim//2, (1,16),activation='relu',padding = 'same')(MaxPoolD_2)
    UpSamplingD_1 = keras.layers.UpSampling2D((1,4))(ConvD_3)
    ConvD_4 = keras.layers.Conv2D(params.latent_dim,(1,32),activation='relu',padding = 'same')(UpSamplingD_1)
    UpSamplingD_2 = keras.layers.UpSampling2D((1,4))(ConvD_4)
    #卷积解码
    decoded = keras.layers.Conv2D(1,(1,32),activation='sigmoid',padding = 'same')(UpSamplingD_2)
    #AE
    autoencoder = keras.Model(inputs, decoded, name='autoencoder')
    #输出摘要
    autoencoder.summary()
    #损失采用交叉熵，优化器采用随机梯度下降，评估指标采用kl散度与均方误差
    autoencoder.compile(
        loss=losses.binary_crossentropy,
        optimizer=keras.optimizers.SGD(lr=params.lr, momentum=0.0, decay=0.0, nesterov=False),
        metrics=[keras.metrics.kullback_leibler_divergence,keras.metrics.mean_squared_error]
        )

    return autoencoder

#保存weights
def save_model(model,tag):
    model.save_weights(params.model_weight_file + os.sep + tag + "_checkpoint")

#加载weights
def load_model(model,tag):
    model.load_weights(params.model_weight_file + os.sep + tag + "_checkpoint")





